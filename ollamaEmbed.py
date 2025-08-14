"""
FastAPI service that:
1. Downloads a PDF from a signed URL
2. Embeds it with Ollama (nomic-embed-text)
3. Answers a batch of questions with llama-3.3-70b-versatile
All heavy work runs in thread-pools so the async loop stays responsive.
"""

import os
from dotenv import load_dotenv
load_dotenv()
import asyncio
import tempfile
from typing import List

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain

# ----------------- CONFIG -----------------
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"
OLLAMA_EMBED_MODEL = "nomic-embed-text:v1.5"

# ----------------- FASTAPI -----------------
app = FastAPI(title="HackRx PDF-QA API", version="1.0")

class QueryRequest(BaseModel):
    documents: str      # signed URL to PDF
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

# ----------------- LANGCHAIN SETUP -----------------
embeddings = OllamaEmbeddings(model=OLLAMA_EMBED_MODEL)
llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name=GROQ_MODEL)

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

template = """
You are an assistant for question answering.
Use the following pieces of retrieved context to answer the question.
<context> {context} </context>
Understand the context and answer the question in simple terms.
Your answer must:
- Be concise.
- Contain plain text only (no Markdown formatting like **bold**, *italic*, `code`, or bullet points).
- Avoid line breaks or newlines — respond in a single line.
Question: {question}
"""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=template
)

chain = load_summarize_chain(
    llm,
    chain_type="stuff",
    prompt=prompt,
    document_variable_name="context",
    verbose=False,
)

# ----------------- HELPER UTILS -----------------
async def download_file(url: str) -> str:
    """Async download → temp PDF file."""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(url)
            resp.raise_for_status()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Download failed: {e}")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    tmp.write(resp.content)
    tmp.close()
    return tmp.name

def load_docs(path: str):
    """Sync PDF → Document list."""
    loader = PyPDFLoader(path)
    return loader.load()

# ----------------- ASYNC PIPELINE -----------------
async def embed_docs_async(docs):
    """Embeds docs in a thread so the loop is not blocked."""
    chunks = splitter.split_documents(docs)
    # FAISS.from_documents is CPU-bound; run in default thread pool
    loop = asyncio.get_event_loop()
    db = await loop.run_in_executor(None, FAISS.from_documents, chunks, embeddings)
    return db.as_retriever()

async def answer_query_async(question: str, retriever):
    """Retrieve + LLM call wrapped in threads."""
    docs = retriever.invoke(question)          # vector lookup (cheap)
    loop = asyncio.get_event_loop()
    out = await loop.run_in_executor(
        None,
        chain.invoke,
        {"input_documents": docs, "question": question},
    )
    return out["output_text"]

async def run_queries_parallel(questions: List[str], retriever):
    """Fire all questions concurrently."""
    tasks = [answer_query_async(q, retriever) for q in questions]
    return await asyncio.gather(*tasks)

# ----------------- ENDPOINT -----------------
@app.post("/api/v1/hackrx/run", response_model=QueryResponse)
async def run_query(req: QueryRequest):
    file_path = None
    try:
        file_path = await download_file(req.documents)
        docs = load_docs(file_path)          # parsing is still sync
        retriever = await embed_docs_async(docs)
        answers = await run_queries_parallel(req.questions, retriever)
        return {"answers": answers}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
