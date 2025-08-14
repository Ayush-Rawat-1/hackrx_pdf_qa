import os
import asyncio
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import requests
import tempfile
from concurrent.futures import ThreadPoolExecutor

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# ------------------ Config ------------------
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_MODEL = "all-MiniLM-L12-v2"
GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# ------------------ FastAPI ------------------
app = FastAPI()

# ------------------ Request & Response ------------------
class QueryRequest(BaseModel):
    documents: str  # Blob URL
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

# ------------------ Core Pipeline ------------------
def download_file(url: str) -> str:
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception("Document download failed")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    tmp.write(response.content)
    tmp.close()
    return tmp.name

def load_docs(file_path: str):
    loader = PyPDFLoader(file_path)
    return loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
embeddings = HuggingFaceEmbeddings(model=HF_MODEL)
llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name=GROQ_MODEL)

template = """
You are an assistant for question answering.
Use the following pieces of retrieved context to answer the question.
<context>
{context}
</context>
Understand the context and answer the question in simple terms.
Your answer must:
- Be concise (no more than 2 sentences).
- Contain plain text only (no Markdown formatting like **bold**, *italic*, `code`, or bullet points).
- Avoid line breaks or newlines — respond in a single line.
Question: {input}
"""
prompt = ChatPromptTemplate.from_template(template)

def embed_docs(docs):
    chunks = splitter.split_documents(docs)
    db = FAISS.from_documents(chunks, embeddings)
    return db.as_retriever()

def answer_query(question: str, retriever):
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever_chain = create_retrieval_chain(retriever, document_chain)
    return retriever_chain.invoke({"input": question})['answer']

# ------------------ Parallel Execution ------------------
async def run_queries_parallel(questions: List[str], retriever):
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor(max_workers=len(questions)) as executor:
        tasks = [
            loop.run_in_executor(executor, answer_query, q, retriever)
            for q in questions
        ]
        return await asyncio.gather(*tasks)

# ------------------ Endpoint ------------------
@app.get("/")
async def home():
    return {"message": "FastAPI RAG Engine is running!"}
    
@app.post("/api/v1/hackrx/run", response_model=QueryResponse)
async def run_query(req: QueryRequest):
    file_path = download_file(req.documents)
    try:
        docs = load_docs(file_path)
        retriever = embed_docs(docs)
        answers = await run_queries_parallel(req.questions, retriever)
        return {"answers": answers}
    finally:
        os.remove(file_path)  # Clean up the temporary file
