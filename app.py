import os
from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from typing import List
import requests
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate,PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import create_retrieval_chain

# ------------------ Config ------------------
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_MODEL = "all-MiniLM-L12-v2"
GROQ_MODEL = "llama-3.1-8b-instant"

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
template = """Answer the question based on the provided context.
    <context>
    {context}
    </context>

    Question: {input}

    Respond concisely and with reasoning."""
prompt=ChatPromptTemplate.from_template(template)

def embed_docs(docs):
    chunks = splitter.split_documents(docs)
    db = FAISS.from_documents(chunks, embeddings)
    return db.as_retriever()

def answer_query(question: str, retriever):
    document_chain=create_stuff_documents_chain(llm,prompt)
    retriever_chain=create_retrieval_chain(retriever,document_chain)

    return retriever_chain.invoke({ "input": question})['answer']

# ------------------ Endpoint ------------------
@app.post("/api/v1/hackrx/run", response_model=QueryResponse)
async def run_query(req: QueryRequest):

    file_path = download_file(req.documents)
    docs = load_docs(file_path)
    retriever = embed_docs(docs)
    answers = [answer_query(q, retriever) for q in req.questions]
    os.remove(file_path)  # Clean up the temporary file

    return {"answers": answers}
