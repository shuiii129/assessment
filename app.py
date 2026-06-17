import os
import sys
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import chromadb
import uvicorn

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from src.rag import query_rag_system

# Initialize FastAPI application
app = FastAPI(
    title="Local RAG System API",
    description="Backend API for local AI Governance RAG system.",
    version="1.0.0"
)

# API Schemas
class QueryRequest(BaseModel):
    query: str
    model: str = "gemini-2.5-flash"
    k: int = 4

class QueryResponse(BaseModel):
    answer: str
    references: list[dict]

class StatusResponse(BaseModel):
    connected: bool
    indexed_chunks: int
    db_path: str
    message: str

# Resolve database path dynamically relative to this file
current_dir = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(current_dir, "chroma_db")

# API Endpoints
@app.post("/api/query", response_model=QueryResponse)
def handle_query(request: QueryRequest):
    """
    Executes a semantic search query against ChromaDB and synthesizes 
    a grounded response via Gemini.
    """
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500, 
            detail="Gemini API Key is not set in the environment. Please configure it in your .env file."
        )
        
    if not os.path.exists(db_path):
        raise HTTPException(
            status_code=404,
            detail="Vector database not found. Please run document ingestion first using the CLI: `python main.py --ingest`"
        )
        
    try:
        answer, references = query_rag_system(
            query_text=request.query,
            k=request.k,
            db_path=db_path,
            model=request.model,
            api_key=api_key
        )
        return QueryResponse(answer=answer, references=references)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/status", response_model=StatusResponse)
def handle_status():
    """
    Retrieves the status of the local vector store database.
    """
    if not os.path.exists(db_path):
        return StatusResponse(
            connected=False,
            indexed_chunks=0,
            db_path=db_path,
            message="Vector database not initialized. Please run ingestion."
        )
        
    try:
        chroma_client = chromadb.PersistentClient(path=db_path)
        # Try to get existing collection
        try:
            collection = chroma_client.get_collection(name="documents")
            count = collection.count()
            return StatusResponse(
                connected=True,
                indexed_chunks=count,
                db_path=db_path,
                message="Vector store successfully loaded and ready."
            )
        except Exception:
            return StatusResponse(
                connected=True,
                indexed_chunks=0,
                db_path=db_path,
                message="Vector store connected but collection 'documents' is empty."
            )
    except Exception as e:
        return StatusResponse(
            connected=False,
            indexed_chunks=0,
            db_path=db_path,
            message=f"Error connecting to vector store: {str(e)}"
        )

# Serve Frontend static assets
static_dir = os.path.join(current_dir, "public")
if os.path.exists(static_dir):
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
else:
    # If public dir is not created yet, register a fallback root endpoint
    @app.get("/")
    def read_root():
        return {"status": "backend operational", "message": "Please create public/ directory containing frontend assets."}

if __name__ == "__main__":
    # Check for API Key warning
    if not os.environ.get("GEMINI_API_KEY") and not os.environ.get("GOOGLE_API_KEY"):
        print("Warning: Neither GEMINI_API_KEY nor GOOGLE_API_KEY environment variable is set.")
        print("Queries will fail until a key is configured in your .env or shell environment.\n")
        
    print("Starting Web Server at http://127.0.0.1:8000 ...")
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
