import argparse
import os
import sys
import uvicorn
import chromadb
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from src.ingest import ingest_documents
from src.rag import query_rag_system

# Initialize FastAPI App (Vercel Serverless Function entry point)
app = FastAPI(
    title="Local RAG System API",
    description="Vercel-optimized routing API for local AI Governance RAG system.",
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

# API Routing Endpoints
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
            detail="Gemini API Key is not set in the environment. Please configure it in your Vercel/environment variables."
        )
        
    current_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(current_dir, "chroma_db")
    
    # Check if database index exists. If not, use Ephemeral (fallback)
    is_ephemeral = not os.path.exists(db_path)
        
    try:
        answer, references = query_rag_system(
            query_text=request.query,
            k=request.k,
            db_path=db_path,
            model=request.model,
            api_key=api_key,
            ephemeral=is_ephemeral
        )
        return QueryResponse(answer=answer, references=references)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/status", response_model=StatusResponse)
def handle_status():
    """
    Retrieves the status of the local vector store database.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(current_dir, "chroma_db")
    resolved_db_path = os.path.abspath(db_path)
    
    # Check Vercel environment
    is_vercel = os.environ.get("VERCEL") == "1"
    effective_db_path = resolved_db_path
    
    if is_vercel:
        effective_db_path = "/tmp/chroma_db"
        if not os.path.exists(effective_db_path) and os.path.exists(resolved_db_path):
            import shutil
            try:
                shutil.copytree(resolved_db_path, effective_db_path)
            except Exception as e:
                return StatusResponse(
                    connected=False,
                    indexed_chunks=0,
                    db_path=resolved_db_path,
                    message=f"Failed to copy database to /tmp: {str(e)}"
                )
    
    if not os.path.exists(effective_db_path):
        return StatusResponse(
            connected=False,
            indexed_chunks=0,
            db_path=resolved_db_path,
            message="Vector database not found. Ephemeral mock fallback mode active."
        )
        
    try:
        chroma_client = chromadb.PersistentClient(path=effective_db_path)
        try:
            collection = chroma_client.get_collection(name="documents")
            count = collection.count()
            return StatusResponse(
                connected=True,
                indexed_chunks=count,
                db_path=resolved_db_path,
                message="Vector store successfully loaded and ready."
            )
        except Exception:
            return StatusResponse(
                connected=True,
                indexed_chunks=0,
                db_path=resolved_db_path,
                message="Vector store connected but collection 'documents' is empty."
            )
    except Exception as e:
        return StatusResponse(
            connected=False,
            indexed_chunks=0,
            db_path=resolved_db_path,
            message=f"Error connecting to vector store: {str(e)}"
        )

# Serve Frontend static assets from public/ folder
current_dir = os.path.dirname(os.path.abspath(__file__))
public_dir = os.path.join(current_dir, "public")
if os.path.exists(public_dir):
    app.mount("/", StaticFiles(directory=public_dir, html=True), name="public")
else:
    @app.get("/")
    def read_root():
        return {"status": "backend operational", "message": "Please ensure public/ directory contains index.html"}

# CLI Parse arguments
def parse_args():
    parser = argparse.ArgumentParser(
        description="Production-Ready Retrieval-Augmented Generation (RAG) System for AI Governance documents."
    )
    
    # Ingestion group
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--ingest", 
        action="store_true",
        help="Run the ingestion pipeline to parse, chunk, embed, and store documents in local ChromaDB."
    )
    group.add_argument(
        "--query", 
        type=str,
        help="The query string to search for and generate a grounded answer."
    )
    
    # Optional parameters
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of raw documents to index during ingestion (useful for fast testing/rate limits)."
    )
    parser.add_argument(
        "--k",
        type=int,
        default=4,
        help="Number of relevant text chunks to retrieve (default: 4)."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-2.5-flash",
        help="Gemini text generation model (default: gemini-2.5-flash)."
    )
    parser.add_argument(
        "--embed-model",
        type=str,
        default="gemini-embedding-001",
        help="Gemini embedding model (default: gemini-embedding-001)."
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="./chroma_db",
        help="Local path to persist ChromaDB index (default: ./chroma_db)."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory containing the raw dataset files (default: auto-detected 'Data' or 'data')."
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Google Gemini API Key (falls back to GEMINI_API_KEY / GOOGLE_API_KEY env variables)."
    )
    parser.add_argument(
        "--ephemeral",
        action="store_true",
        help="Force the vector database to run in-memory without persistent writes."
    )
    
    return parser.parse_args()

def cli_main():
    args = parse_args()
    
    # Resolve API Key from argument or environment variables
    api_key = args.api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Error: Gemini API key is not set.", file=sys.stderr)
        print("Please provide it via --api-key, a local .env file, or set GEMINI_API_KEY / GOOGLE_API_KEY environment variable.", file=sys.stderr)
        sys.exit(1)
        
    if args.ingest:
        print("======================================================================")
        print("                   STARTING DOCUMENT INGESTION                        ")
        print("======================================================================")
        try:
            ingest_documents(
                limit=args.limit,
                db_path=args.db_path,
                embed_model=args.embed_model,
                data_dir=args.data_dir,
                api_key=api_key,
                ephemeral=args.ephemeral
            )
        except Exception as e:
            print(f"\nIngestion failed with error: {e}", file=sys.stderr)
            sys.exit(1)
            
    elif args.query:
        # Check if database is initialized (unless ephemeral is forced)
        if not args.ephemeral and not os.path.exists(args.db_path):
            print(f"Error: Vector store directory '{args.db_path}' does not exist.", file=sys.stderr)
            print("Please run document ingestion first using:", file=sys.stderr)
            print("  python main.py --ingest --limit 10", file=sys.stderr)
            sys.exit(1)
            
        try:
            answer, references = query_rag_system(
                query_text=args.query,
                k=args.k,
                db_path=args.db_path,
                model=args.model,
                embed_model=args.embed_model,
                api_key=api_key,
                ephemeral=args.ephemeral
            )
            
            # Print answer block
            print("\n" + "=" * 70)
            print("                         GROUNDED ANSWER                              ")
            print("=" * 70)
            print(answer)
            print("=" * 70 + "\n")
            
            # Print references block
            print("=" * 70)
            print("                     EXACT SOURCE REFERENCES                          ")
            print("=" * 70)
            
            if not references:
                print("No source references utilized.")
            else:
                for idx, ref in enumerate(references, start=1):
                    title = ref.get("title") or f"Document {ref.get('doc_id')}"
                    source = ref.get("source", "N/A")
                    link = ref.get("link", "")
                    authority = ref.get("authority", "N/A")
                    collections = ref.get("collections", "N/A")
                    chunk_idx = ref.get("chunk_index", 0)
                    
                    print(f"{idx}. {title} [File: {source}, Chunk: {chunk_idx}]")
                    print(f"   - Authority:   {authority}")
                    print(f"   - Collection:  {collections}")
                    if link:
                        print(f"   - Source Link: {link}")
                    print("-" * 50)
            print("=" * 70 + "\n")
            
        except Exception as e:
            print(f"\nQuery execution failed with error: {e}", file=sys.stderr)
            sys.exit(1)

# Helper for executing CLI vs Server mode
if __name__ == "__main__":
    # If running with CLI flags, execute command line script.
    # Otherwise, execute FastAPI server via Uvicorn.
    if len(sys.argv) > 1 and (sys.argv[1].startswith("-") or sys.argv[1] in ("--query", "--ingest")):
        cli_main()
    else:
        print("Starting Web Server at http://127.0.0.1:8000 ...")
        # Check API Key warning
        if not os.environ.get("GEMINI_API_KEY") and not os.environ.get("GOOGLE_API_KEY"):
            print("Warning: Neither GEMINI_API_KEY nor GOOGLE_API_KEY environment variable is set.")
            print("Queries will fail until a key is configured in your Vercel/environment variables.\n")
        uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
