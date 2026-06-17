import argparse
import sys
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from src.ingest import ingest_documents
from src.rag import query_rag_system

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
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Resolve API Key from argument or environment variables
    api_key = args.api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Error: Gemini API key is not set.", file=sys.stderr)
        print("Please provide it via --api-key, a local .env file, or set GEMINI_API_KEY / GOOGLE_API_KEY environment variable.", file=sys.stderr)
        print("Example:", file=sys.stderr)
        print("  Windows (PowerShell): $env:GEMINI_API_KEY='your-key-here'", file=sys.stderr)
        print("  Windows (CMD):        set GEMINI_API_KEY=your-key-here", file=sys.stderr)
        print("  Linux/macOS:          export GEMINI_API_KEY='your-key-here'", file=sys.stderr)
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
                api_key=api_key
            )
        except Exception as e:
            print(f"\nIngestion failed with error: {e}", file=sys.stderr)
            sys.exit(1)
            
    elif args.query:
        if not os.path.exists(args.db_path):
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
                api_key=api_key
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

if __name__ == "__main__":
    main()
