import os
import re
import time
import random
import pandas as pd
import chromadb
from google import genai
from tqdm import tqdm

def clean_text(text: str) -> str:
    """
    Cleans raw document text, normalizing whitespace and line endings 
    while preserving structure (paragraphs and sections).
    """
    # Normalize line endings
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    # Replace multiple spaces/tabs with a single space
    text = re.sub(r'[ \t]+', ' ', text)
    # Replace multiple consecutive newlines with single newline
    text = re.sub(r'\n+', '\n', text)
    return text.strip()

def chunk_document(text: str, chunk_size: int = 800, chunk_overlap: int = 80) -> list[str]:
    """
    Splits text into chunks of roughly chunk_size characters with chunk_overlap.
    Attempts to align chunk boundaries with word or sentence breaks (spaces/newlines).
    """
    chunks = []
    if not text:
        return chunks
    
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = min(start + chunk_size, text_len)
        
        # If we are not at the end of the text, try to find a word boundary (space or newline)
        # to avoid splitting words. Lookback up to 10% of the chunk size.
        if end < text_len:
            lookback_limit = max(start, end - int(chunk_size * 0.1))
            boundary = -1
            for i in range(end - 1, lookback_limit - 1, -1):
                if text[i] in (' ', '\n'):
                    boundary = i
                    break
            if boundary != -1:
                end = boundary + 1
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
            
        start = end - chunk_overlap
        if start >= text_len or end >= text_len:
            break
            
    return chunks

def load_metadata(data_dir: str) -> dict:
    """
    Loads documents metadata from documents.csv and returns a mapping from AGORA ID to metadata.
    """
    csv_path = os.path.join(data_dir, "documents.csv")
    if not os.path.exists(csv_path):
        print(f"Warning: Metadata file not found at {csv_path}. Proceeding with empty metadata.")
        return {}
    
    try:
        # Load the CSV, specifying string dtype for safety on ID matching
        df = pd.read_csv(csv_path, dtype={"AGORA ID": str})
        
        # Replace NaN values with empty string or sensible defaults for clean indexing
        df = df.fillna("")
        
        metadata_map = {}
        for _, row in df.iterrows():
            agora_id = row.get("AGORA ID", "").strip()
            if not agora_id:
                continue
                
            metadata_map[agora_id] = {
                "title": str(row.get("Official name", "")).strip(),
                "casual_name": str(row.get("Casual name", "")).strip(),
                "link": str(row.get("Link to document", "")).strip(),
                "authority": str(row.get("Authority", "")).strip(),
                "collections": str(row.get("Collections", "")).strip(),
                "summary": str(row.get("Short summary", "")).strip(),
            }
        return metadata_map
    except Exception as e:
        print(f"Error reading metadata from {csv_path}: {e}")
        return {}

def get_embeddings_with_retry(client: genai.Client, texts: list[str], model: str = "gemini-embedding-001") -> list[list[float]]:
    """
    Generates text embeddings using Gemini API with batching and exponential backoff retry.
    Includes smart rate limit handling (sleeping for 60 seconds on 429/Resource Exhausted).
    """
    embeddings = []
    batch_size = 50 # Small batch size to manage rate limits and payload boundaries
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        
        max_retries = 5
        base_delay = 2.0
        
        for attempt in range(max_retries):
            try:
                response = client.models.embed_content(
                    model=model,
                    contents=batch
                )
                for emb in response.embeddings:
                    embeddings.append(emb.values)
                break
            except Exception as e:
                error_str = str(e).lower()
                is_rate_limit = "429" in error_str or "resource_exhausted" in error_str or "quota" in error_str
                
                if attempt == max_retries - 1:
                    raise e
                
                if is_rate_limit:
                    # Sleep for 60 seconds to reset the 1-minute RPM/TPM window
                    wait_time = 60.0 + random.uniform(0, 5)
                    print(f"\n[Rate Limit] Gemini API 429 Resource Exhausted. Sleeping {wait_time:.1f}s to reset window (Attempt {attempt+1}/{max_retries})...")
                    time.sleep(wait_time)
                else:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 0.5)
                    print(f"\n[Warning] Embedding API call failed: {e}. Retrying in {delay:.2f}s (Attempt {attempt+1}/{max_retries})...")
                    time.sleep(delay)
                
    return embeddings

def get_data_dir() -> str:
    """
    Finds the active data directory, handling case variations (Data/ vs data/).
    """
    for folder in ["Data", "data"]:
        if os.path.exists(folder):
            return folder
    # Fallback to current directory if not found
    return "Data"

def ingest_documents(
    limit: int = None, 
    chunk_size: int = 500, 
    chunk_overlap: int = 50, 
    db_path: str = "./chroma_db",
    embed_model: str = "gemini-embedding-001",
    data_dir: str = None,
    api_key: str = None,
    ephemeral: bool = False
):
    """
    Main ingestion function. Loads metadata, reads text files, chunks content,
    generates embeddings via Gemini, and indexes into ChromaDB.
    """
    if not data_dir:
        data_dir = get_data_dir()
        
    fulltext_dir = os.path.join(data_dir, "fulltext")
    if not os.path.exists(fulltext_dir):
        raise FileNotFoundError(f"Raw text documents folder not found at {fulltext_dir}")
        
    # Initialize Google GenAI client
    resolved_api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not resolved_api_key:
        raise ValueError("API Key is not set. Please set GEMINI_API_KEY / GOOGLE_API_KEY environment variable or pass --api-key.")
    
    client = genai.Client(api_key=resolved_api_key)
    
    # Load metadata mapping
    print("Loading document metadata...")
    metadata_map = load_metadata(data_dir)
    print(f"Loaded metadata for {len(metadata_map)} documents.")
    
    # Initialize ChromaDB client dynamically
    if ephemeral:
        chroma_client = chromadb.EphemeralClient()
    else:
        # Resolve path robustly relative to root
        current_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.dirname(current_dir)
        resolved_db_path = os.path.abspath(db_path) if os.path.isabs(db_path) else os.path.join(root_dir, db_path)
        
        # If we are on Vercel or ephemeral mode is forced via environment
        if os.environ.get("VERCEL") == "1" or os.environ.get("CHROMA_EPHEMERAL") == "true":
            if os.path.exists(resolved_db_path):
                if os.environ.get("VERCEL") == "1":
                    import shutil
                    tmp_db_path = "/tmp/chroma_db"
                    if not os.path.exists(tmp_db_path):
                        try:
                            shutil.copytree(resolved_db_path, tmp_db_path)
                        except Exception as e:
                            print(f"Warning: Failed to copy database to /tmp: {e}")
                            tmp_db_path = None
                    
                    if tmp_db_path:
                        chroma_client = chromadb.PersistentClient(path=tmp_db_path)
                    else:
                        chroma_client = chromadb.EphemeralClient()
                else:
                    chroma_client = chromadb.PersistentClient(path=resolved_db_path)
            else:
                chroma_client = chromadb.EphemeralClient()
        else:
            chroma_client = chromadb.PersistentClient(path=resolved_db_path)
            
    collection = chroma_client.get_or_create_collection(name="documents")
    
    # Check already ingested documents for incremental ingestion
    existing_sources = set()
    try:
        existing = collection.get(include=["metadatas"])
        if existing and existing["metadatas"]:
            for meta in existing["metadatas"]:
                if "source" in meta:
                    existing_sources.add(meta["source"])
        print(f"Found {len(existing_sources)} unique files already indexed in ChromaDB.")
    except Exception as e:
        print(f"Starting with a fresh index (or unable to check existing: {e})")
        
    # List all txt files
    txt_files = [f for f in os.listdir(fulltext_dir) if f.endswith(".txt")]
    
    # Filter out files that are already ingested
    new_files = [f for f in txt_files if f not in existing_sources]
    
    if not new_files:
        print("All documents are already indexed. Nothing to ingest.")
        return
        
    print(f"Found {len(txt_files)} total documents, {len(new_files)} new documents to index.")
    
    # Limit files for testing/limits if specified
    if limit is not None:
        new_files = new_files[:limit]
        print(f"Limiting ingestion to {limit} files as requested.")
        
    all_chunks = []
    all_metadatas = []
    all_ids = []
    
    print("Extracting and chunking documents...")
    for filename in tqdm(new_files, desc="Parsing files"):
        file_path = os.path.join(fulltext_dir, filename)
        doc_id = os.path.splitext(filename)[0]
        
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        except Exception as e:
            print(f"Failed to read {file_path}: {e}")
            continue
            
        cleaned_content = clean_text(content)
        chunks = chunk_document(cleaned_content, chunk_size, chunk_overlap)
        
        # Fetch associated metadata
        doc_meta = metadata_map.get(doc_id, {})
        title = doc_meta.get("title", f"Document {doc_id}")
        link = doc_meta.get("link", "")
        authority = doc_meta.get("authority", "")
        collections = doc_meta.get("collections", "")
        
        for idx, chunk_text in enumerate(chunks):
            chunk_metadata = {
                "doc_id": doc_id,
                "source": filename,
                "title": title,
                "link": link,
                "authority": authority,
                "collections": collections,
                "chunk_index": idx
            }
            all_chunks.append(chunk_text)
            all_metadatas.append(chunk_metadata)
            all_ids.append(f"{doc_id}_chunk_{idx}")
            
    if not all_chunks:
        print("No new chunks extracted.")
        return
        
    print(f"Extracted {len(all_chunks)} chunks from {len(new_files)} files.")
    
    # Index chunks in batches to avoid API/payload limits
    batch_size = 100
    print("Generating embeddings and writing to vector database...")
    for i in tqdm(range(0, len(all_chunks), batch_size), desc="Ingesting Batches"):
        batch_chunks = all_chunks[i:i + batch_size]
        batch_metadatas = all_metadatas[i:i + batch_size]
        batch_ids = all_ids[i:i + batch_size]
        
        try:
            batch_embeddings = get_embeddings_with_retry(client, batch_chunks, model=embed_model)
            collection.add(
                ids=batch_ids,
                embeddings=batch_embeddings,
                metadatas=batch_metadatas,
                documents=batch_chunks
            )
        except Exception as e:
            print(f"\nFatal error ingesting batch starting at index {i}: {e}")
            print("\nIngestion stopped early due to rate limits or API issues.")
            print("Since ingestion is incremental, you can simply run the command again to resume from where you left off.")
            return
            
    print("Ingestion completed successfully.")
