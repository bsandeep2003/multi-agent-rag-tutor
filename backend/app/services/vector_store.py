from __future__ import annotations

import re
import shutil
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = DATA_DIR / "chroma"

def get_chroma_client() -> chromadb.PersistentClient:
    """Gets or creates the persistent ChromaDB client."""
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_DIR))

def _get_collection_name(domain: str) -> str:
    """Converts a domain name into a clean, valid Chroma collection name."""
    clean_name = re.sub(r'[^a-zA-Z0-9_-]', '_', domain)
    clean_name = clean_name[:60]
    if len(clean_name) < 3:
        clean_name = f"domain_{clean_name}"
    return clean_name

async def index_document_chunks(
    document_id: str, 
    filename: str, 
    domain: str, 
    chunks: list
) -> int:
    """
    Indexes a list of chunks in ChromaDB.
    Chunks can be ChunkRecord model instances.
    """
    client = get_chroma_client()
    default_ef = embedding_functions.DefaultEmbeddingFunction()
    
    collection_name = _get_collection_name(domain)
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=default_ef
    )
    
    ids = []
    documents = []
    metadatas = []
    
    for chunk in chunks:
        ids.append(chunk.chunk_id)
        documents.append(chunk.text)
        
        # Build breadcrumb text if it exists in metadata
        headers = chunk.metadata.get("headers", []) if hasattr(chunk, "metadata") else []
        header_context = " > ".join(headers) if isinstance(headers, list) else str(headers)
        
        metadatas.append({
            "document_id": document_id,
            "filename": filename,
            "domain": domain,
            "chunk_index": int(chunk.index),
            "header_context": header_context,
            "start_char": int(chunk.start_char),
            "end_char": int(chunk.end_char)
        })
        
    # Add to Chroma in batches of 500 to prevent batch payload limits
    batch_size = 500
    for i in range(0, len(ids), batch_size):
        collection.add(
            ids=ids[i:i+batch_size],
            documents=documents[i:i+batch_size],
            metadatas=metadatas[i:i+batch_size]
        )
        
    return len(ids)

async def search_vector_store(
    query: str, 
    domain: str, 
    limit: int = 5
) -> list[dict]:
    """
    Queries ChromaDB for chunks matching the query text.
    """
    client = get_chroma_client()
    collection_name = _get_collection_name(domain)
    
    try:
        # If collection doesn't exist, we return empty results
        collection = client.get_collection(
            name=collection_name,
            embedding_function=embedding_functions.DefaultEmbeddingFunction()
        )
    except Exception:
        return []
        
    results = collection.query(
        query_texts=[query],
        n_results=limit
    )
    
    formatted_results = []
    if not results or not results["documents"]:
        return []
        
    # Format database results into standard list of dictionaries
    docs = results["documents"][0]
    ids = results["ids"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0] if "distances" in results and results["distances"] else [0.0] * len(docs)
    
    for idx in range(len(docs)):
        formatted_results.append({
            "chunk_id": ids[idx],
            "text": docs[idx],
            "metadata": metadatas[idx],
            "distance": float(distances[idx])
        })
        
    return formatted_results

async def delete_document_chunks(document_id: str, domain: str, chunk_ids: list[str] | None = None) -> bool:
    """
    Deletes all chunks belonging to a document from the domain collection.
    Either by specific chunk IDs or by metadata filter.
    """
    client = get_chroma_client()
    collection_name = _get_collection_name(domain)
    
    try:
        collection = client.get_collection(
            name=collection_name,
            embedding_function=embedding_functions.DefaultEmbeddingFunction()
        )
        
        # Method 1: Delete by IDs if provided (most reliable)
        if chunk_ids and len(chunk_ids) > 0:
            batch_size = 500
            for i in range(0, len(chunk_ids), batch_size):
                batch = chunk_ids[i:i+batch_size]
                collection.delete(ids=batch)
            return True
        
        # Method 2: Fallback to metadata filter with proper ChromaDB syntax
        # Use the $eq operator for metadata filtering
        collection.delete(where={"document_id": {"$eq": document_id}})
        return True
        
    except Exception as exc:
        # Log the error so we know what went wrong
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Failed to delete chunks for document {document_id}: {exc}")
        return False
