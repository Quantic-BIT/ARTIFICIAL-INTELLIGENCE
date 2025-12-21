"""
Vector Store Module

Handles storage and retrieval of document embeddings using ChromaDB.
"""
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings

from app.rag.embeddings import get_embedding_model
from app.rag.ingestion import Document, ingest_documents


class VectorStore:
    """ChromaDB-based vector store for document retrieval."""
    
    COLLECTION_NAME = "policy_documents"
    
    def __init__(self, persist_directory: str = None):
        if persist_directory is None:
            persist_directory = str(Path(__file__).parent.parent.parent / "data")
        
        self.persist_directory = persist_directory
        self.embedding_model = get_embedding_model()
        
        # Initialize ChromaDB with persistence
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"description": "Acme Corporation Policy Documents"}
        )
        
        print(f"Vector store initialized. Documents in collection: {self.collection.count()}")
    
    def add_documents(self, documents: List[Document]) -> int:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of Document objects to add
        
        Returns:
            Number of documents added
        """
        if not documents:
            return 0
        
        # Prepare data for ChromaDB
        texts = [doc.page_content for doc in documents]
        embeddings = self.embedding_model.embed_documents(texts)
        
        # Create unique IDs
        ids = [f"doc_{i}_{doc.metadata.get('source', 'unknown')}_{doc.metadata.get('chunk_id', 0)}" 
               for i, doc in enumerate(documents)]
        
        # Prepare metadata (ChromaDB requires simple types)
        metadatas = []
        for doc in documents:
            metadata = {
                "source": str(doc.metadata.get("source", "")),
                "title": str(doc.metadata.get("title", "")),
                "chunk_id": int(doc.metadata.get("chunk_id", 0)),
                "chunk_total": int(doc.metadata.get("chunk_total", 1))
            }
            metadatas.append(metadata)
        
        # Add to collection
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )
        
        print(f"Added {len(documents)} documents to vector store")
        return len(documents)
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar documents.
        
        Args:
            query: Search query text
            k: Number of results to return
        
        Returns:
            List of results with content, metadata, and similarity score
        """
        query_embedding = self.embedding_model.embed_query(query)
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"]
        )
        
        # Format results
        formatted_results = []
        
        if results and results['documents'] and results['documents'][0]:
            for i in range(len(results['documents'][0])):
                result = {
                    "content": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i] if results['metadatas'] else {},
                    "distance": results['distances'][0][i] if results['distances'] else 0,
                    "score": 1 - (results['distances'][0][i] if results['distances'] else 0)  # Convert distance to similarity
                }
                formatted_results.append(result)
        
        return formatted_results
    
    def clear(self):
        """Clear all documents from the collection."""
        self.client.delete_collection(self.COLLECTION_NAME)
        self.collection = self.client.create_collection(
            name=self.COLLECTION_NAME,
            metadata={"description": "Acme Corporation Policy Documents"}
        )
        print("Vector store cleared")
    
    @property
    def count(self) -> int:
        """Return the number of documents in the store."""
        return self.collection.count()


def initialize_vectorstore(force_reindex: bool = False) -> VectorStore:
    """
    Initialize the vector store and index documents if needed.
    
    Args:
        force_reindex: If True, clear and re-index all documents
    
    Returns:
        Initialized VectorStore instance
    """
    store = VectorStore()
    
    # Check if we need to index
    if force_reindex or store.count == 0:
        print("Indexing documents...")
        
        if force_reindex:
            store.clear()
        
        # Load and chunk documents
        chunks = ingest_documents()
        
        # Add to vector store
        store.add_documents(chunks)
        
        print(f"Indexing complete. Total documents: {store.count}")
    else:
        print(f"Using existing index with {store.count} documents")
    
    return store


# Singleton instance
_vectorstore = None


def get_vectorstore() -> VectorStore:
    """Get or create the singleton vector store instance."""
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = initialize_vectorstore()
    return _vectorstore


if __name__ == "__main__":
    # Test vector store
    store = initialize_vectorstore(force_reindex=True)
    
    # Test search
    query = "How many vacation days do I get?"
    results = store.search(query, k=3)
    
    print(f"\nSearch results for: '{query}'")
    for i, result in enumerate(results):
        print(f"\n--- Result {i+1} (score: {result['score']:.3f}) ---")
        print(f"Source: {result['metadata'].get('source', 'unknown')}")
        print(f"Content: {result['content'][:300]}...")
