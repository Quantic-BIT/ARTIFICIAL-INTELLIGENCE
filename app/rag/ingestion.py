"""
Document Ingestion Module

Handles loading, parsing, and chunking of policy documents.
"""
import os
from pathlib import Path
from typing import List, Dict, Any


class Document:
    """Simple document class to hold content and metadata."""
    
    def __init__(self, content: str, metadata: Dict[str, Any] = None):
        self.page_content = content
        self.metadata = metadata or {}


class DocumentLoader:
    """Loads documents from the policies directory."""
    
    def __init__(self, policies_dir: str = None):
        if policies_dir is None:
            # Default to policies directory relative to project root
            self.policies_dir = Path(__file__).parent.parent.parent / "policies"
        else:
            self.policies_dir = Path(policies_dir)
    
    def load_documents(self) -> List[Document]:
        """Load all markdown files from the policies directory."""
        documents = []
        
        if not self.policies_dir.exists():
            print(f"Warning: Policies directory not found: {self.policies_dir}")
            return documents
        
        for file_path in self.policies_dir.glob("*.md"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Extract title from first heading or filename
                title = self._extract_title(content, file_path.stem)
                
                doc = Document(
                    content=content,
                    metadata={
                        "source": file_path.name,
                        "title": title,
                        "file_path": str(file_path)
                    }
                )
                documents.append(doc)
                print(f"Loaded: {file_path.name}")
                
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        return documents
    
    def _extract_title(self, content: str, fallback: str) -> str:
        """Extract title from the first markdown heading."""
        lines = content.split('\n')
        for line in lines:
            if line.startswith('# '):
                return line[2:].strip()
        return fallback.replace('_', ' ').title()


class TextChunker:
    """Chunks documents into smaller pieces for embedding."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into smaller chunks."""
        chunks = []
        
        for doc in documents:
            doc_chunks = self._split_text(doc.page_content)
            
            for i, chunk_text in enumerate(doc_chunks):
                chunk = Document(
                    content=chunk_text,
                    metadata={
                        **doc.metadata,
                        "chunk_id": i,
                        "chunk_total": len(doc_chunks)
                    }
                )
                chunks.append(chunk)
        
        return chunks
    
    def _split_text(self, text: str) -> List[str]:
        """Split text into chunks with overlap."""
        chunks = []
        
        # Split by paragraphs first (double newline)
        paragraphs = text.split('\n\n')
        
        current_chunk = ""
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # If adding this paragraph exceeds chunk size
            if len(current_chunk) + len(para) + 2 > self.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    # Keep overlap from end of current chunk
                    overlap_text = current_chunk[-self.chunk_overlap:] if len(current_chunk) > self.chunk_overlap else current_chunk
                    current_chunk = overlap_text + "\n\n" + para
                else:
                    # Paragraph itself is too long, split by sentences/lines
                    para_chunks = self._split_large_paragraph(para)
                    chunks.extend(para_chunks[:-1])
                    current_chunk = para_chunks[-1] if para_chunks else ""
            else:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _split_large_paragraph(self, para: str) -> List[str]:
        """Split a large paragraph by lines or sentences."""
        lines = para.split('\n')
        chunks = []
        current = ""
        
        for line in lines:
            if len(current) + len(line) + 1 > self.chunk_size:
                if current:
                    chunks.append(current.strip())
                current = line
            else:
                current = current + "\n" + line if current else line
        
        if current:
            chunks.append(current.strip())
        
        return chunks if chunks else [para]


def ingest_documents(policies_dir: str = None, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """
    Main ingestion function - loads and chunks all policy documents.
    
    Args:
        policies_dir: Path to the policies directory
        chunk_size: Maximum size of each chunk
        chunk_overlap: Overlap between chunks
    
    Returns:
        List of chunked documents ready for embedding
    """
    loader = DocumentLoader(policies_dir)
    documents = loader.load_documents()
    
    print(f"Loaded {len(documents)} documents")
    
    chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = chunker.chunk_documents(documents)
    
    print(f"Created {len(chunks)} chunks")
    
    return chunks


if __name__ == "__main__":
    # Test ingestion
    chunks = ingest_documents()
    for chunk in chunks[:3]:
        print(f"\n--- {chunk.metadata['source']} (chunk {chunk.metadata['chunk_id']}) ---")
        print(chunk.page_content[:200] + "...")
