"""RAG pipeline for PDF processing and vector search."""

import os
import time
import hashlib
import logging
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import faiss
import pdfplumber
import tiktoken
from openai import OpenAI

logger = logging.getLogger(__name__)


@dataclass 
class Chunk:
    """Represents a text chunk with metadata."""
    text: str
    source: str
    page: int
    chunk_id: int
    doc_type: str
    brand: Optional[str] = None
    model: Optional[str] = None
    year: Optional[int] = None
    region: Optional[str] = None
    checksum: Optional[str] = None
    
    def __post_init__(self):
        """Calculate checksum after initialization."""
        if not self.checksum:
            self.checksum = hashlib.md5(self.text.encode()).hexdigest()


@dataclass
class SearchResult:
    """Search result with relevance score."""
    chunk: Chunk
    score: float
    rank: int


class DocumentProcessor:
    """Processes PDF documents into text chunks."""
    
    def __init__(self, chunk_size: int = 600, overlap: int = 100):
        """Initialize document processor.
        
        Args:
            chunk_size: Target chunk size in tokens
            overlap: Overlap between chunks in tokens
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.tokenizer = tiktoken.get_encoding("cl100k_base")  # GPT-4 tokenizer
        
    def process_document(self, doc_path: Path) -> List[Chunk]:
        """Process a document (PDF or text) into chunks.
        
        Args:
            doc_path: Path to document file
            
        Returns:
            List of text chunks with metadata
        """
        if doc_path.suffix.lower() == '.pdf':
            return self._process_pdf(doc_path)
        elif doc_path.suffix.lower() in ['.txt', '.md']:
            return self._process_text(doc_path)
        else:
            logger.warning(f"Unsupported file type: {doc_path.suffix}")
            return []
    
    def process_pdf(self, pdf_path: Path) -> List[Chunk]:
        """Process a PDF into chunks (backward compatibility)."""
        return self._process_pdf(pdf_path)
        
    def _process_pdf(self, pdf_path: Path) -> List[Chunk]:
        """Process a single PDF into chunks.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of text chunks with metadata
        """
        chunks = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                logger.info(f"Processing {pdf_path.name} ({len(pdf.pages)} pages)")
                
                for page_num, page in enumerate(pdf.pages, 1):
                    text = page.extract_text()
                    if text:
                        # Clean the text
                        text = self._clean_text(text)
                        
                        # Create chunks for this page
                        page_chunks = self._create_chunks(text, pdf_path, page_num)
                        chunks.extend(page_chunks)
                        
        except Exception as e:
            logger.error(f"Failed to process {pdf_path}: {e}")
            return []
            
        logger.info(f"Created {len(chunks)} chunks from {pdf_path.name}")
        return chunks
        
    def _process_text(self, text_path: Path) -> List[Chunk]:
        """Process a text file into chunks.
        
        Args:
            text_path: Path to text file
            
        Returns:
            List of text chunks with metadata
        """
        chunks = []
        
        try:
            with open(text_path, 'r', encoding='utf-8') as f:
                text = f.read()
                
            logger.info(f"Processing {text_path.name} ({len(text)} characters)")
            
            # Clean the text
            text = self._clean_text(text)
            
            # Create chunks (treat whole file as one "page")
            chunks = self._create_chunks(text, text_path, page_num=1)
                
        except Exception as e:
            logger.error(f"Failed to process {text_path}: {e}")
            return []
            
        logger.info(f"Created {len(chunks)} chunks from {text_path.name}")
        return chunks
        
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Remove page numbers and headers/footers (basic cleanup)
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            # Skip very short lines that might be headers/footers
            if len(line) > 10:
                cleaned_lines.append(line)
                
        return ' '.join(cleaned_lines)
        
    def _create_chunks(self, text: str, pdf_path: Path, page_num: int) -> List[Chunk]:
        """Create overlapping chunks from text."""
        
        # Tokenize the text
        tokens = self.tokenizer.encode(text)
        
        if len(tokens) <= self.chunk_size:
            # Single chunk if text is small enough
            return [self._create_chunk_object(text, pdf_path, page_num, 0)]
            
        chunks = []
        chunk_id = 0
        
        for start in range(0, len(tokens), self.chunk_size - self.overlap):
            end = min(start + self.chunk_size, len(tokens))
            
            # Decode chunk tokens back to text
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            
            # Create chunk object
            chunk = self._create_chunk_object(chunk_text, pdf_path, page_num, chunk_id)
            chunks.append(chunk)
            
            chunk_id += 1
            
            # Stop if we've reached the end
            if end >= len(tokens):
                break
                
        return chunks
        
    def _create_chunk_object(self, text: str, pdf_path: Path, page_num: int, chunk_id: int) -> Chunk:
        """Create a Chunk object with inferred metadata."""
        
        # Infer document type and metadata from filename and content
        filename = pdf_path.stem.lower()
        text_lower = text.lower()
        doc_type = "unknown"
        brand = None
        year = None
        
        if "contract" in filename:
            doc_type = "contract"
            if "toyota" in filename:
                brand = "Toyota"
            elif "lexus" in filename:
                brand = "Lexus"
        elif "warranty" in filename:
            doc_type = "warranty"
        elif "manual" in filename:
            doc_type = "manual"
            # Check content for brand info
            if "toyota" in text_lower and "lexus" not in text_lower:
                brand = "Toyota"
            elif "lexus" in text_lower and "toyota" not in text_lower:
                brand = "Lexus"
            
        # Extract year from filename
        import re
        year_match = re.search(r'20\d{2}', filename)
        if year_match:
            year = int(year_match.group())
            
        return Chunk(
            text=text,
            source=pdf_path.name,
            page=page_num,
            chunk_id=chunk_id,
            doc_type=doc_type,
            brand=brand,
            year=year,
            region="Europe"  # Assuming EU context based on PRD
        )


class VectorIndex:
    """FAISS-based vector index for semantic search."""
    
    def __init__(self, embedding_model: str = "text-embedding-3-small"):
        """Initialize vector index.
        
        Args:
            embedding_model: OpenAI embedding model to use
        """
        self.embedding_model = embedding_model
        self.dimension = 1536  # text-embedding-3-small dimension
        self.index = None
        self.chunks: List[Chunk] = []
        self.embeddings: Optional[np.ndarray] = None
        
    def create_embeddings(self, texts: List[str], openai_client: OpenAI) -> np.ndarray:
        """Create embeddings for a list of texts.
        
        Args:
            texts: List of text strings
            openai_client: OpenAI client instance
            
        Returns:
            Numpy array of embeddings
        """
        logger.info(f"Creating embeddings for {len(texts)} texts")
        
        # Process in batches to respect API limits
        batch_size = 100  # Conservative batch size
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            try:
                response = openai_client.embeddings.create(
                    input=batch_texts,
                    model=self.embedding_model
                )
                
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                
                logger.info(f"Processed batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
                
            except Exception as e:
                logger.error(f"Failed to create embeddings for batch {i//batch_size + 1}: {e}")
                raise
                
        embeddings_array = np.array(all_embeddings, dtype=np.float32)
        logger.info(f"Created embeddings with shape {embeddings_array.shape}")
        
        return embeddings_array
        
    def build_index(self, chunks: List[Chunk], openai_client: OpenAI):
        """Build FAISS index from chunks.
        
        Args:
            chunks: List of text chunks
            openai_client: OpenAI client instance
        """
        self.chunks = chunks
        
        # Extract text from chunks
        texts = [chunk.text for chunk in chunks]
        
        # Create embeddings
        self.embeddings = self.create_embeddings(texts, openai_client)
        
        # Build FAISS index
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        normalized_embeddings = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.index.add(normalized_embeddings)
        
        logger.info(f"Built FAISS index with {self.index.ntotal} vectors")
        
    def search(self, query: str, openai_client: OpenAI, k: int = 5) -> List[SearchResult]:
        """Search the index for similar chunks.
        
        Args:
            query: Search query
            openai_client: OpenAI client instance
            k: Number of results to return
            
        Returns:
            List of search results ranked by similarity
        """
        if not self.index or not self.chunks:
            raise ValueError("Index not built. Call build_index() first.")
            
        # Create query embedding
        query_embedding = self.create_embeddings([query], openai_client)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)  # Normalize
        
        # Search index
        scores, indices = self.index.search(query_embedding, k)
        
        # Create search results
        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0])):
            if idx < len(self.chunks):  # Safety check
                result = SearchResult(
                    chunk=self.chunks[idx],
                    score=float(score),
                    rank=rank
                )
                results.append(result)
                
        logger.info(f"Found {len(results)} results for query: {query[:50]}...")
        return results
        
    def save(self, path: Path):
        """Save index to disk.
        
        Args:
            path: Directory to save index files
        """
        path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(path / "index.faiss"))
        
        # Save chunks metadata
        with open(path / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)
            
        # Save embeddings
        np.save(path / "embeddings.npy", self.embeddings)
        
        logger.info(f"Saved index to {path}")
        
    def load(self, path: Path) -> bool:
        """Load index from disk.
        
        Args:
            path: Directory containing index files
            
        Returns:
            True if loaded successfully
        """
        try:
            # Load FAISS index
            self.index = faiss.read_index(str(path / "index.faiss"))
            
            # Load chunks metadata
            with open(path / "chunks.pkl", "rb") as f:
                self.chunks = pickle.load(f)
                
            # Load embeddings
            self.embeddings = np.load(path / "embeddings.npy")
            
            logger.info(f"Loaded index from {path} ({len(self.chunks)} chunks)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load index from {path}: {e}")
            return False


class RagPipeline:
    """Main RAG pipeline orchestrator."""
    
    def __init__(self, openai_client: OpenAI, index_path: Optional[Path] = None):
        """Initialize RAG pipeline.
        
        Args:
            openai_client: OpenAI client instance
            index_path: Path to save/load vector index
        """
        self.client = openai_client
        self.processor = DocumentProcessor()
        self.index = VectorIndex()
        self.index_path = index_path or Path("./vector_index")
        
    def ingest_documents(self, doc_paths: List[Path], force_rebuild: bool = False):
        """Ingest PDF documents into the vector index.
        
        Args:
            doc_paths: List of PDF file paths
            force_rebuild: Force rebuilding even if index exists
        """
        # Try to load existing index first
        if not force_rebuild and self.index_path.exists() and self.index.load(self.index_path):
            logger.info("Loaded existing vector index")
            return
            
        logger.info(f"Ingesting {len(doc_paths)} documents...")
        
        # Process all documents
        all_chunks = []
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_path = {
                executor.submit(self.processor.process_document, path): path 
                for path in doc_paths
            }
            
            for future in future_to_path:
                path = future_to_path[future]
                try:
                    chunks = future.result()
                    all_chunks.extend(chunks)
                    logger.info(f"Processed {path.name}: {len(chunks)} chunks")
                except Exception as e:
                    logger.error(f"Failed to process {path}: {e}")
                    
        if not all_chunks:
            raise ValueError("No chunks were created from the documents")
            
        logger.info(f"Created {len(all_chunks)} total chunks")
        
        # Build vector index
        self.index.build_index(all_chunks, self.client)
        
        # Save index
        self.index.save(self.index_path)
        
    def search(self, query: str, k: int = 5, doc_type_filter: Optional[str] = None) -> List[SearchResult]:
        """Search documents for relevant chunks.
        
        Args:
            query: Search query
            k: Number of results to return
            doc_type_filter: Filter by document type (contract, warranty, manual)
            
        Returns:
            List of ranked search results
        """
        # Get initial results
        results = self.index.search(query, self.client, k=k*2)  # Get more to allow filtering
        
        # Apply filters if specified
        if doc_type_filter:
            results = [r for r in results if r.chunk.doc_type == doc_type_filter]
            
        # Return top k results
        return results[:k]
        
    def get_corpus_stats(self) -> Dict[str, Any]:
        """Get statistics about the ingested corpus."""
        if not self.index.chunks:
            return {"total_chunks": 0}
            
        stats = {
            "total_chunks": len(self.index.chunks),
            "sources": list(set(chunk.source for chunk in self.index.chunks)),
            "doc_types": {},
            "brands": {},
            "total_text_length": sum(len(chunk.text) for chunk in self.index.chunks)
        }
        
        # Count by document type
        for chunk in self.index.chunks:
            stats["doc_types"][chunk.doc_type] = stats["doc_types"].get(chunk.doc_type, 0) + 1
            if chunk.brand:
                stats["brands"][chunk.brand] = stats["brands"].get(chunk.brand, 0) + 1
                
        return stats