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
    """Represents a text chunk with its associated metadata.

    Attributes
    ----------
    text : str
        The text content of the chunk.
    source : str
        The name of the source document file.
    page : int
        The page number in the source document.
    chunk_id : int
        The sequential ID of the chunk within a page.
    doc_type : str
        The inferred type of the document (e.g., 'manual', 'contract').
    brand : Optional[str]
        The inferred brand (e.g., 'Toyota', 'Lexus').
    model : Optional[str]
        The inferred vehicle model.
    year : Optional[int]
        The inferred year from the document filename.
    region : Optional[str]
        The inferred geographical region.
    checksum : Optional[str]
        An MD5 hash of the chunk's text content.

    """
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
    """Represents a single search result from the vector index.

    Attributes
    ----------
    chunk : Chunk
        The retrieved text chunk.
    score : float
        The similarity score of the chunk to the query.
    rank : int
        The rank of the result in the search results list.
    """
    chunk: Chunk
    score: float
    rank: int


class DocumentProcessor:
    """Processes PDF documents into text chunks."""
    
    def __init__(self, chunk_size: int = 600, overlap: int = 100):
        """Initialize the DocumentProcessor.

        Parameters
        ----------
        chunk_size : int, optional
            The target size of each chunk in tokens.
        overlap : int, optional
            The number of tokens to overlap between consecutive chunks.

        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.tokenizer = tiktoken.get_encoding("cl100k_base")  # GPT-4 tokenizer
        
    def process_document(self, doc_path: Path) -> List[Chunk]:
        """Process a single document file (PDF or text) into chunks.

        Parameters
        ----------
        doc_path : Path
            The path to the document file.

        Returns
        -------
        List[Chunk]
            A list of Chunk objects extracted from the document.

        """
        if doc_path.suffix.lower() == '.pdf':
            return self._process_pdf(doc_path)
        elif doc_path.suffix.lower() in ['.txt', '.md']:
            return self._process_text(doc_path)
        else:
            logger.warning(f"Unsupported file type: {doc_path.suffix}")
            return []
    
    def _process_pdf(self, pdf_path: Path) -> List[Chunk]:
        """Extract text from a PDF and split it into chunks.

        Parameters
        ----------
        pdf_path : Path
            The path to the PDF file.

        Returns
        -------
        List[Chunk]
            A list of chunks from the PDF.

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
        """Read a text file and split it into chunks.

        Parameters
        ----------
        text_path : Path
            The path to the text file.

        Returns
        -------
        List[Chunk]
            A list of chunks from the text file.

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
        """Perform basic cleaning and normalization of extracted text.

        Parameters
        ----------
        text : str
            The raw text to clean.

        Returns
        -------
        str
            The cleaned text.
        """
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
        """Split a single block of text into overlapping chunks.

        Parameters
        ----------
        text : str
            The text to be chunked.
        pdf_path : Path
            The path to the source document, used for metadata.
        page_num : int
            The page number of the text.

        Returns
        -------
        List[Chunk]
            A list of Chunk objects.
        """
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
        """Create a Chunk object and infer metadata from its context.

        Parameters
        ----------
        text : str
            The text content of the chunk.
        pdf_path : Path
            The path to the source document.
        page_num : int
            The page number of the chunk.
        chunk_id : int
            The ID of the chunk within the page.

        Returns
        -------
        Chunk
            A fully populated Chunk object.

        """
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
    
    def __init__(self, embedding_model: str = "text-embedding-3-small", cost_tracker=None):
        """Initialize the VectorIndex.

        Parameters
        ----------
        embedding_model : str, optional
            The name of the OpenAI embedding model to use.
        cost_tracker : object, optional
            An instance of CostTracker to record token usage.

        """
        self.embedding_model = embedding_model
        self.dimension = 1536  # text-embedding-3-small dimension
        self.index: Optional[faiss.Index] = None
        self.chunks: List[Chunk] = []
        self.embeddings: Optional[np.ndarray] = None
        self.cost_tracker = cost_tracker
        
    def create_embeddings(self, texts: List[str], openai_client: OpenAI) -> np.ndarray:
        """Create embeddings for a list of texts using the OpenAI API.

        Parameters
        ----------
        texts : List[str]
            A list of text strings to embed.
        openai_client : OpenAI
            An initialized OpenAI client.

        Returns
        -------
        np.ndarray
            A NumPy array of the resulting embeddings.
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
                
                # Token tracking: use API usage if available, otherwise estimate
                if self.cost_tracker is not None:
                    usage = getattr(response, 'usage', None)
                    if usage and hasattr(usage, 'total_tokens') and usage.total_tokens:
                        self.cost_tracker.add_embeddings_usage(self.embedding_model, int(usage.total_tokens))
                    else:
                        try:
                            est = self.cost_tracker.estimate_tokens(batch_texts)
                            self.cost_tracker.add_embeddings_usage(self.embedding_model, est)
                        except Exception:
                            pass
                
                logger.info(f"Processed batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
                
            except Exception as e:
                logger.error(f"Failed to create embeddings for batch {i//batch_size + 1}: {e}")
                raise
                
        embeddings_array = np.array(all_embeddings, dtype=np.float32)
        logger.info(f"Created embeddings with shape {embeddings_array.shape}")
        
        return embeddings_array
        
    def build_index(self, chunks: List[Chunk], openai_client: OpenAI):
        """Build the FAISS index from a list of chunks.

        This method creates embeddings for the chunks and adds them to the
        FAISS index.

        Parameters
        ----------
        chunks : List[Chunk]
            A list of Chunk objects to be indexed.
        openai_client : OpenAI
            An initialized OpenAI client.
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
        """Search the index for chunks similar to a query.

        Parameters
        ----------
        query : str
            The search query string.
        openai_client : OpenAI
            An initialized OpenAI client.
        k : int, optional
            The number of top results to return.

        Returns
        -------
        List[SearchResult]
            A list of ranked search results.
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
        """Save the FAISS index and associated metadata to disk.

        Parameters
        ----------
        path : Path
            The directory where the index files will be saved.
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
        """Load the FAISS index and metadata from disk.

        Parameters
        ----------
        path : Path
            The directory containing the index files.

        Returns
        -------
        bool
            True if the index was loaded successfully, False otherwise.
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
    
    def __init__(self, openai_client: OpenAI, index_path: Optional[Path] = None, cost_tracker=None):
        """Initialize the RAGPipeline.

        Parameters
        ----------
        openai_client : OpenAI
            An initialized OpenAI client.
        index_path : Optional[Path], optional
            The path to save or load the vector index.
        cost_tracker : object, optional
            An instance of CostTracker to record token usage.
        """
        self.client = openai_client
        self.processor = DocumentProcessor()
        self.index = VectorIndex(cost_tracker=cost_tracker)
        self.index_path = index_path or Path("./vector_index")
        
    def ingest_documents(self, doc_paths: List[Path], force_rebuild: bool = False):
        """Ingest documents, creating and saving a vector index.

        If an index already exists, it will be loaded unless `force_rebuild` is True.

        Parameters
        ----------
        doc_paths : List[Path]
            A list of paths to the document files to ingest.
        force_rebuild : bool, optional
            If True, forces the pipeline to rebuild the index from scratch.
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
        """Search the indexed documents for relevant chunks.

        Parameters
        ----------
        query : str
            The search query string.
        k : int, optional
            The number of top results to return.
        doc_type_filter : Optional[str], optional
            An optional filter to restrict results to a specific document type
            (e.g., 'contract', 'warranty', 'manual').

        Returns
        -------
        List[SearchResult]
            A list of ranked search results.
        """
        # Get initial results
        results = self.index.search(query, self.client, k=k*2)  # Get more to allow filtering
        
        # Apply filters if specified
        if doc_type_filter:
            results = [r for r in results if r.chunk.doc_type == doc_type_filter.lower()]
            
        # Return top k results
        return results[:k]
        
    def get_corpus_stats(self) -> Dict[str, Any]:
        """Get statistics about the ingested document corpus.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing statistics like total chunks, sources,
            and counts by document type and brand.
        """
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