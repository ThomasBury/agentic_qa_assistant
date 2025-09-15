"""RAG Tool with citations and answer synthesis."""

import time
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from openai import OpenAI

from .rag_pipeline import RagPipeline, SearchResult

logger = logging.getLogger(__name__)


@dataclass
class Citation:
    """Citation with source and page information."""
    source: str
    page: int
    chunk_id: int
    text_snippet: str
    doc_type: str
    brand: Optional[str] = None


@dataclass
class RagResult:
    """RAG tool result with answer and citations."""
    answer: str
    citations: List[Citation]
    retrieval_ms: int
    synthesis_ms: int
    total_ms: int
    chunks_retrieved: int
    confidence_score: Optional[float] = None


class RagTool:
    """RAG tool that retrieves and synthesizes answers with citations."""
    
    def __init__(self, openai_client: OpenAI, rag_pipeline: RagPipeline):
        """Initialize RAG tool.
        
        Args:
            openai_client: OpenAI client instance
            rag_pipeline: RAG pipeline for document retrieval
        """
        self.client = openai_client
        self.rag_pipeline = rag_pipeline
        
    def answer_question(self, question: str, k: int = 5, min_score: float = 0.5) -> RagResult:
        """Answer a question using retrieved documents.
        
        Args:
            question: User question
            k: Number of chunks to retrieve
            min_score: Minimum similarity score threshold
            
        Returns:
            RAG result with answer and citations
        """
        start_time = time.time()
        
        # Retrieve relevant chunks
        retrieval_start = time.time()
        search_results = self.rag_pipeline.search(question, k=k)
        retrieval_ms = int((time.time() - retrieval_start) * 1000)
        
        # Filter by score threshold
        filtered_results = [r for r in search_results if r.score >= min_score]
        
        if not filtered_results:
            # Lower threshold if no results
            filtered_results = search_results[:min(3, len(search_results))]
            logger.warning(f"No results above threshold {min_score}, using top {len(filtered_results)} results")
            
        logger.info(f"Retrieved {len(filtered_results)} relevant chunks for question: {question[:50]}...")
        
        # Synthesize answer
        synthesis_start = time.time()
        answer, citations, confidence = self._synthesize_answer(question, filtered_results)
        synthesis_ms = int((time.time() - synthesis_start) * 1000)
        
        total_ms = int((time.time() - start_time) * 1000)
        
        return RagResult(
            answer=answer,
            citations=citations,
            retrieval_ms=retrieval_ms,
            synthesis_ms=synthesis_ms,
            total_ms=total_ms,
            chunks_retrieved=len(filtered_results),
            confidence_score=confidence
        )
        
    def _synthesize_answer(self, question: str, search_results: List[SearchResult]) -> tuple[str, List[Citation], float]:
        """Synthesize answer from retrieved chunks.
        
        Args:
            question: Original question
            search_results: Search results with chunks
            
        Returns:
            Tuple of (answer, citations, confidence_score)
        """
        if not search_results:
            return "Insufficient evidence to answer the question.", [], 0.0
            
        # Prepare context from chunks
        context_parts = []
        citations = []
        
        for i, result in enumerate(search_results):
            chunk = result.chunk
            citation_id = i + 1
            
            # Create citation
            citation = Citation(
                source=chunk.source,
                page=chunk.page,
                chunk_id=chunk.chunk_id,
                text_snippet=chunk.text[:100] + "..." if len(chunk.text) > 100 else chunk.text,
                doc_type=chunk.doc_type,
                brand=chunk.brand
            )
            citations.append(citation)
            
            # Add to context with citation marker
            context_parts.append(f"[{citation_id}] {chunk.text}")
            
        context = "\n\n".join(context_parts)
        
        # Create synthesis prompt
        system_prompt = """You are an expert assistant that answers questions using only the provided context.

Rules:
1. Answer ONLY based on the provided context - do not use external knowledge
2. Include inline citations using [1], [2], etc. format for each fact
3. If the context doesn't contain enough information, say "The available documents don't contain sufficient information"
4. Quote exact phrases when relevant, using quotation marks
5. Be precise and factual
6. If warranty periods or technical details are mentioned, include them exactly as stated

Format citations as [Document p.X] where Document is the source name and X is the page number."""
        
        user_prompt = f"""Context:
{context}

Question: {question}

Please answer the question using only the information provided in the context above. Include appropriate citations."""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=800
            )
            
            answer = response.choices[0].message.content.strip()
            
            # Convert citation format from [1] to [source p.X]
            answer = self._format_citations(answer, citations)
            
            # Calculate confidence based on result scores
            avg_score = sum(r.score for r in search_results) / len(search_results)
            confidence = min(0.95, avg_score * 1.2)  # Scale and cap confidence
            
            logger.info(f"Synthesized answer with {len(citations)} citations (confidence: {confidence:.2f})")
            return answer, citations, confidence
            
        except Exception as e:
            logger.error(f"Answer synthesis failed: {e}")
            return f"Error generating answer: {e}", citations, 0.0
            
    def _format_citations(self, answer: str, citations: List[Citation]) -> str:
        """Convert numbered citations [1] to document format [source p.X].
        
        Args:
            answer: Answer text with numbered citations
            citations: List of citation objects
            
        Returns:
            Answer with formatted citations
        """
        for i, citation in enumerate(citations, 1):
            old_format = f"[{i}]"
            new_format = f"[{citation.source} p.{citation.page}]"
            answer = answer.replace(old_format, new_format)
            
        return answer
        
    def get_source_info(self) -> Dict[str, Any]:
        """Get information about available sources.
        
        Returns:
            Dictionary with source statistics
        """
        return self.rag_pipeline.get_corpus_stats()
        
    def search_documents(self, query: str, k: int = 5, doc_type_filter: Optional[str] = None) -> List[SearchResult]:
        """Search documents without answer synthesis.
        
        Args:
            query: Search query
            k: Number of results
            doc_type_filter: Filter by document type
            
        Returns:
            List of search results
        """
        return self.rag_pipeline.search(query, k=k, doc_type_filter=doc_type_filter)
        
    def validate_answer_quality(self, question: str, answer: str, citations: List[Citation]) -> Dict[str, Any]:
        """Validate answer quality and citation coverage.
        
        Args:
            question: Original question
            answer: Generated answer
            citations: Citations used
            
        Returns:
            Quality metrics dictionary
        """
        metrics = {
            "has_answer": len(answer) > 10 and "insufficient" not in answer.lower(),
            "has_citations": len(citations) > 0,
            "citation_count": len(citations),
            "answer_length": len(answer),
            "unique_sources": len(set(c.source for c in citations)),
            "doc_types_used": list(set(c.doc_type for c in citations))
        }
        
        # Check if answer actually uses the citations
        citation_usage = 0
        for citation in citations:
            citation_ref = f"[{citation.source} p.{citation.page}]"
            if citation_ref in answer:
                citation_usage += 1
                
        metrics["citations_used_in_answer"] = citation_usage
        metrics["citation_usage_rate"] = citation_usage / len(citations) if citations else 0.0
        
        return metrics
