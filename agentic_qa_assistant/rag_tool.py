"""RAG Tool with citations and answer synthesis."""

import time
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from openai import OpenAI
from .llm_utils import chat_params_for_model, is_reasoning_model

from .rag_pipeline import RagPipeline, SearchResult

logger = logging.getLogger(__name__)


@dataclass
class Citation:
    """Represents a citation to a source document.

    Attributes
    ----------
    source : str
        The name of the source document file.
    page : int
        The page number in the source document.
    chunk_id : int
        The ID of the chunk within the page.
    text_snippet : str
        A short snippet of the text from the cited chunk.
    doc_type : str
        The inferred type of the document.
    brand : Optional[str]
        The inferred brand associated with the document.
    """
    source: str
    page: int
    chunk_id: int
    text_snippet: str
    doc_type: str
    brand: Optional[str] = None


@dataclass
class RagResult:
    """Represents the result of a RAG tool execution.

    Attributes
    ----------
    answer : str
        The synthesized answer to the question.
    citations : List[Citation]
        A list of citations that support the answer.
    retrieval_ms : int
        The time taken for document retrieval in milliseconds.
    synthesis_ms : int
        The time taken for answer synthesis in milliseconds.
    total_ms : int
        The total execution time in milliseconds.
    chunks_retrieved : int
        The number of chunks retrieved from the vector store.
    confidence_score : Optional[float]
        An optional score representing the confidence in the answer,
        often derived from retrieval scores.
    """
    answer: str
    citations: List[Citation]
    retrieval_ms: int
    synthesis_ms: int
    total_ms: int
    chunks_retrieved: int
    confidence_score: Optional[float] = None


class RagTool:
    """RAG tool that retrieves and synthesizes answers with citations."""
    
    def __init__(self, openai_client: OpenAI, rag_pipeline: RagPipeline, model_name: str = "gpt-5-nano", cost_tracker=None, synthesis_model: Optional[str] = None):
        """Initialize the RagTool.

        Parameters
        ----------
        openai_client : OpenAI
            An initialized OpenAI client.
        rag_pipeline : RagPipeline
            An instance of the RAG pipeline for document retrieval.
        model_name : str, optional
            The name of the OpenAI model to use for answer synthesis.
        cost_tracker : object, optional
            An instance of CostTracker to record token usage.
        """
        self.model_name = model_name
        self.client = openai_client
        self.rag_pipeline = rag_pipeline
        self.cost_tracker = cost_tracker
        self.synthesis_model = synthesis_model
        
    def answer_question(self, question: str, k: int = 5, min_score: float = 0.5) -> RagResult:
        """Answer a question by retrieving relevant documents and synthesizing an answer.

        Parameters
        ----------
        question : str
            The user's question.
        k : int, optional
            The number of document chunks to retrieve.
        min_score : float, optional
            The minimum similarity score for a chunk to be considered relevant.

        Returns
        -------
        RagResult
            An object containing the answer, citations, and execution metrics.
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
        """Synthesize an answer from a list of retrieved search results using an LLM.

        Parameters
        ----------
        question : str
            The original user question.
        search_results : List[SearchResult]
            A list of search results containing relevant chunks.

        Returns
        -------
        tuple[str, List[Citation], float]
            A tuple containing the synthesized answer, a list of generated
            citations, and a confidence score.
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
            # Choose a faster synthesis model when the primary model is a reasoning model
            model_to_use = self.synthesis_model or ("gpt-4o-mini" if is_reasoning_model(self.model_name) else self.model_name)
            response = self.client.chat.completions.create(
                model=model_to_use,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                **chat_params_for_model(model_to_use, 800, temperature=0.1)
            )
            
            # Cost tracking
            try:
                if self.cost_tracker and getattr(response, 'usage', None):
                    u = response.usage
                    self.cost_tracker.add_chat_usage_detailed(
                        self.model_name,
                        prompt_tokens=getattr(u, 'prompt_tokens', 0),
                        completion_tokens=getattr(u, 'completion_tokens', 0),
                        total_tokens=getattr(u, 'total_tokens', 0),
                        cached_prompt_tokens=getattr(u, 'prompt_tokens_details', {}).get('cached_tokens', 0) if hasattr(u, 'prompt_tokens_details') else 0,
                        reasoning_tokens=getattr(u, 'completion_tokens_details', {}).get('reasoning_tokens', 0) if hasattr(u, 'completion_tokens_details') else 0,
                    )
            except Exception:
                pass
            
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
        """Convert numbered citations (e.g., [1]) to a full format (e.g., [source p.X]).

        Parameters
        ----------
        answer : str
            The answer text containing numbered citations.
        citations : List[Citation]
            The list of citation objects corresponding to the numbers.

        Returns
        -------
        str
            The answer with citations reformatted.
        """
        for i, citation in enumerate(citations, 1):
            old_format = f"[{i}]"
            new_format = f"[{citation.source} p.{citation.page}]"
            answer = answer.replace(old_format, new_format)
            
        return answer
        
    def get_source_info(self) -> Dict[str, Any]:
        """Get statistics about the available document sources in the RAG pipeline.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing corpus statistics.
        """
        return self.rag_pipeline.get_corpus_stats()
        
    def search_documents(self, query: str, k: int = 5, doc_type_filter: Optional[str] = None) -> List[SearchResult]:
        """Search for relevant document chunks without synthesizing an answer.

        Parameters
        ----------
        query : str
            The search query.
        k : int, optional
            The number of results to return.
        doc_type_filter : Optional[str], optional
            An optional filter for the document type.

        Returns
        -------
        List[SearchResult]
            A list of raw search results.
        """
        return self.rag_pipeline.search(query, k=k, doc_type_filter=doc_type_filter)
        
    def validate_answer_quality(self, question: str, answer: str, citations: List[Citation]) -> Dict[str, Any]:
        """Perform a basic validation of answer quality and citation coverage.

        Parameters
        ----------
        question : str
            The original question asked.
        answer : str
            The generated answer.
        citations : List[Citation]
            The list of citations provided with the answer.

        Returns
        -------
        Dict[str, Any]
            A dictionary of quality metrics.
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
