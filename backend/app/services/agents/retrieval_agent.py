"""Retrieval agent - handles context retrieval from vector store."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.services.vector_store import search_vector_store
from app.services.agents.router_agent import RouterAgent, QueryAnalysis, QueryType


@dataclass
class RetrievedChunk:
    """A chunk retrieved from the vector store."""
    text: str
    chunk_id: str
    filename: str
    header_context: str
    distance: float
    relevance_score: float = 0.0


@dataclass
class RetrievalResult:
    """Result from the retrieval agent."""
    chunks: list[RetrievedChunk]
    query_used: str
    total_found: int
    quality_score: float  # 0.0-1.0 assessment of retrieval quality


class RetrievalAgent:
    """
    Retrieval Agent - Handles context retrieval from the vector store.
    
    Responsibilities:
    - Execute searches based on analyzed queries
    - Optimize search parameters based on query type
    - Score and rank retrieved chunks
    - Handle fallback strategies when initial retrieval fails
    
    Works in coordination with the Router Agent to determine
    optimal retrieval strategy.
    """
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
    
    async def retrieve(
        self,
        query: str,
        domain: str,
        analysis: QueryAnalysis | None = None,
        top_k: int = 5,
    ) -> RetrievalResult:
        """
        Retrieve relevant chunks from vector store.
        
        Args:
            query: The search query (can be reformulated)
            domain: The topic/domain to search in
            analysis: Optional query analysis from router
            top_k: Number of chunks to retrieve
            
        Returns:
            RetrievalResult with chunks and quality metrics
        """
        # Use reformulated query if available
        search_query = analysis.reformulated_query if analysis else query
        
        # Adjust top_k based on analysis
        if analysis:
            top_k = RouterAgent().get_top_k(analysis)
        
        # Execute search
        results = await search_vector_store(
            query=search_query,
            domain=domain,
            limit=top_k
        )
        
        # Convert to RetrievedChunk objects
        chunks = []
        for result in results:
            chunk = RetrievedChunk(
                text=result["text"],
                chunk_id=result["chunk_id"],
                filename=result["metadata"].get("filename", "Unknown"),
                header_context=result["metadata"].get("header_context", ""),
                distance=result.get("distance", 1.0),
            )
            # Calculate relevance score (inverse of distance)
            chunk.relevance_score = self._calculate_relevance(chunk)
            chunks.append(chunk)
        
        # Calculate quality score
        quality_score = self._assess_quality(chunks)
        
        return RetrievalResult(
            chunks=chunks,
            query_used=search_query,
            total_found=len(chunks),
            quality_score=quality_score,
        )
    
    async def retrieve_with_fallback(
        self,
        query: str,
        domain: str,
        analysis: QueryAnalysis | None = None,
    ) -> RetrievalResult:
        """
        Retrieve with fallback strategy if initial retrieval is poor.
        
        If the initial retrieval returns low-quality results, this method
        will try alternative queries to improve results.
        """
        # First attempt
        result = await self.retrieve(query, domain, analysis)
        
        # If quality is poor, try fallback strategies
        if result.quality_score < 0.3 and result.total_found > 0:
            # Try with original query
            original_result = await self.retrieve(query, domain, None, top_k=5)
            if original_result.quality_score > result.quality_score:
                return original_result
        
        # If no results found, try keyword-based search
        if result.total_found == 0:
            if analysis and analysis.keywords:
                # Try searching with just keywords
                keyword_query = " ".join(analysis.keywords[:3])
                fallback_result = await self.retrieve(keyword_query, domain, None, top_k=3)
                if fallback_result.total_found > 0:
                    return fallback_result
            
            # Last resort: try generic search
            generic_result = await self.retrieve(
                f"{domain} fundamentals basics",
                domain,
                None,
                top_k=3
            )
            if generic_result.total_found > 0:
                return generic_result
        
        return result
    
    def _calculate_relevance(self, chunk: RetrievedChunk) -> float:
        """Calculate relevance score from distance."""
        # Distance typically 0-2, convert to 0-1 relevance
        # 0 distance = 1.0 relevance, 2 distance = 0.0 relevance
        relevance = max(0.0, 1.0 - (chunk.distance / 1.5))
        return round(relevance, 3)
    
    def _assess_quality(self, chunks: list[RetrievedChunk]) -> float:
        """Assess overall retrieval quality."""
        if not chunks:
            return 0.0
        
        # Factors:
        # 1. Average relevance score
        avg_relevance = sum(c.relevance_score for c in chunks) / len(chunks)
        
        # 2. Number of chunks found
        count_factor = min(1.0, len(chunks) / 5.0)
        
        # 3. Diversity (different sources = better)
        unique_files = len(set(c.filename for c in chunks))
        diversity_factor = min(1.0, unique_files / 3.0)
        
        # Weighted combination
        quality = (avg_relevance * 0.5) + (count_factor * 0.3) + (diversity_factor * 0.2)
        
        return round(quality, 3)
    
    def rank_chunks(
        self,
        chunks: list[RetrievedChunk],
        query: str,
        query_type: QueryType | None = None,
    ) -> list[RetrievedChunk]:
        """
        Rank chunks based on query type and content relevance.
        
        Different query types may benefit from different ranking strategies.
        """
        query_lower = query.lower()
        
        def chunk_score(chunk: RetrievedChunk) -> float:
            score = chunk.relevance_score
            
            # Boost chunks with matching keywords
            chunk_lower = chunk.text.lower()
            if any(kw in chunk_lower for kw in query_lower.split() if len(kw) > 3):
                score += 0.1
            
            # For conceptual queries, prefer overview sections
            if query_type == QueryType.CONCEPTUAL:
                if "introduction" in chunk.header_context.lower() or "overview" in chunk.header_context.lower():
                    score += 0.15
            
            # For procedural queries, prefer step-by-step sections
            elif query_type == QueryType.PROCEDURAL:
                if "step" in chunk.header_context.lower() or "process" in chunk.header_context.lower():
                    score += 0.15
            
            # For examples, prefer sections with "example" in header
            elif query_type == QueryType.EXAMPLE:
                if "example" in chunk.header_context.lower() or "instance" in chunk.header_context.lower():
                    score += 0.2
            
            return score
        
        return sorted(chunks, key=chunk_score, reverse=True)
