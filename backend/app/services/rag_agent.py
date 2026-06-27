from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from app.services.vector_store import search_vector_store
from app.services.llm_provider import LLMProvider
from app.core.config import settings


# Default system prompt for the tutor
DEFAULT_SYSTEM_PROMPT = """You are an expert tutor helping students learn about the provided topic.
Your goal is to provide clear, accurate, and helpful explanations based ONLY on the context provided.
If the context doesn't contain enough information to answer the question, say so honestly.
Always cite your sources using the format [Source: filename] when using information from the context.
Be encouraging and patient. Break down complex concepts into smaller, understandable parts."""


@dataclass
class RetrievedChunk:
    """A chunk retrieved from the vector store."""
    text: str
    chunk_id: str
    filename: str
    header_context: str
    distance: float


@dataclass
class AgentResponse:
    """Response from the RAG agent."""
    answer: str
    citations: list[str]
    sources: list[RetrievedChunk]
    confidence: float  # 0.0 to 1.0


class RAGAgent:
    """
    Single-agent RAG system for the tutor.
    
    Handles:
    - Query analysis and intent detection
    - Context retrieval from vector store
    - Answer synthesis with citations
    - Fallback handling when no context is found
    """
    
    def __init__(
        self,
        model_name: str = "gemma4:e2b",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        base_url: str = "http://localhost:11434",
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = base_url
        
        # Initialize the LLM
        from app.core.config import settings
        self.llm = LLMProvider(
            provider=settings.llm_provider,
            model_path=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            openai_api_key=settings.openai_api_key,
            openai_model=settings.openai_model,
            openai_base_url=settings.openai_base_url,
            groq_api_key=settings.groq_api_key,
            groq_model=settings.groq_model,
            ollama_base_url=base_url,
        )
        
    def _build_context_string(self, chunks: list[RetrievedChunk]) -> str:
        """Build a formatted context string from retrieved chunks."""
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            header = f"[Section: {chunk.header_context}]" if chunk.header_context else ""
            context_parts.append(
                f"--- Context {i} {header} ---\n{chunk.text}\n"
            )
        return "\n".join(context_parts)
    
    def _extract_citations(self, sources: list[RetrievedChunk]) -> list[str]:
        """Extract unique citations from sources."""
        citations = []
        seen = set()
        for chunk in sources:
            if chunk.filename not in seen:
                seen.add(chunk.filename)
                citations.append(chunk.filename)
        return citations
    
    def _estimate_confidence(self, sources: list[RetrievedChunk]) -> float:
        """Estimate confidence based on retrieval quality."""
        if not sources:
            return 0.0
        
        # Lower distance = higher similarity = higher confidence
        # Distance typically ranges from 0 (identical) to ~2 (very different)
        avg_distance = sum(c.distance for c in sources) / len(sources)
        
        # Convert distance to confidence (0-1 scale)
        # Using a sigmoid-like mapping: distance 0 -> 1.0, distance 1.5 -> 0.1
        confidence = max(0.0, min(1.0, 1.0 - (avg_distance / 1.2)))
        
        return round(confidence, 2)
    
    async def retrieve(self, query: str, domain: str, top_k: int = 5) -> list[RetrievedChunk]:
        """
        Retrieve relevant chunks from the vector store.
        
        Args:
            query: The user's question
            domain: The topic/domain to search in
            top_k: Number of chunks to retrieve
            
        Returns:
            List of RetrievedChunk objects
        """
        results = await search_vector_store(
            query=query,
            domain=domain,
            limit=top_k
        )
        
        chunks = []
        for result in results:
            chunks.append(
                RetrievedChunk(
                    text=result["text"],
                    chunk_id=result["chunk_id"],
                    filename=result["metadata"].get("filename", "Unknown"),
                    header_context=result["metadata"].get("header_context", ""),
                    distance=result.get("distance", 1.0)
                )
            )
        
        return chunks
    
    async def generate(
        self,
        query: str,
        context_chunks: list[RetrievedChunk],
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> str:
        """
        Generate an answer using the LLM with the provided context.
        
        Args:
            query: The user's question
            context_chunks: Retrieved context chunks
            system_prompt: Custom system prompt (optional)
            
        Returns:
            Generated answer string
        """
        if not context_chunks:
            return "I don't have enough information to answer that question based on the available materials. Could you try rephrasing or ask about a different topic?"
        
        context = self._build_context_string(context_chunks)
        
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"""Based on the following context, please answer the question.

Context:
{context}

Question: {query}

Answer:""")
        ])
        
        # Invoke the LLM
        response = await self.llm.ainvoke(prompt.format())
        
        return response.content
    
    async def chat(
        self,
        query: str,
        domain: str = None,
        top_k: int = 5,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> AgentResponse:
        """
        Full RAG chat pipeline: retrieve context and generate answer.
        
        Args:
            query: The user's question
            domain: The topic/domain to search in (uses default if not provided)
            top_k: Number of chunks to retrieve
            system_prompt: Custom system prompt
            
        Returns:
            AgentResponse with answer, citations, sources, and confidence
        """
        domain = domain or settings.default_domain
        
        # Step 1: Retrieve relevant context
        sources = await self.retrieve(query, domain, top_k)
        
        # Step 2: Generate answer from context
        answer = await self.generate(query, sources, system_prompt)
        
        # Step 3: Extract citations and confidence
        citations = self._extract_citations(sources)
        confidence = self._estimate_confidence(sources)
        
        return AgentResponse(
            answer=answer,
            citations=citations,
            sources=sources,
            confidence=confidence
        )


# Singleton instance for reuse
_agent: RAGAgent | None = None


def get_rag_agent(
    model_name: str = "gemma4:e2b",
    temperature: float = 0.7,
    max_tokens: int = 1024,
) -> RAGAgent:
    """Get or create the RAG agent singleton."""
    global _agent
    if _agent is None:
        _agent = RAGAgent(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    return _agent
