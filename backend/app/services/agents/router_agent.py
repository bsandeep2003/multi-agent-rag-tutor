"""Router agent - analyzes queries and determines the best approach."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.services.llm_provider import LLMProvider


class QueryType(Enum):
    """Types of student queries."""
    CONCEPTUAL = "conceptual"
    PROCEDURAL = "procedural"
    PROBLEM_SOLVING = "problem_solving"
    COMPARISON = "comparison"
    EXAMPLE = "example"
    CLARIFICATION = "clarification"


class Complexity(Enum):
    """Complexity level of a query."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


@dataclass
class QueryAnalysis:
    """Analysis result from the router agent."""
    query_type: QueryType
    complexity: Complexity
    keywords: list[str]
    suggested_approach: str
    confidence: float
    reformulated_query: str


ROUTER_SYSTEM_PROMPT = """You are a Query Router Agent. Your job is to analyze student questions and determine:
1. What type of question it is (conceptual, procedural, problem-solving, comparison, example, clarification)
2. How complex it is (simple, moderate, complex)
3. What keywords are relevant
4. How to reformulate the query for better retrieval

Return ONLY a JSON object with these fields:
- query_type: one of [conceptual, procedural, problem_solving, comparison, example, clarification]
- complexity: one of [simple, moderate, complex]
- keywords: list of 3-5 relevant keywords
- suggested_approach: brief description of how to answer
- confidence: float 0.0-1.0
- reformulated_query: improved version for vector search"""


class RouterAgent:
    """
    Router Agent - Analyzes queries and determines the best approach.
    
    Responsibilities:
    - Intent classification
    - Query complexity assessment
    - Keyword extraction
    - Query reformulation for better retrieval
    """
    
    def __init__(
        self,
        model_name: str = "gemma4:e2b",
        temperature: float = 0.3,
        max_tokens: int = 512,
        base_url: str = "http://localhost:11434",
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = base_url
        
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
    
    def _extract_keywords(self, query: str) -> list[str]:
        """Extract keywords from a query using simple heuristics."""
        # Common stop words to filter
        stop_words = {
            "what", "is", "are", "the", "a", "an", "how", "does", "do", "to",
            "in", "on", "at", "for", "of", "with", "about", "explain", "describe",
            "tell", "me", "can", "you", "i", "need", "want", "help", "understand",
            "difference", "between", "compare", "example", "step", "by", "using"
        }
        
        words = query.lower().split()
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        # Deduplicate and limit
        return list(dict.fromkeys(keywords))[:5]
    
    def _determine_complexity(self, query: str) -> Complexity:
        """Determine query complexity based on length and structure."""
        word_count = len(query.split())
        
        # Check for complex indicators
        complex_indicators = [
            "compare", "difference", "versus", "vs", "trade-off", "advantages",
            "disadvantages", "multiple", "complex", "advanced", "deep dive",
            "in detail", "thoroughly", "comprehensive"
        ]
        simple_indicators = [
            "what is", "define", "simple", "basic", "briefly", "quick",
            "short", "one sentence", "in simple terms"
        ]
        
        query_lower = query.lower()
        complex_score = sum(1 for ind in complex_indicators if ind in query_lower)
        simple_score = sum(1 for ind in simple_indicators if ind in query_lower)
        
        if complex_score > 0 or word_count > 20:
            return Complexity.COMPLEX
        elif simple_score > 0 or word_count < 8:
            return Complexity.SIMPLE
        return Complexity.MODERATE
    
    def _determine_query_type(self, query: str) -> QueryType:
        """Determine query type based on keywords."""
        query_lower = query.lower()
        
        if any(w in query_lower for w in ["compare", "versus", "vs", "difference between", "differences"]):
            return QueryType.COMPARISON
        elif any(w in query_lower for w in ["how to", "steps", "step by step", "process", "procedure", "implement"]):
            return QueryType.PROCEDURAL
        elif any(w in query_lower for w in ["solve", "problem", "calculate", "find the", "compute", "evaluate"]):
            return QueryType.PROBLEM_SOLVING
        elif any(w in query_lower for w in ["example", "instance", "sample", "show me", "demonstrate"]):
            return QueryType.EXAMPLE
        elif any(w in query_lower for w in ["clarify", "confused", "don't understand", "what does", "mean by"]):
            return QueryType.CLARIFICATION
        else:
            return QueryType.CONCEPTUAL
    
    def _reformulate_query(self, query: str, query_type: QueryType) -> str:
        """Reformulate query for better vector retrieval."""
        # Simple reformulation based on query type
        if query_type == QueryType.CONCEPTUAL:
            return f"What is {query}? Definition, explanation, and key concepts."
        elif query_type == QueryType.PROCEDURAL:
            return f"How to {query}. Steps, process, and implementation."
        elif query_type == QueryType.PROBLEM_SOLVING:
            return f"Solving {query}. Approach, solution, and explanation."
        elif query_type == QueryType.COMPARISON:
            return f"Comparing {query}. Differences, similarities, and trade-offs."
        elif query_type == QueryType.EXAMPLE:
            return f"Examples of {query}. Demonstrations and use cases."
        elif query_type == QueryType.CLARIFICATION:
            return f"Explanation of {query}. Clarification and simplified explanation."
        return query
    
    def get_top_k(self, analysis: QueryAnalysis) -> int:
        """Determine optimal number of chunks to retrieve based on analysis."""
        base_k = 5
        
        # Adjust based on complexity
        if analysis.complexity == Complexity.COMPLEX:
            base_k += 2
        elif analysis.complexity == Complexity.SIMPLE:
            base_k -= 1
        
        # Adjust based on query type
        if analysis.query_type in (QueryType.COMPARISON, QueryType.PROBLEM_SOLVING):
            base_k += 1
        
        return max(3, min(10, base_k))
    
    async def analyze(self, query: str, domain: str) -> QueryAnalysis:
        """
        Analyze a query and return structured analysis.
        
        This uses the LLM for sophisticated analysis but falls back to
        heuristic-based analysis if the LLM fails.
        """
        try:
            # Try LLM-based analysis first
            prompt = f"""Analyze this student query and return a JSON object.

Domain: {domain}
Query: {query}

Return ONLY a JSON object with these exact fields:
{{
    "query_type": "conceptual|procedural|problem_solving|comparison|example|clarification",
    "complexity": "simple|moderate|complex",
    "keywords": ["keyword1", "keyword2", ...],
    "suggested_approach": "brief approach description",
    "confidence": 0.85,
    "reformulated_query": "improved query for search"
}}"""
            
            response = await self.llm.ainvoke([
                SystemMessage(content=ROUTER_SYSTEM_PROMPT),
                HumanMessage(content=prompt)
            ])
            
            # Parse the response
            import json
            content = response.content.strip()
            # Remove markdown code blocks if present
            if content.startswith("```"):
                content = content.split("\n", 1)[1].rsplit("\n", 1)[0]
                if content.startswith("json"):
                    content = content[4:].strip()
            
            result = json.loads(content)
            
            return QueryAnalysis(
                query_type=QueryType(result.get("query_type", "conceptual")),
                complexity=Complexity(result.get("complexity", "moderate")),
                keywords=result.get("keywords", self._extract_keywords(query)),
                suggested_approach=result.get("suggested_approach", "Standard retrieval and synthesis"),
                confidence=float(result.get("confidence", 0.8)),
                reformulated_query=result.get("reformulated_query", query),
            )
            
        except Exception:
            # Fallback to heuristic-based analysis
            query_type = self._determine_query_type(query)
            complexity = self._determine_complexity(query)
            keywords = self._extract_keywords(query)
            reformulated = self._reformulate_query(query, query_type)
            
            return QueryAnalysis(
                query_type=query_type,
                complexity=complexity,
                keywords=keywords,
                suggested_approach=f"Retrieve relevant content about {', '.join(keywords[:3])} and synthesize a clear answer.",
                confidence=0.75,
                reformulated_query=reformulated,
            )
