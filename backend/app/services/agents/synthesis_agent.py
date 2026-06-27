"""Synthesis agent - generates and refines answers with pedagogical quality."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import Enum

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from app.services.agents.retrieval_agent import RetrievedChunk
from app.services.agents.router_agent import QueryType, Complexity
from app.services.llm_provider import LLMProvider


class Tone(Enum):
    """Tone options for the response."""
    EDUCATIONAL = "educational"
    CONCISE = "concise"
    DETAILED = "detailed"
    SOCRATIC = "socratic"


@dataclass
class SynthesisResult:
    """Result from the synthesis agent."""
    answer: str
    citations: list[str]
    confidence: float
    needs_followup: bool
    suggested_followups: list[str]


# Pedagogical tutor system prompt
TUTOR_SYSTEM_PROMPT = """You are a brilliant, patient, and encouraging computer science tutor. You help students truly *understand* concepts, not just memorize facts.

Your teaching style:
- Start with intuition and real-world analogies before diving into technical details
- Use the "Explain Like I'm Five → Explain Like I'm a Student → Explain Like I'm an Expert" progression
- Provide concrete code examples and walk through them step-by-step
- Anticipate common misconceptions and address them proactively
- Ask reflective questions to check understanding
- Celebrate small wins and encourage the student
- Connect new concepts to things the student already knows
- Offer practice problems with increasing difficulty

STRUCTURE every answer like this:
1. **The Big Picture** (1-2 sentences): What's this about, and why should I care?
2. **The Intuition** (analogy or real-world example): Make it click mentally first
3. **The Technical Details** (definitions, properties, mechanics): Now we get precise
4. **Code Walkthrough** (if applicable): A concrete example with line-by-line explanation
5. **Common Pitfalls** (2-3 mistakes students often make): Prevent confusion before it happens
6. **Practice Checkpoint** (1-2 questions): "Can you tell me why...?" or "What would happen if...?"
7. **Where to Go Next** (connections): How this connects to other topics

Rules:
- ONLY use information from the provided context — do not hallucinate
- When using context, cite the source with [Source: filename]
- If the context is insufficient, say so honestly and suggest what to study first
- Use encouraging, conversational language — not dry textbook prose
- Bold key terms when first introduced
- Keep paragraphs short (3-5 sentences max) for readability
- Use code blocks for any code, with inline comments explaining each line"""


# Query-type specific tutor prompts
TUTOR_QUERY_PROMPTS = {
    QueryType.CONCEPTUAL: '''This is a conceptual question. Teach it like this:

1. **The Big Picture**: Why does this concept exist? What problem does it solve?
2. **The Intuition**: Use an everyday analogy (e.g., 'A stack is like a stack of plates...')
3. **The Definition**: Formal definition with key properties
4. **Visual/Concrete Example**: Show how it works with a diagram or simple example
5. **Common Confusion**: 'Students often think X, but actually it is Y because...'
6. **Practice Checkpoint**: Ask a question that tests whether they truly understand (not just memorized)
7. **Where to Go Next**: Connect to related concepts they will learn next

Goal: The student should feel like they just had an aha moment.''',

    QueryType.PROCEDURAL: '''This is a procedural question. Teach it like this:

1. **The Big Picture**: What are we trying to accomplish, and why?
2. **The Strategy**: High-level approach before diving into steps
3. **Step-by-Step Walkthrough**: Numbered steps, each with a one-line explanation of WHY we do it
4. **Code Example**: A clean, commented implementation
5. **Common Mistakes**: 'Do not forget to...' / 'Watch out for...'
6. **Practice Checkpoint**: Give a similar problem and ask them to trace through it
7. **Alternative Approaches**: Briefly mention if there is another way to do it

Goal: The student should feel confident they can do this themselves.''',

    QueryType.PROBLEM_SOLVING: '''This is a problem-solving question. Teach it like this:

1. **Understand the Problem**: Restate what we are solving in plain English
2. **Brute Force First**: What is the naive approach? (Even if inefficient, it builds intuition)
3. **The Insight**: What pattern or optimization makes this work better?
4. **Step-by-Step Solution**: Walk through the algorithm with a concrete example
5. **Code + Trace**: Show code, then trace it on a small example step by step
6. **Complexity Analysis**: Time and space complexity, with brief justification
7. **Common Mistakes**: Off-by-one errors, edge cases, wrong initialization, etc.
8. **Practice Checkpoint**: A similar problem with a hint

Goal: The student should see the pattern and feel ready to solve similar problems.''',

    QueryType.COMPARISON: '''This is a comparison question. Teach it like this:

1. **When to Use Each**: Start with the decision criteria — when would I pick A vs B?
2. **Side-by-Side Comparison**: A clear comparison table or structured list
3. **Deep Dive on Differences**: The 2-3 most important differences explained in detail
4. **Code Example**: Show both approaches on the same problem
5. **Trade-offs**: Time vs space, simplicity vs efficiency, etc.
6. **Common Confusion**: 'Students often think X is always better, but...'
7. **Practice Checkpoint**: 'Given this scenario, which would you choose and why?'

Goal: The student should know exactly WHEN to use each option.''',

    QueryType.EXAMPLE: '''This is an example request. Teach it like this:

1. **What We are Demonstrating**: What concept does this example illustrate?
2. **The Setup**: What data/structure are we working with?
3. **The Walkthrough**: Step-by-step execution with state at each step
4. **The Code**: Clean, commented implementation
5. **Why It Works**: The key insight that makes this example tick
6. **Variations**: 'What if we changed X to Y?' — explore edge cases
7. **Practice Checkpoint**: Ask them to predict the output of a slightly modified version

Goal: The student should be able to trace through every step with confidence.''',

    QueryType.CLARIFICATION: '''This is a clarification question. The student is confused. Teach it like this:

1. **Acknowledge the Confusion**: 'This is a really common point of confusion — you are not alone!'
2. **The Misconception**: What are they likely thinking? Address it directly.
3. **The Correct Intuition**: Explain it differently from the textbook — use a fresh analogy
4. **Concrete Example**: Show a case where the misconception fails and the correct view works
5. **The Rule and Why**: Formal explanation with the reasoning behind it
6. **Memory Aid**: A mnemonic, pattern, or heuristic to remember it
7. **Practice Checkpoint**: A quick check: 'So if I gave you X, what would you say?'

Goal: The student should realize what they were misunderstanding and feel clear now.''',
}


class SynthesisAgent:
    """
    Synthesis Agent — Generates pedagogical, tutor-quality answers from retrieved context.
    
    Responsibilities:
    - Transform retrieved chunks into structured, engaging lessons
    - Apply query-type-specific teaching strategies
    - Ensure proper citations
    - Assess confidence and suggest meaningful follow-ups
    - Refine answers for clarity and depth
    """
    
    def __init__(
        self,
        model_name: str = "gemma4:e2b",
        temperature: float = 0.7,
        max_tokens: int = 2048,
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
    
    def _build_context_string(self, chunks: list[RetrievedChunk]) -> str:
        """Build a formatted context string from retrieved chunks."""
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            header = f"[Section: {chunk.header_context}]" if chunk.header_context else ""
            source = f"[Source: {chunk.filename}]"
            context_parts.append(
                f"--- Context {i} {header} {source} ---\n{chunk.text}\n"
            )
        return "\n".join(context_parts)
    
    def _extract_citations(self, chunks: list[RetrievedChunk]) -> list[str]:
        """Extract unique citations from sources."""
        citations = []
        seen = set()
        for chunk in chunks:
            if chunk.filename not in seen:
                seen.add(chunk.filename)
                citations.append(chunk.filename)
        return citations
    
    async def synthesize(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        query_type: QueryType = QueryType.CONCEPTUAL,
        complexity: Complexity = Complexity.MODERATE,
        refine: bool = False,
    ) -> SynthesisResult:
        """Synthesize a tutor-quality answer from retrieved chunks."""
        if not chunks:
            return SynthesisResult(
                answer=(
                    "I don't have enough material to answer that well yet. "
                    "Could you try uploading a textbook or notes on this topic? "
                    "Or rephrase your question to match what I've already learned!"
                ),
                citations=[],
                confidence=0.0,
                needs_followup=True,
                suggested_followups=[
                    "Upload a PDF or text file about this topic",
                    "Ask about something covered in your uploaded materials",
                    "Try a more specific question",
                ],
            )
        
        context = self._build_context_string(chunks)
        tutor_guidance = TUTOR_QUERY_PROMPTS.get(query_type, "")
        
        # Adjust depth based on complexity
        if complexity == Complexity.SIMPLE:
            depth = "Keep it approachable. Use simple analogies. Avoid heavy jargon. One clear example is enough."
        elif complexity == Complexity.COMPLEX:
            depth = "Go deep. Multiple examples, edge cases, and connections to advanced topics. Don't shy away from nuance."
        else:
            depth = "Balanced depth. One strong analogy, one solid example, and the key technical details."
        
        prompt = f"""A student just asked: "{query}"

{tutor_guidance}

Depth level: {depth}

Here is the material from their uploaded documents to work with:

{context}

---

Now, teach this student in your warm, engaging tutor voice. Follow the structure above. Use bold for key terms. Use code blocks for any code. Cite sources with [Source: filename]."""
        
        # Generate with error handling
        llm_error_msg = None
        try:
            response = await self.llm.ainvoke(
                [SystemMessage(content=TUTOR_SYSTEM_PROMPT),
                 HumanMessage(content=prompt)]
            )
            answer = response.content
        except Exception as llm_error:
            llm_error_msg = str(llm_error)
            answer = self._build_fallback_answer(query, chunks, query_type, complexity, llm_error_msg)
        
        # Refine if enabled and no error
        if refine and len(chunks) >= 3 and not llm_error_msg:
            try:
                answer = await self._refine_answer(answer, query, chunks)
            except Exception:
                pass
        
        citations = self._extract_citations(chunks)
        confidence = self._estimate_confidence(chunks)
        if llm_error_msg:
            confidence = 0.3
        
        needs_followup = confidence < 0.5 or len(chunks) < 2 or bool(llm_error_msg)
        suggested_followups = self._generate_followups(query, chunks, query_type)
        
        if llm_error_msg and "llama-cpp-python" in llm_error_msg:
            suggested_followups.insert(0, "Install llama-cpp-python: pip install llama-cpp-python")
        
        return SynthesisResult(
            answer=answer,
            citations=citations,
            confidence=confidence,
            needs_followup=needs_followup,
            suggested_followups=suggested_followups,
        )
    
    def _build_fallback_answer(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        query_type: QueryType,
        complexity: Complexity,
        error_msg: str | None = None,
    ) -> str:
        """Build a readable answer from chunks when the LLM is unavailable."""
        lines: list[str] = []
        
        lines.append(f"Here's what I found about **{query}**:"
                     "\n")
        
        for i, chunk in enumerate(chunks[:5], 1):
            header = chunk.header_context or f"Source {i}"
            text = chunk.text.strip()
            text = " ".join(text.split())
            if len(text) > 1000:
                text = text[:1000] + "..."
            
            lines.append(f"---")
            lines.append(f"**{header}**  [Source: {chunk.filename}]")
            lines.append(text)
        
        lines.append("")
        
        if error_msg and "llama-cpp-python" in error_msg:
            lines.append(
                "⚠️ **Note:** The LLM model could not be loaded because `llama-cpp-python` is not installed. "
                "The answer above is assembled directly from your uploaded documents.\n\n"
                "**To enable the LLM, run:**\n"
                "```\n"
                "pip install llama-cpp-python\n"
                "```\n"
                "For GPU acceleration:\n"
                "```\n"
                "CMAKE_ARGS=-DLLAMA_CUDA=on pip install llama-cpp-python\n"
                "```"
            )
        elif error_msg:
            lines.append(
                f"⚠️ **Note:** The LLM encountered an error: `{error_msg}`. "
                "The answer above is assembled directly from your uploaded documents."
            )
        else:
            lines.append(
                "*Note: This is a direct summary from your uploaded documents. "
                "For a more synthesized answer, please ensure the LLM is available.*"
            )
        
        return "\n".join(lines)
    
    async def _refine_answer(
        self,
        initial_answer: str,
        query: str,
        chunks: list[RetrievedChunk],
    ) -> str:
        """Refine the answer for better pedagogical quality."""
        refinement_prompt = f"""You are an expert teaching coach. Review this tutor response and improve it.

Student question: {query}

Current response:
{initial_answer}

Improve it by:
1. Making the explanation more intuitive — add a better analogy or real-world connection
2. Ensuring the structure is clear and scannable (use bold headers, bullet points)
3. Adding a "Common Pitfalls" section if missing
4. Adding a "Try This Yourself" practice question at the end
5. Making the tone more encouraging and conversational (less dry, more "I'm sitting next to you explaining this")
6. Fixing any awkward phrasing or overly long sentences

Provide the improved response:"""
        
        response = await self.llm.ainvoke([HumanMessage(content=refinement_prompt)])
        return response.content
    
    def _estimate_confidence(self, chunks: list[RetrievedChunk]) -> float:
        """Estimate confidence based on retrieval quality."""
        if not chunks:
            return 0.0
        avg_relevance = sum(c.relevance_score for c in chunks) / len(chunks)
        count_factor = min(1.0, len(chunks) / 5.0)
        confidence = (avg_relevance * 0.7) + (count_factor * 0.3)
        return round(max(0.0, min(1.0, confidence)), 2)
    
    def _generate_followups(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        query_type: QueryType,
    ) -> list[str]:
        """Generate tutor-style follow-up questions that check understanding."""
        suggestions = []
        
        if query_type == QueryType.CONCEPTUAL:
            suggestions.extend([
                "Can you explain this back to me in your own words?",
                "What would happen if we changed one key assumption?",
                "How is this similar to something you've already learned?",
            ])
        elif query_type == QueryType.PROCEDURAL:
            suggestions.extend([
                "Want to walk through another example together?",
                "What do you think the first step should be for [related problem]?",
                "Can you spot the bug in this variation of the code?",
            ])
        elif query_type == QueryType.PROBLEM_SOLVING:
            suggestions.extend([
                "Ready for a slightly harder version of this problem?",
                "Can you solve it using a different approach this time?",
                "What would the time complexity be if we changed the data structure?",
            ])
        elif query_type == QueryType.COMPARISON:
            suggestions.extend([
                "In what scenario would you definitely pick A over B?",
                "Can you think of a case where the 'worse' option is actually better?",
                "How would you explain this trade-off to a beginner?",
            ])
        elif query_type == QueryType.EXAMPLE:
            suggestions.extend([
                "What do you think would happen if we changed the input to [X]?",
                "Can you trace through this example step by step for me?",
                "Try writing the code yourself — I'll check it!",
            ])
        elif query_type == QueryType.CLARIFICATION:
            suggestions.extend([
                "Does this clear up the confusion, or should I try another angle?",
                "Can you tell me what you thought the answer was before?",
                "Let's try a quick practice question to solidify this — ready?",
            ])
        else:
            suggestions.extend([
                "Does this make sense? Let me know if any part is still unclear!",
                "Want to dive deeper into any part of this?",
                "Shall we do a quick practice problem to lock this in?",
            ])
        
        return suggestions[:3]
    
    async def generate_with_evaluation(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        query_type: QueryType = QueryType.CONCEPTUAL,
        complexity: Complexity = Complexity.MODERATE,
    ) -> SynthesisResult:
        """Generate answer with self-evaluation and potential refinement."""
        result = await self.synthesize(query, chunks, query_type, complexity)
        
        if result.confidence < 0.4 and len(chunks) >= 2:
            refined = await self.synthesize(
                query, chunks, query_type, complexity, refine=True
            )
            if refined.confidence > result.confidence:
                return refined
        
        return result
