"""Teacher agent — pedagogical explanation with structured teaching."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.services.llm_provider import LLMProvider
from app.services.agents.retrieval_agent import RetrievedChunk
from app.services.agents.planner_agent import LessonPlan
from app.core.config import settings


@dataclass
class TeachingResult:
    """Result from the teacher agent."""
    explanation: str
    key_points: list[str]
    analogy: str
    code_example: str | None
    common_pitfalls: list[str]
    practice_hint: str
    citations: list[str]
    confidence: float


TEACHER_SYSTEM_PROMPT = """You are an exceptional Computer Science tutor named "Prof". You are patient, encouraging, and brilliant at making complex concepts click.

Teaching Principles:
1. ALWAYS start with intuition — never with formal definitions
2. Use vivid analogies (stack of plates, library card catalog, etc.)
3. Walk through code line by line with inline comments
4. Bold key terms when first introduced
5. Anticipate confusion and address it before it happens
6. End every explanation with a "Try This Yourself" prompt
7. Use encouraging language: "You can do this!", "This is a common point of confusion..."
8. Structure everything into clear sections with headers

Format your response like this:

## The Big Picture
(1-2 sentences: why does this matter?)

## The Intuition
(A vivid, everyday analogy that makes the concept click)

## The Technical Details
- **Key Term**: Definition
- **Key Term**: Definition
(Use bullet points, short paragraphs)

## Code Example
```python
# Walk through this line by line
```

## Common Pitfalls
"🚨 Students often think X, but actually Y..."

## Try This Yourself
"💡 Can you tell me: what would happen if we changed Z to W?"

## Where to Go Next
(Connect to related concepts)"""


class TeacherAgent:
    """Teacher Agent — Generates pedagogical explanations from retrieved context."""

    def __init__(self):
        self.llm = LLMProvider(
            provider=settings.llm_provider,
            model_path=settings.llm_model,
            temperature=0.7,
            max_tokens=2048,
            openai_api_key=settings.openai_api_key,
            openai_model=settings.openai_model,
            openai_base_url=settings.openai_base_url,
            groq_api_key=settings.groq_api_key,
            groq_model=settings.groq_model,
            ollama_base_url=settings.ollama_base_url,
        )

    async def teach(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        lesson_plan: Any | None = None,
        difficulty: str = "intermediate",
    ) -> TeachingResult:
        """Generate a structured pedagogical explanation."""
        if not chunks:
            return TeachingResult(
                explanation="I don't have enough material on this topic yet. Please upload a textbook or notes about it!",
                key_points=[],
                analogy="",
                code_example=None,
                common_pitfalls=[],
                practice_hint="Try uploading a PDF or text file about this topic.",
                citations=[],
                confidence=0.0,
            )

        context = "\n\n".join(
            f"[Source: {c.filename}] {c.header_context or 'Section'}:\n{c.text}"
            for c in chunks[:5]
        )

        lesson_info = ""
        if lesson_plan:
            lesson_info = f"""
Lesson Plan: {lesson_plan.topic}
Type: {lesson_plan.lesson_type.value}
Objectives: {', '.join(lesson_plan.objectives)}
Prerequisites: {', '.join(lesson_plan.prerequisites)}
"""

        prompt = f"""A student asked: "{query}"

{lesson_info}

Use ONLY the following material to teach them. Do not make up information.

{context}

Now, teach this student in your warm, encouraging tutor voice. Follow the exact structure from your system prompt."""

        try:
            response = await self.llm.ainvoke([
                SystemMessage(content=TEACHER_SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ])
            explanation = response.content
        except Exception:
            # Fallback: assemble from chunks
            explanation = self._build_fallback_teaching(query, chunks)

        citations = list(set(c.filename for c in chunks))

        return TeachingResult(
            explanation=explanation,
            key_points=self._extract_key_points(explanation),
            analogy=self._extract_analogy(explanation),
            code_example=self._extract_code(explanation),
            common_pitfalls=self._extract_pitfalls(explanation),
            practice_hint=self._extract_practice_hint(explanation),
            citations=citations,
            confidence=0.8 if len(chunks) >= 3 else 0.5,
        )

    def _build_fallback_teaching(self, query: str, chunks: list[RetrievedChunk]) -> str:
        lines = [f"## What I Found About {query}", ""]
        for c in chunks[:5]:
            lines.append(f"**{c.header_context or 'Source'}**")
            text = " ".join(c.text.strip().split())
            if len(text) > 800:
                text = text[:800] + "..."
            lines.append(text)
            lines.append("")
        lines.append("*Note: This is assembled from your uploaded documents. The full tutor explanation requires the LLM to be available.*")
        return "\n".join(lines)

    def _extract_key_points(self, text: str) -> list[str]:
        """Extract key points from the explanation."""
        points = []
        for line in text.split("\n"):
            if line.strip().startswith("- **") or line.strip().startswith("* **"):
                points.append(line.strip().lstrip("- *").strip())
        return points[:5]

    def _extract_analogy(self, text: str) -> str:
        """Find the analogy section."""
        in_analogy = False
        analogy_lines = []
        for line in text.split("\n"):
            if "intuition" in line.lower() or "analogy" in line.lower():
                in_analogy = True
                continue
            if in_analogy and line.strip().startswith("##"):
                break
            if in_analogy and line.strip():
                analogy_lines.append(line)
        return " ".join(analogy_lines).strip()[:300]

    def _extract_code(self, text: str) -> str | None:
        """Extract code blocks."""
        if "```" in text:
            parts = text.split("```")
            if len(parts) >= 2:
                return parts[1].strip()
        return None

    def _extract_pitfalls(self, text: str) -> list[str]:
        """Extract common pitfalls."""
        pitfalls = []
        in_pitfalls = False
        for line in text.split("\n"):
            if "pitfall" in line.lower() or "mistake" in line.lower() or "common" in line.lower():
                in_pitfalls = True
                continue
            if in_pitfalls and line.strip().startswith("##"):
                break
            if in_pitfalls and line.strip():
                pitfalls.append(line.strip())
        return pitfalls[:3]

    def _extract_practice_hint(self, text: str) -> str:
        """Extract the practice hint."""
        for line in text.split("\n"):
            if "try this" in line.lower() or "practice" in line.lower() or "can you" in line.lower():
                if len(line) > 20:
                    return line.strip()
        return "Can you explain this concept back to me in your own words?"
