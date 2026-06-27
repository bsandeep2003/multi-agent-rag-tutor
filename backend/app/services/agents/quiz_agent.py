"""Quiz agent — generates practice problems and assessments."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.services.llm_provider import LLMProvider
from app.services.agents.retrieval_agent import RetrievedChunk
from app.core.config import settings


@dataclass
class QuizResult:
    """Result from the quiz agent."""
    question: str
    question_type: str  # multiple_choice, code_trace, open_ended, fill_blank
    options: list[str] | None
    correct_answer: str
    explanation: str
    hint: str
    difficulty: str
    topic: str
    citations: list[str]


QUIZ_SYSTEM_PROMPT = """You are a Quiz Generator Agent for a CS tutor. You create practice problems that test true understanding, not just memorization.

Question Types:
1. **Code Trace** — Give a short code snippet and ask what the output/result is. This tests whether the student can mentally simulate execution.
2. **Multiple Choice** — 3-4 options with ONE clearly correct answer and distractors that test common misconceptions.
3. **Fill in the Blank** — A code snippet with a missing line. The student must supply the correct logic.
4. **Open-Ended** — "What would happen if...?" or "Explain why..." — tests deep understanding.

Rules:
- Questions must be based ONLY on the provided context
- Distractors should be plausible but wrong (test common misconceptions)
- Include a helpful hint that doesn't give away the answer
- Provide the correct answer with a brief explanation of WHY it's correct
- Adjust difficulty based on the student's level
- Make questions engaging and relevant to real programming scenarios"""


class QuizAgent:
    """Quiz Agent — Generates practice problems based on retrieved content."""

    def __init__(self):
        self.llm = LLMProvider(
            provider=settings.llm_provider,
            model_path=settings.llm_model,
            temperature=0.8,
            max_tokens=1024,
            openai_api_key=settings.openai_api_key,
            openai_model=settings.openai_model,
            openai_base_url=settings.openai_base_url,
            groq_api_key=settings.groq_api_key,
            groq_model=settings.groq_model,
            ollama_base_url=settings.ollama_base_url,
        )

    async def generate_question(
        self,
        topic: str,
        chunks: list[RetrievedChunk],
        difficulty: str = "intermediate",
        question_type: str | None = None,
    ) -> QuizResult:
        """Generate a practice question based on the topic and context."""
        if not chunks:
            return QuizResult(
                question="I need more material on this topic to generate a good question. Please upload a textbook or notes!",
                question_type="open_ended",
                options=None,
                correct_answer="",
                explanation="Upload material to get practice questions.",
                hint="Try uploading a PDF about this topic.",
                difficulty=difficulty,
                topic=topic,
                citations=[],
            )

        context = "\n\n".join(
            f"[Source: {c.filename}] {c.header_context or 'Section'}:\n{c.text}"
            for c in chunks[:3]
        )

        qtype = question_type or self._pick_question_type(difficulty)

        prompt = f"""Create a {difficulty} {qtype} question about: {topic}

Use ONLY this material to create the question:

{context}

Return ONLY a JSON object with these exact fields:
{{
    "question": "the question text",
    "question_type": "multiple_choice|code_trace|open_ended|fill_blank",
    "options": ["A. option1", "B. option2", "C. option3", "D. option4"],
    "correct_answer": "A",
    "explanation": "why this is correct",
    "hint": "a helpful hint without giving the answer",
    "difficulty": "beginner|intermediate|advanced"
}}"""

        try:
            response = await self.llm.ainvoke([
                SystemMessage(content=QUIZ_SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ])
            import json
            content = response.content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[1].rsplit("\n", 1)[0]
                if content.startswith("json"):
                    content = content[4:].strip()
            result = json.loads(content)

            return QuizResult(
                question=result.get("question", "What is the key concept?"),
                question_type=result.get("question_type", qtype),
                options=result.get("options"),
                correct_answer=result.get("correct_answer", ""),
                explanation=result.get("explanation", ""),
                hint=result.get("hint", "Think about the core concept."),
                difficulty=result.get("difficulty", difficulty),
                topic=topic,
                citations=list(set(c.filename for c in chunks)),
            )
        except Exception:
            # Fallback question
            return QuizResult(
                question=f"Based on the material about {topic}, can you explain the core concept in your own words?",
                question_type="open_ended",
                options=None,
                correct_answer="Any correct explanation referencing the key concepts.",
                explanation="The core concept involves understanding the fundamental principles described in the material.",
                hint="Think about the main definition and its key properties.",
                difficulty=difficulty,
                topic=topic,
                citations=list(set(c.filename for c in chunks)),
            )

    def _pick_question_type(self, difficulty: str) -> str:
        """Pick a question type based on difficulty."""
        import random
        if difficulty == "beginner":
            return random.choice(["multiple_choice", "code_trace"])
        elif difficulty == "advanced":
            return random.choice(["open_ended", "fill_blank"])
        return random.choice(["code_trace", "multiple_choice"])
