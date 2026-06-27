"""Critic agent — evaluates student answers and provides feedback."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.services.llm_provider import LLMProvider
from app.services.learner_profile import LearnerProfile
from app.core.config import settings


@dataclass
class EvaluationResult:
    """Result from the critic agent."""
    is_correct: bool
    score: float  # 0.0 to 1.0
    feedback: str
    what_was_right: str
    what_was_wrong: str
    how_to_improve: str
    related_concept: str
    should_retry: bool
    next_question_hint: str


CRITIC_SYSTEM_PROMPT = """You are a Critic Agent for an intelligent CS tutor. You evaluate student answers with constructive, encouraging feedback.

Your evaluation style:
- ALWAYS start with something the student did RIGHT (even if mostly wrong)
- Be specific about what was wrong and why
- Explain the correct reasoning clearly
- Suggest exactly how to improve next time
- NEVER be harsh or discouraging — this is a safe learning space
- Use the "sandwich" method: praise → correction → encouragement

Scoring Guide:
- 1.0: Perfect answer with clear reasoning
- 0.8-0.9: Correct with minor gaps or unclear explanation
- 0.6-0.7: Partially correct — understood the main idea but missed details
- 0.4-0.5: Incorrect approach but showed some understanding
- 0.2-0.3: Significant misunderstanding — needs re-teaching
- 0.0-0.1: Completely wrong or no attempt"""


class CriticAgent:
    """Critic Agent — Evaluates student answers and provides constructive feedback."""

    def __init__(self):
        self.llm = LLMProvider(
            provider=settings.llm_provider,
            model_path=settings.llm_model,
            temperature=0.3,
            max_tokens=1024,
            openai_api_key=settings.openai_api_key,
            openai_model=settings.openai_model,
            openai_base_url=settings.openai_base_url,
            groq_api_key=settings.groq_api_key,
            groq_model=settings.groq_model,
            ollama_base_url=settings.ollama_base_url,
        )

    async def evaluate_answer(
        self,
        question: str,
        student_answer: str,
        correct_answer: str,
        context: str = "",
        topic: str = "",
    ) -> EvaluationResult:
        """Evaluate a student's answer and provide feedback."""

        prompt = f"""Evaluate this student's answer.

Topic: {topic}

Question: {question}

Correct Answer: {correct_answer}

Student's Answer: {student_answer}

Context from material: {context[:1500] if context else 'N/A'}

Return ONLY a JSON object:
{{
    "is_correct": true|false,
    "score": 0.0-1.0,
    "feedback": "overall encouraging feedback",
    "what_was_right": "what they got right",
    "what_was_wrong": "what they got wrong and why",
    "how_to_improve": "specific advice for next time",
    "related_concept": "a related concept they should study",
    "should_retry": true|false,
    "next_question_hint": "hint for the next question"
}}"""

        try:
            response = await self.llm.ainvoke([
                SystemMessage(content=CRITIC_SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ])
            import json
            content = response.content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[1].rsplit("\n", 1)[0]
                if content.startswith("json"):
                    content = content[4:].strip()
            result = json.loads(content)

            return EvaluationResult(
                is_correct=result.get("is_correct", False),
                score=result.get("score", 0.0),
                feedback=result.get("feedback", "Good effort!"),
                what_was_right=result.get("what_was_right", ""),
                what_was_wrong=result.get("what_was_wrong", ""),
                how_to_improve=result.get("how_to_improve", ""),
                related_concept=result.get("related_concept", topic),
                should_retry=result.get("should_retry", True),
                next_question_hint=result.get("next_question_hint", ""),
            )
        except Exception:
            # Fallback evaluation
            student_lower = student_answer.lower().strip()
            correct_lower = correct_answer.lower().strip()
            is_correct = student_lower == correct_lower or correct_lower in student_lower
            score = 1.0 if is_correct else 0.0

            if is_correct:
                feedback = "✅ Correct! Well done — you clearly understood the concept."
                what_right = "Your answer matches the expected solution."
                what_wrong = "Nothing! You nailed it."
                how_improve = "Try a slightly harder question to challenge yourself."
                should_retry = False
            else:
                feedback = "❌ Not quite right — but that's okay! This is how we learn."
                what_right = "You attempted the question, which is the first step!"
                what_wrong = f"The expected answer was: {correct_answer}"
                how_improve = "Review the concept and try again with the hint provided."
                should_retry = True

            return EvaluationResult(
                is_correct=is_correct,
                score=score,
                feedback=feedback,
                what_was_right=what_right,
                what_was_wrong=what_wrong,
                how_to_improve=how_improve,
                related_concept=topic,
                should_retry=should_retry,
                next_question_hint="Think about the key properties of the concept.",
            )
