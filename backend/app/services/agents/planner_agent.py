"""Planner agent — analyzes student state and creates personalized lesson plans."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.services.llm_provider import LLMProvider
from app.services.learner_profile import LearnerProfile
from app.core.config import settings


class LessonType(Enum):
    """Types of lessons the planner can schedule."""
    NEW_CONCEPT = "new_concept"
    REVIEW = "review"
    PRACTICE = "practice"
    ASSESSMENT = "assessment"
    DEEP_DIVE = "deep_dive"


@dataclass
class LessonPlan:
    """A structured lesson plan for a student."""
    topic: str
    domain: str
    lesson_type: LessonType
    objectives: list[str]
    prerequisites: list[str]
    estimated_time: int
    content_outline: list[dict]
    difficulty: str
    confidence: float

    def to_dict(self) -> dict:
        return {
            "topic": self.topic,
            "domain": self.domain,
            "lesson_type": self.lesson_type.value,
            "objectives": self.objectives,
            "prerequisites": self.prerequisites,
            "estimated_time": self.estimated_time,
            "content_outline": self.content_outline,
            "difficulty": self.difficulty,
            "confidence": self.confidence,
        }


class PlannerAgent:
    """Planner Agent — creates personalized lesson plans based on student profile."""

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

    async def plan_lesson(
        self,
        query: str,
        profile: LearnerProfile,
        domain: str = "Data Structures and Algorithms",
        retrieved_context: str = "",
    ) -> LessonPlan:
        """Create a personalized lesson plan."""
        weak_areas = profile.get_weak_areas(domain)
        strong_areas = profile.get_strong_areas(domain)
        current_level = self._estimate_overall_level(profile, domain)

        prompt = f"""You are a CS tutor planning a personalized lesson.

Student Query: {query}
Domain: {domain}
Student Level: {current_level:.0%} (0=beginner, 1=expert)
Weak Areas: {weak_areas if weak_areas else 'None yet'}
Strong Areas: {strong_areas if strong_areas else 'None yet'}

Context from Knowledge Base:
{retrieved_context[:2000] if retrieved_context else 'No context retrieved.'}

Return ONLY a JSON object:
{{
    "topic": "specific topic",
    "lesson_type": "new_concept|review|practice|deep_dive",
    "objectives": ["list", "of", "objectives"],
    "prerequisites": ["what", "they", "should", "know"],
    "estimated_time": 15,
    "content_outline": [
        {{"section": "Introduction", "description": "why this matters"}},
        {{"section": "Core Concept", "description": "the main idea"}},
        {{"section": "Example", "description": "walkthrough"}},
        {{"section": "Practice", "description": "try it yourself"}}
    ],
    "difficulty": "beginner|intermediate|advanced",
    "confidence": 0.8
}}"""

        try:
            response = await self.llm.ainvoke([
                SystemMessage(content="You are a lesson planner for a CS tutor. Output valid JSON only."),
                HumanMessage(content=prompt),
            ])
            import json
            content = response.content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[1].rsplit("\n", 1)[0]
                if content.startswith("json"):
                    content = content[4:].strip()
            result = json.loads(content)

            return LessonPlan(
                topic=result.get("topic", query),
                domain=domain,
                lesson_type=LessonType(result.get("lesson_type", "new_concept")),
                objectives=result.get("objectives", ["Understand the concept"]),
                prerequisites=result.get("prerequisites", []),
                estimated_time=result.get("estimated_time", 15),
                content_outline=result.get("content_outline", []),
                difficulty=result.get("difficulty", "intermediate"),
                confidence=result.get("confidence", 0.75),
            )
        except Exception:
            # Fallback lesson plan
            return LessonPlan(
                topic=query,
                domain=domain,
                lesson_type=LessonType.NEW_CONCEPT,
                objectives=["Understand the core concept", "Apply it to a simple example"],
                prerequisites=["Basic programming knowledge"],
                estimated_time=15,
                content_outline=[
                    {"section": "Introduction", "description": "What is this and why does it matter?"},
                    {"section": "Core Concept", "description": "The main idea explained simply"},
                    {"section": "Example", "description": "Walk through a concrete example"},
                    {"section": "Practice", "description": "Try it yourself"},
                ],
                difficulty="intermediate",
                confidence=0.6,
            )

    def _estimate_overall_level(self, profile: LearnerProfile, domain: str | None) -> float:
        levels = [t.level for t in profile.topics.values() if not domain or t.domain == domain]
        if not levels:
            return 0.3
        return sum(levels) / len(levels)
