"""Learner Profile system — tracks student knowledge, progress, and gaps."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from app.core.config import settings


BASE_DIR = Path(__file__).resolve().parents[2]
PROFILE_DIR = BASE_DIR / "data" / "profiles"
PROFILE_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class TopicMastery:
    """Mastery level for a single topic."""
    topic: str
    domain: str
    level: float = 0.0  # 0.0 to 1.0 (beginner to expert)
    questions_attempted: int = 0
    questions_correct: int = 0
    last_interaction: str = ""
    struggling_with: list[str] = field(default_factory=list)
    mastered_concepts: list[str] = field(default_factory=list)
    
    @property
    def accuracy(self) -> float:
        if self.questions_attempted == 0:
            return 0.0
        return self.questions_correct / self.questions_attempted
    
    def to_dict(self) -> dict:
        return {
            "topic": self.topic,
            "domain": self.domain,
            "level": self.level,
            "questions_attempted": self.questions_attempted,
            "questions_correct": self.questions_correct,
            "accuracy": round(self.accuracy, 2),
            "last_interaction": self.last_interaction,
            "struggling_with": self.struggling_with,
            "mastered_concepts": self.mastered_concepts,
        }


@dataclass
class InteractionRecord:
    """A single interaction with the tutor."""
    timestamp: str
    query: str
    query_type: str
    topic: str
    confidence: float
    needs_followup: bool
    student_answer: str | None = None
    answer_correct: bool | None = None
    
    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "query": self.query,
            "query_type": self.query_type,
            "topic": self.topic,
            "confidence": self.confidence,
            "needs_followup": self.needs_followup,
            "student_answer": self.student_answer,
            "answer_correct": self.answer_correct,
        }


@dataclass
class LearnerProfile:
    """Complete profile of a learner."""
    session_id: str
    created_at: str
    topics: dict[str, TopicMastery] = field(default_factory=dict)
    interactions: list[InteractionRecord] = field(default_factory=list)
    current_lesson: str | None = None
    lesson_plan: list[dict] = field(default_factory=list)
    weak_areas: list[str] = field(default_factory=list)
    strong_areas: list[str] = field(default_factory=list)
    preferred_pace: str = "moderate"  # slow, moderate, fast
    
    def record_interaction(self, interaction: InteractionRecord) -> None:
        self.interactions.append(interaction)
        
        # Update topic mastery if applicable
        if interaction.topic in self.topics:
            topic = self.topics[interaction.topic]
            topic.questions_attempted += 1
            if interaction.answer_correct:
                topic.questions_correct += 1
                topic.level = min(1.0, topic.level + 0.1)
            else:
                topic.level = max(0.0, topic.level - 0.05)
                if interaction.student_answer and interaction.student_answer not in topic.struggling_with:
                    topic.struggling_with.append(interaction.student_answer)
            topic.last_interaction = interaction.timestamp
            
            # Update mastered concepts if level is high
            if topic.level > 0.8 and interaction.topic not in topic.mastered_concepts:
                topic.mastered_concepts.append(interaction.topic)
    
    def get_weak_areas(self, domain: str | None = None) -> list[str]:
        """Get topics where the learner is struggling."""
        weak = []
        for topic_name, topic in self.topics.items():
            if domain and topic.domain != domain:
                continue
            if topic.level < 0.5 or topic.accuracy < 0.6:
                weak.append(topic_name)
        return weak
    
    def get_strong_areas(self, domain: str | None = None) -> list[str]:
        """Get topics where the learner is proficient."""
        strong = []
        for topic_name, topic in self.topics.items():
            if domain and topic.domain != domain:
                continue
            if topic.level > 0.7 and topic.accuracy > 0.75:
                strong.append(topic_name)
        return strong
    
    def get_next_topic(self, domain: str | None = None) -> str | None:
        """Determine the next topic to learn based on gaps and prerequisites."""
        # Priority: weak areas > new topics in plan
        weak = self.get_weak_areas(domain)
        if weak:
            return weak[0]
        
        # If no weak areas, return next topic in lesson plan
        if self.lesson_plan:
            for item in self.lesson_plan:
                if not item.get("completed", False):
                    return item.get("topic")
        
        return None
    
    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "topics": {k: v.to_dict() for k, v in self.topics.items()},
            "interactions": [i.to_dict() for i in self.interactions],
            "current_lesson": self.current_lesson,
            "lesson_plan": self.lesson_plan,
            "weak_areas": self.get_weak_areas(),
            "strong_areas": self.get_strong_areas(),
            "preferred_pace": self.preferred_pace,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "LearnerProfile":
        profile = cls(
            session_id=data["session_id"],
            created_at=data["created_at"],
        )
        profile.topics = {
            k: TopicMastery(**v) for k, v in data.get("topics", {}).items()
        }
        profile.interactions = [
            InteractionRecord(**i) for i in data.get("interactions", [])
        ]
        profile.current_lesson = data.get("current_lesson")
        profile.lesson_plan = data.get("lesson_plan", [])
        profile.preferred_pace = data.get("preferred_pace", "moderate")
        return profile


# In-memory profile store with persistence
_profiles: dict[str, LearnerProfile] = {}


def get_or_create_profile(session_id: str) -> LearnerProfile:
    """Get existing profile or create a new one."""
    if session_id in _profiles:
        return _profiles[session_id]
    
    # Try to load from disk
    profile_path = PROFILE_DIR / f"{session_id}.json"
    if profile_path.exists():
        data = json.loads(profile_path.read_text(encoding="utf-8"))
        profile = LearnerProfile.from_dict(data)
        _profiles[session_id] = profile
        return profile
    
    # Create new profile
    profile = LearnerProfile(
        session_id=session_id,
        created_at=datetime.now().isoformat(),
    )
    _profiles[session_id] = profile
    return profile


def save_profile(session_id: str) -> None:
    """Persist profile to disk."""
    if session_id not in _profiles:
        return
    
    profile = _profiles[session_id]
    profile_path = PROFILE_DIR / f"{session_id}.json"
    profile_path.write_text(
        json.dumps(profile.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def get_profile(session_id: str) -> LearnerProfile | None:
    """Get an existing profile."""
    return _profiles.get(session_id) or get_or_create_profile(session_id)
