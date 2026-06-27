"""Multi-agent tutoring system.

This module provides a team of specialized agents for pedagogical tutoring:
- PlannerAgent: Analyzes student profile and creates lesson plans
- RetrievalAgent: Retrieves relevant context from vector store
- TeacherAgent: Generates structured pedagogical explanations
- QuizAgent: Creates practice problems and assessments
- CriticAgent: Evaluates student answers with constructive feedback
- LearnerProfile: Tracks student knowledge state and progress
- TutorOrchestrator: Coordinates the 6-agent tutoring flow
"""

from app.services.agents.planner_agent import PlannerAgent, LessonPlan, LessonType
from app.services.agents.retrieval_agent import RetrievalAgent, RetrievedChunk, RetrievalResult
from app.services.agents.teacher_agent import TeacherAgent, TeachingResult
from app.services.agents.quiz_agent import QuizAgent, QuizResult
from app.services.agents.critic_agent import CriticAgent, EvaluationResult
from app.services.agents.orchestrator import TutorOrchestrator, TutorResponse, get_orchestrator, reset_orchestrator

__all__ = [
    # Planner
    "PlannerAgent",
    "LessonPlan",
    "LessonType",
    # Retrieval
    "RetrievalAgent",
    "RetrievedChunk",
    "RetrievalResult",
    # Teacher
    "TeacherAgent",
    "TeachingResult",
    # Quiz
    "QuizAgent",
    "QuizResult",
    # Critic
    "CriticAgent",
    "EvaluationResult",
    # Orchestrator
    "TutorOrchestrator",
    "TutorResponse",
    "get_orchestrator",
    "reset_orchestrator",
]
