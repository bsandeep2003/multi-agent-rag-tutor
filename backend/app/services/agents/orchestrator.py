"""Multi-agent orchestrator for the pedagogical tutoring system."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from app.services.agents.planner_agent import PlannerAgent, LessonPlan
from app.services.agents.retrieval_agent import RetrievalAgent
from app.services.agents.teacher_agent import TeacherAgent, TeachingResult
from app.services.agents.quiz_agent import QuizAgent, QuizResult
from app.services.agents.critic_agent import CriticAgent, EvaluationResult
from app.services.learner_profile import LearnerProfile, InteractionRecord, get_or_create_profile, save_profile
from app.core.config import settings


@dataclass
class TutorResponse:
    """Complete response from the multi-agent tutoring system."""
    # Lesson plan
    lesson_plan: dict | None
    # Teaching
    explanation: str
    key_points: list[str]
    analogy: str
    code_example: str | None
    common_pitfalls: list[str]
    practice_hint: str
    # Quiz
    quiz_question: dict | None
    # Evaluation (if student answered)
    evaluation: dict | None
    # Meta
    citations: list[str]
    confidence: float
    agent_trace: list[str]
    latency_ms: int


class TutorOrchestrator:
    """
    Multi-Agent Tutor Orchestrator — Coordinates 6 specialized agents for pedagogy.
    
    Flow:
    1. PLANNER → Analyzes query, checks learner profile, creates lesson plan
    2. RETRIEVER → Fetches relevant content from knowledge base
    3. TEACHER → Generates structured pedagogical explanation
    4. QUIZ → Creates a practice problem based on the lesson
    5. CRITIC → Evaluates student answers (when provided)
    6. LEARNER PROFILE → Updates knowledge state after each interaction
    """
    
    def __init__(self):
        self.planner = PlannerAgent()
        self.retriever = RetrievalAgent()
        self.teacher = TeacherAgent()
        self.quiz = QuizAgent()
        self.critic = CriticAgent()
    
    async def teach(
        self,
        query: str,
        domain: str = None,
        session_id: str = None,
    ) -> TutorResponse:
        """
        Main tutoring flow: Plan → Retrieve → Teach → Quiz.
        
        Returns a complete lesson with explanation and practice question.
        """
        import time
        start_time = time.time()
        trace = []
        
        domain = domain or settings.default_domain
        profile = get_or_create_profile(session_id or "anonymous")
        
        # Step 1: Planner — Analyze and create lesson plan
        trace.append("🎯 [orchestrator] Received student query")
        trace.append("📋 [planner] Analyzing student profile and creating lesson plan...")
        
        # Quick retrieval for planner context
        quick_retrieval = await self.retriever.retrieve(query, domain, top_k=3)
        context_for_planner = "\n".join(c.text[:500] for c in quick_retrieval.chunks[:2])
        
        lesson_plan = await self.planner.plan_lesson(query, profile, domain, context_for_planner)
        trace.append(f"📋 [planner] Topic: {lesson_plan.topic}")
        trace.append(f"📋 [planner] Type: {lesson_plan.lesson_type.value}")
        trace.append(f"📋 [planner] Difficulty: {lesson_plan.difficulty}")
        trace.append(f"📋 [planner] Estimated time: {lesson_plan.estimated_time} min")
        
        # Update profile current lesson
        profile.current_lesson = lesson_plan.topic
        
        # Step 2: Retriever — Fetch relevant content
        trace.append("📚 [retriever] Searching knowledge base for relevant material...")
        retrieval_result = await self.retriever.retrieve_with_fallback(
            query=query,
            domain=domain,
        )
        trace.append(f"📚 [retriever] Found {retrieval_result.total_found} relevant chunks")
        
        # Step 3: Teacher — Generate pedagogical explanation
        trace.append("👨‍🏫 [teacher] Preparing structured explanation...")
        teaching = await self.teacher.teach(
            query=query,
            chunks=retrieval_result.chunks,
            lesson_plan=lesson_plan,
            difficulty=lesson_plan.difficulty,
        )
        trace.append(f"👨‍🏫 [teacher] Explanation generated with {len(teaching.key_points)} key points")
        if teaching.analogy:
            trace.append(f"👨‍🏫 [teacher] Analogy: {teaching.analogy[:60]}...")
        if teaching.code_example:
            trace.append("👨‍🏫 [teacher] Code example included")
        
        # Step 4: Quiz — Generate practice question
        trace.append("📝 [quiz] Creating practice question...")
        quiz_result = await self.quiz.generate_question(
            topic=lesson_plan.topic,
            chunks=retrieval_result.chunks,
            difficulty=lesson_plan.difficulty,
        )
        trace.append(f"📝 [quiz] Generated {quiz_result.question_type} question")
        
        # Update profile with this interaction
        profile.record_interaction(InteractionRecord(
            timestamp=datetime.now().isoformat(),
            query=query,
            query_type=lesson_plan.lesson_type.value,
            topic=lesson_plan.topic,
            confidence=teaching.confidence,
            needs_followup=True,
        ))
        save_profile(profile.session_id)
        
        trace.append("💾 [profile] Learner profile updated")
        trace.append("🎯 [orchestrator] Lesson complete!")
        
        latency_ms = int((time.time() - start_time) * 1000)
        
        return TutorResponse(
            lesson_plan=lesson_plan.to_dict(),
            explanation=teaching.explanation,
            key_points=teaching.key_points,
            analogy=teaching.analogy,
            code_example=teaching.code_example,
            common_pitfalls=teaching.common_pitfalls,
            practice_hint=teaching.practice_hint,
            quiz_question={
                "question": quiz_result.question,
                "type": quiz_result.question_type,
                "options": quiz_result.options,
                "correct_answer": quiz_result.correct_answer,
                "explanation": quiz_result.explanation,
                "hint": quiz_result.hint,
                "difficulty": quiz_result.difficulty,
            },
            evaluation=None,
            citations=teaching.citations,
            confidence=teaching.confidence,
            agent_trace=trace,
            latency_ms=latency_ms,
        )
    
    async def evaluate(
        self,
        question: str,
        student_answer: str,
        correct_answer: str,
        topic: str = "",
        context: str = "",
        session_id: str = None,
    ) -> dict:
        """
        Evaluate a student's answer and update their profile.
        """
        trace = []
        trace.append("🔍 [critic] Evaluating student answer...")
        
        evaluation = await self.critic.evaluate_answer(
            question=question,
            student_answer=student_answer,
            correct_answer=correct_answer,
            context=context,
            topic=topic,
        )
        
        trace.append(f"🔍 [critic] Score: {evaluation.score:.0%}")
        trace.append(f"🔍 [critic] {'✅ Correct' if evaluation.is_correct else '❌ Needs work'}")
        
        # Update profile
        profile = get_or_create_profile(session_id or "anonymous")
        profile.record_interaction(InteractionRecord(
            timestamp=datetime.now().isoformat(),
            query=question,
            query_type="assessment",
            topic=topic,
            confidence=evaluation.score,
            needs_followup=evaluation.should_retry,
            student_answer=student_answer,
            answer_correct=evaluation.is_correct,
        ))
        save_profile(profile.session_id)
        trace.append("💾 [profile] Profile updated with assessment result")
        
        return {
            "evaluation": {
                "is_correct": evaluation.is_correct,
                "score": evaluation.score,
                "feedback": evaluation.feedback,
                "what_was_right": evaluation.what_was_right,
                "what_was_wrong": evaluation.what_was_wrong,
                "how_to_improve": evaluation.how_to_improve,
                "related_concept": evaluation.related_concept,
                "should_retry": evaluation.should_retry,
                "next_question_hint": evaluation.next_question_hint,
            },
            "agent_trace": trace,
        }


# Singleton
_orchestrator: TutorOrchestrator | None = None

def get_orchestrator() -> TutorOrchestrator:
    """Get or create the tutor orchestrator singleton."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = TutorOrchestrator()
    return _orchestrator

def reset_orchestrator():
    """Reset the orchestrator."""
    global _orchestrator
    _orchestrator = None
