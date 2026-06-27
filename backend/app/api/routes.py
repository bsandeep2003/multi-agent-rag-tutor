from fastapi import APIRouter, File, Form, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import json
import time

from app.services.ingestion import ingest_document
from app.core.config import settings
from app.services.learner_profile import get_or_create_profile, save_profile

router = APIRouter()


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None
    domain: str | None = None


class EvaluateRequest(BaseModel):
    question: str
    student_answer: str
    correct_answer: str
    domain: str | None = None
    session_id: str | None = None


class TutorResponse(BaseModel):
    # Lesson plan
    lesson_plan: dict | None = Field(default=None)
    # Teaching
    explanation: str
    key_points: list[str] = Field(default_factory=list)
    analogy: str = ""
    code_example: str | None = None
    common_pitfalls: list[str] = Field(default_factory=list)
    practice_hint: str = ""
    # Quiz
    quiz_question: dict | None = Field(default=None)
    # Evaluation (if student answered)
    evaluation: dict | None = Field(default=None)
    # Meta
    citations: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    agent_trace: list[str] = Field(default_factory=list)
    latency_ms: int = 0
    llm_provider: str = "unknown"
    llm_model: str = "unknown"


class UploadResponse(BaseModel):
    document_id: str
    filename: str
    domain: str
    chunk_count: int
    status: str
    message: str


class AgentDetailsResponse(BaseModel):
    agents: list[dict] = Field(default_factory=list)


@router.get("/health")
async def health() -> dict[str, str]:
    from app.core.config import settings
    active_model = settings.groq_model if settings.groq_api_key else (
        settings.openai_model if settings.openai_api_key else settings.llm_model
    )
    return {
        "status": "ok",
        "llm_provider": settings.llm_provider,
        "llm_model": active_model,
    }


@router.get("/agents", response_model=AgentDetailsResponse)
async def get_agents() -> AgentDetailsResponse:
    """Return the 6-agent tutoring system configuration."""
    return AgentDetailsResponse(
        agents=[
            {
                "id": "planner",
                "title": "Planner Agent",
                "subtitle": "Lesson Planner",
                "color": "blue",
                "icon": "fa-map",
                "description": "Analyzes the student's query and knowledge profile, then creates a personalized lesson plan with objectives, prerequisites, and difficulty level.",
                "responsibilities": [
                    "Analyzes student query intent",
                    "Checks learner profile for knowledge gaps",
                    "Creates structured lesson plans",
                    "Decides lesson type (new concept, review, practice, deep dive)",
                    "Identifies prerequisites and learning objectives",
                ],
                "tools": ["Learner Profile DB", "Query Analyzer", "Lesson Template Engine"]
            },
            {
                "id": "retriever",
                "title": "Retriever Agent",
                "subtitle": "Knowledge Fetcher",
                "color": "purple",
                "icon": "fa-database",
                "description": "Performs semantic search over the vector database to find relevant material for the lesson.",
                "responsibilities": [
                    "Converts query to embedding vector",
                    "Performs approximate nearest neighbor search",
                    "Filters by metadata (domain, source, topic)",
                    "Retrieves top-k relevant chunks",
                ],
                "tools": ["Vector DB (ChromaDB)", "Embedding Model", "HNSW Index"]
            },
            {
                "id": "teacher",
                "title": "Teacher Agent",
                "subtitle": "Explainer",
                "color": "emerald",
                "icon": "fa-chalkboard-user",
                "description": "Generates structured pedagogical explanations with analogies, code examples, and common pitfalls.",
                "responsibilities": [
                    "Starts with intuition before technical details",
                    "Uses vivid everyday analogies",
                    "Provides step-by-step code walkthroughs",
                    "Anticipates common misconceptions",
                    "Structures answers into clear sections",
                ],
                "tools": ["LLM (Groq/OpenAI)", "Prompt Templates", "Code Formatter"]
            },
            {
                "id": "quiz",
                "title": "Quiz Agent",
                "subtitle": "Practice Generator",
                "color": "amber",
                "icon": "fa-clipboard-question",
                "description": "Creates practice problems that test true understanding — code traces, multiple choice, fill-in-the-blank, and open-ended questions.",
                "responsibilities": [
                    "Generates code trace questions",
                    "Creates multiple choice with distractors",
                    "Designs fill-in-the-blank problems",
                    "Provides hints without giving answers",
                    "Adjusts difficulty based on student level",
                ],
                "tools": ["Question Generator LLM", "Distractor Builder", "Difficulty Scaler"]
            },
            {
                "id": "critic",
                "title": "Critic Agent",
                "subtitle": "Answer Checker",
                "color": "red",
                "icon": "fa-scale-balanced",
                "description": "Evaluates student answers with constructive, encouraging feedback using the 'praise → correction → encouragement' sandwich method.",
                "responsibilities": [
                    "Scores answers on a 0.0-1.0 scale",
                    "Identifies what was right and what was wrong",
                    "Provides specific improvement advice",
                    "Suggests related concepts to study",
                    "Recommends whether to retry",
                ],
                "tools": ["Answer Scorer", "Feedback Generator", "Concept Mapper"]
            },
            {
                "id": "learner_profile",
                "title": "Learner Profile",
                "subtitle": "Student Model",
                "color": "indigo",
                "icon": "fa-user-graduate",
                "description": "Tracks the student's knowledge state, progress, weak areas, and strong areas across all topics.",
                "responsibilities": [
                    "Records every interaction and assessment",
                    "Tracks topic mastery levels (0.0-1.0)",
                    "Identifies weak and strong areas",
                    "Determines next topic to learn",
                    "Persists data across sessions",
                ],
                "tools": ["Profile Database", "Mastery Tracker", "Progress Analyzer"]
            },
        ]
    )


@router.post("/teach", response_model=TutorResponse)
async def teach(payload: ChatRequest) -> TutorResponse:
    """
    Main tutoring endpoint: Plan → Retrieve → Teach → Quiz.
    Returns a complete lesson with explanation and practice question.
    """
    from app.services.agents import get_orchestrator, reset_orchestrator
    import time

    start_time = time.time()
    reset_orchestrator()

    try:
        orchestrator = get_orchestrator()

        response = await orchestrator.teach(
            query=payload.message,
            domain=payload.domain or settings.default_domain,
            session_id=payload.session_id,
        )

        latency_ms = int((time.time() - start_time) * 1000)

        return TutorResponse(
            lesson_plan=response.lesson_plan,
            explanation=response.explanation,
            key_points=response.key_points,
            analogy=response.analogy,
            code_example=response.code_example,
            common_pitfalls=response.common_pitfalls,
            practice_hint=response.practice_hint,
            quiz_question=response.quiz_question,
            evaluation=response.evaluation,
            citations=response.citations,
            confidence=response.confidence,
            agent_trace=response.agent_trace,
            latency_ms=latency_ms,
            llm_provider=settings.llm_provider,
            llm_model=settings.groq_model if settings.groq_api_key else (settings.openai_model if settings.openai_api_key else settings.llm_model),
        )

    except Exception as e:
        latency_ms = int((time.time() - start_time) * 1000)
        raise HTTPException(status_code=500, detail=f"Tutoring pipeline failed: {str(e)}")


@router.post("/evaluate")
async def evaluate(payload: EvaluateRequest) -> dict:
    """
    Evaluate a student's answer to a quiz question.
    Updates the learner profile with the result.
    """
    from app.services.agents import get_orchestrator, reset_orchestrator

    reset_orchestrator()

    try:
        orchestrator = get_orchestrator()

        result = await orchestrator.evaluate(
            question=payload.question,
            student_answer=payload.student_answer,
            correct_answer=payload.correct_answer,
            topic=payload.domain or settings.default_domain,
            session_id=payload.session_id,
        )

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")


@router.get("/profile/{session_id}")
async def get_profile(session_id: str) -> dict:
    """Get the learner profile for a session."""
    profile = get_or_create_profile(session_id)
    return profile.to_dict()


@router.post("/documents/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    domain: str = Form(default="Data Structures and Algorithms"),
) -> UploadResponse:
    result = await ingest_document(file=file, domain=domain)
    return UploadResponse(**result.model_dump())


@router.get("/documents")
async def list_documents():
    """List all uploaded documents."""
    from app.services.ingestion import get_all_documents
    docs = await get_all_documents()
    return {"documents": docs, "count": len(docs)}


@router.get("/documents/{document_id}")
async def get_document_detail(document_id: str):
    """Get details of a specific uploaded document."""
    from app.services.ingestion import get_document
    doc = await get_document(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


@router.delete("/documents/{document_id}")
async def delete_document_endpoint(document_id: str):
    """Delete a specific document and all its chunks."""
    from app.services.ingestion import delete_document
    success = await delete_document(document_id)
    if not success:
        raise HTTPException(status_code=404, detail="Document not found or could not be deleted")
    return {"status": "deleted", "document_id": document_id}


@router.delete("/documents/all/clean")
async def clean_all_documents():
    """Delete ALL documents, chunks, manifests, and source files. Use with caution!"""
    from app.services.ingestion import delete_all_documents
    result = await delete_all_documents()
    return result


@router.post("/documents/upload/stream")
async def upload_document_stream(
    file: UploadFile = File(...),
    domain: str = Form(default="Data Structures and Algorithms"),
):
    """Streaming upload endpoint with SSE progress events."""
    from app.services.vector_store import index_document_chunks
    from app.core.config import settings

    BASE_DIR = Path(__file__).resolve().parents[2]
    DATA_DIR = BASE_DIR / "data"
    UPLOAD_DIR = DATA_DIR / "uploads"
    MANIFEST_DIR = DATA_DIR / "manifests"

    file_bytes = await file.read()
    original_filename = file.filename or "uploaded-document"
    safe_filename = Path(original_filename).name
    document_id = str(uuid.uuid4())
    suffix = Path(safe_filename).suffix.lower()

    _ensure_data_dirs()
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    document_dir = UPLOAD_DIR / document_id
    document_dir.mkdir(parents=True, exist_ok=True)
    source_path = document_dir / safe_filename

    async def event_generator():
        try:
            yield f"data: {json.dumps({'type': 'progress', 'progress': 5, 'stage': 'saving', 'message': 'Receiving file...'})}\n\n"
            with source_path.open("wb") as target_file:
                target_file.write(file_bytes)
            yield f"data: {json.dumps({'type': 'progress', 'progress': 10, 'stage': 'saving', 'message': 'File saved.'})}\n\n"

            yield f"data: {json.dumps({'type': 'progress', 'progress': 15, 'stage': 'parsing', 'message': 'Parsing document...'})}\n\n"
            if suffix in {".txt", ".md", ".markdown", ".rst"}:
                text = _read_text_file(file_bytes)
                parsed_text = _normalize_text(text)
            elif suffix == ".pdf":
                parsed_text = _normalize_text(_extract_pdf_text(source_path))
            elif suffix == ".docx":
                parsed_text = _normalize_text(_extract_docx_text(source_path))
            else:
                raise ValueError(f"Unsupported file type: {suffix}")
            yield f"data: {json.dumps({'type': 'progress', 'progress': 40, 'stage': 'parsing', 'message': f'Document parsed ({len(parsed_text)} chars).'})}\n\n"

            yield f"data: {json.dumps({'type': 'progress', 'progress': 45, 'stage': 'chunking', 'message': 'Splitting into chunks...'})}\n\n"
            chunks = chunk_text(parsed_text)
            yield f"data: {json.dumps({'type': 'progress', 'progress': 70, 'stage': 'chunking', 'message': f'Created {len(chunks)} chunks.'})}\n\n"

            yield f"data: {json.dumps({'type': 'progress', 'progress': 75, 'stage': 'indexing', 'message': 'Indexing in vector store...'})}\n\n"
            await index_document_chunks(
                document_id=document_id,
                filename=safe_filename,
                domain=domain or settings.default_domain,
                chunks=chunks
            )
            yield f"data: {json.dumps({'type': 'progress', 'progress': 95, 'stage': 'indexing', 'message': 'Indexing complete.'})}\n\n"

            manifest = {
                "document_id": document_id,
                "filename": safe_filename,
                "domain": domain or settings.default_domain,
                "source_type": suffix.lstrip(".") or "text",
                "source_path": str(source_path),
                "chunk_count": len(chunks),
            }
            manifest_path = MANIFEST_DIR / f"{document_id}.json"
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            yield f"data: {json.dumps({'type': 'progress', 'progress': 100, 'stage': 'complete', 'message': 'Ingestion complete!', 'document_id': document_id, 'filename': safe_filename, 'chunk_count': len(chunks)})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )
