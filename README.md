# Multi-Agent RAG Tutor

An intelligent Computer Science tutoring system powered by **6 specialized AI agents** working in concert. The system retrieves context from your uploaded textbooks, generates personalized lessons with pedagogical structure, creates practice quizzes, evaluates your answers, and tracks your learning progress across sessions.

> **Plan → Retrieve → Teach → Quiz → Evaluate → Learn**

---

## Architecture

```
Student Query
     │
     ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Planner   │────▶│  Retriever  │────▶│   Teacher   │
│   Agent     │     │   Agent     │     │   Agent     │
└─────────────┘     └─────────────┘     └─────────────┘
     │                                        │
     │                                        ▼
     │                                   ┌─────────────┐
     │                                   │   Quiz      │
     │                                   │   Agent     │
     │                                   └─────────────┘
     │                                        │
     ▼                                        ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Learner   │◀────│   Critic    │◀────│  Student    │
│   Profile   │     │   Agent     │     │  Answer     │
└─────────────┘     └─────────────┘     └─────────────┘
```

### The 6 Agents

| Agent | Role | What It Does |
|-------|------|-------------|
| **Planner** | Lesson Designer | Analyzes your query, checks your knowledge profile, and creates a personalized lesson plan with objectives, prerequisites, and difficulty level. |
| **Retriever** | Knowledge Fetcher | Performs semantic search over your uploaded documents using ChromaDB vector embeddings to find the most relevant passages. |
| **Teacher** | Explainer | Generates structured pedagogical explanations: intuition → analogy → technical details → code walkthrough → common pitfalls → practice hint. |
| **Quiz** | Practice Generator | Creates code trace questions, multiple-choice with distractors, fill-in-the-blank, and open-ended problems based on the lesson material. |
| **Critic** | Answer Checker | Evaluates your answers with constructive feedback: what was right, what was wrong, and how to improve. |
| **Learner Profile** | Student Model | Tracks your topic mastery, weak areas, strong areas, and interaction history across sessions — persisted in local storage. |

---

## Features

- **RAG-based teaching** — All explanations are grounded in your uploaded documents, not hallucinated
- **Persistent learner profiles** — Your progress survives page refreshes and browser restarts via `localStorage` + server-side JSON persistence
- **Document management** — Upload PDFs, DOCX, TXT, or Markdown; view, delete, or clean all documents
- **Streaming ingestion** — Watch real-time progress as your document is parsed, chunked, and indexed
- **Markdown rendering** — Code blocks, bold text, lists, and headers render beautifully in the UI
- **Multi-provider LLM support** — Groq (default), OpenAI, local GGUF, or Ollama
- **Agent execution pipeline** — Toggle a live log to see which agent is running and what it's doing

---

## Tech Stack

### Backend
- **FastAPI** — High-performance async API framework
- **ChromaDB** — Persistent vector database for semantic search
- **LangChain** — LLM orchestration and message passing
- **Groq** — Default LLM provider (fast, free tier available)
- **PyMuPDF** — PDF text extraction with heading detection
- **python-docx** — DOCX parsing

### Frontend
- **Next.js 14** — React framework with App Router
- **TypeScript** — Type-safe development
- **Tailwind CSS** — Utility-first styling (via inline styles in this version)

### Infrastructure
- **LocalStorage** — Session ID persistence across page reloads
- **JSON files** — Learner profiles and document manifests stored on disk

---

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- A Groq API key (free at [console.groq.com](https://console.groq.com))

### 1. Clone & Install Backend

```bash
cd backend

# Create virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

# Install dependencies
pip install -e .
```

### 2. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` and add your Groq API key:

```env
GROQ_API_KEY=your_groq_api_key_here
```

### 3. Start Backend

```bash
python -m uvicorn app.main:app --reload --port 8000
```

### 4. Install & Start Frontend

```bash
cd ../frontend
npm install
npm run dev
```

### 5. Open Browser

Navigate to [http://localhost:3000](http://localhost:3000)

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GROQ_API_KEY` | ✅ Yes | — | Your Groq API key (fastest, recommended) |
| `LLM_PROVIDER` | ❌ No | `groq` | Choose: `groq`, `openai`, `gguf`, `ollama` |
| `GROQ_MODEL` | ❌ No | `llama-3.1-8b-instant` | Model to use on Groq |
| `OPENAI_API_KEY` | ❌ No | — | OpenAI API key (optional alternative) |
| `OPENAI_MODEL` | ❌ No | `gpt-4o-mini` | OpenAI model to use |
| `LLM_MODEL` | ❌ No | `gemma-4-E2B-it-Q4_K_M.gguf` | Path to local GGUF file |
| `OLLAMA_BASE_URL` | ❌ No | `http://localhost:11434` | Ollama server URL |

---

## Project Structure

```
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   └── routes.py          # FastAPI endpoints
│   │   ├── core/
│   │   │   └── config.py          # Settings & env vars
│   │   ├── services/
│   │   │   ├── agents/            # 6 tutoring agents
│   │   │   │   ├── planner_agent.py
│   │   │   │   ├── retrieval_agent.py
│   │   │   │   ├── teacher_agent.py
│   │   │   │   ├── quiz_agent.py
│   │   │   │   ├── critic_agent.py
│   │   │   │   └── orchestrator.py
│   │   │   ├── ingestion.py       # Document parsing & chunking
│   │   │   ├── vector_store.py    # ChromaDB integration
│   │   │   ├── llm_provider.py    # Unified LLM interface
│   │   │   └── learner_profile.py # Student progress tracking
│   │   └── main.py                # App entry point
│   ├── data/                      # Runtime data (gitignored)
│   │   ├── chroma/               # Vector database
│   │   ├── manifests/            # Document metadata
│   │   ├── uploads/              # Source files
│   │   └── profiles/             # Learner profiles
│   └── pyproject.toml             # Python dependencies
│
├── frontend/
│   ├── app/
│   │   └── page.tsx               # Main UI
│   ├── lib/
│   │   └── api.ts                 # API client
│   ├── public/
│   └── package.json
│
├── .env.example                   # Template for environment vars
├── .gitignore                     # Excludes data, models, secrets
└── README.md                      # This file
```

---

## Usage Guide

### Upload a Textbook
1. Drag or select a PDF, DOCX, TXT, or Markdown file
2. Click **Upload & Ingest**
3. Watch the progress bar: Saving → Parsing → Chunking → Indexing

### Ask a Question
1. Type a question like: *"Teach me stacks and queues with examples"*
2. Click **Teach Me**
3. The pipeline runs: Planner → Retriever → Teacher → Quiz

### Take a Quiz
1. Read the practice question (may include code snippets)
2. Select an option or type your answer
3. Click **Check Answer** for feedback from the Critic agent

### Track Progress
- Your **Learner Profile** card shows:
  - Current lesson
  - Weak areas (topics with low accuracy)
  - Strong areas (topics you've mastered)
  - Topic progress bars with accuracy scores
- Click **New Session** to start fresh

### Manage Documents
- View all uploaded documents in **Your Library**
- Delete individual documents
- Click **Clean All** to wipe everything and start over

---

## LLM Provider Options

The system supports four LLM backends. Set `LLM_PROVIDER` in your `.env`:

| Provider | Speed | Cost | Setup |
|----------|-------|------|-------|
| **Groq** | ⚡ Fastest | Free tier | Just add `GROQ_API_KEY` |
| **OpenAI** | Fast | Pay-per-use | Add `OPENAI_API_KEY` |
| **GGUF** | Slow (first load) | Free | Place `.gguf` file, set `LLM_MODEL` path |
| **Ollama** | Medium | Free | Install Ollama, run a model locally |

> **Recommendation:** Start with Groq. It's free, fast (~1-2s per response), and requires zero local setup.

---

## Roadmap

- [ ] Docker Compose setup for one-command deployment
- [ ] User authentication & multi-user support
- [ ] Export learner profile as PDF progress report
- [ ] Voice input / speech-to-text for questions
- [ ] Mobile-responsive UI
- [ ] Integration with external LMS (Canvas, Moodle)
- [ ] Support for more file types (EPUB, HTML, PPTX)
- [ ] Collaborative learning (study groups)

---

## License

MIT License — feel free to use, modify, and distribute.

---

## Acknowledgments

- Built with [FastAPI](https://fastapi.tiangolo.com/), [ChromaDB](https://www.trychroma.com/), and [LangChain](https://www.langchain.com/)
- LLM inference powered by [Groq](https://groq.com/)
- Frontend framework: [Next.js](https://nextjs.org/)

---

<p align="center">
  <strong>Happy Learning!</strong>
</p>
