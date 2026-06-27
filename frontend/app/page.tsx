"use client";

import { FormEvent, useEffect, useRef, useState } from "react";
import {
  TeachResponse,
  EvaluateResponse,
  AgentDetails,
  UploadProgressEvent,
  DocumentInfo,
  LearnerProfile,
  fetchAgentDetails,
  teach,
  evaluate,
  uploadDocumentStream,
  listDocuments,
  deleteDocument,
  cleanAllDocuments,
  getProfile,
} from "@/lib/api";

type Message = {
  role: "assistant" | "user";
  content: string;
};

const agentColor: Record<string, string> = {
  planner: "#3b82f6",
  retriever: "#8b5cf6",
  teacher: "#10b981",
  quiz: "#f59e0b",
  critic: "#ef4444",
  learner_profile: "#6366f1",
};

const agentLabel: Record<string, string> = {
  planner: "Planner",
  retriever: "Retriever",
  teacher: "Teacher",
  quiz: "Quiz",
  critic: "Critic",
  learner_profile: "Profile",
};

const defaultAgents: AgentDetails[] = [
  { id: "planner", title: "Planner Agent", subtitle: "Lesson Planner", color: "blue", icon: "fa-map", description: "Analyzes the student's query and knowledge profile, then creates a personalized lesson plan.", responsibilities: ["Analyzes query intent", "Checks learner profile", "Creates lesson plans", "Identifies prerequisites"], tools: ["Learner Profile DB", "Query Analyzer"] },
  { id: "retriever", title: "Retriever Agent", subtitle: "Knowledge Fetcher", color: "purple", icon: "fa-database", description: "Performs semantic search to find relevant material for the lesson.", responsibilities: ["Converts query to embedding", "ANN search", "Retrieves top-k chunks"], tools: ["Vector DB", "Embedding Model"] },
  { id: "teacher", title: "Teacher Agent", subtitle: "Explainer", color: "emerald", icon: "fa-chalkboard-user", description: "Generates structured pedagogical explanations with analogies and code examples.", responsibilities: ["Starts with intuition", "Uses analogies", "Code walkthroughs", "Anticipates misconceptions"], tools: ["LLM (Groq)", "Prompt Templates"] },
  { id: "quiz", title: "Quiz Agent", subtitle: "Practice Generator", color: "amber", icon: "fa-clipboard-question", description: "Creates practice problems that test true understanding.", responsibilities: ["Code trace questions", "Multiple choice", "Fill-in-the-blank", "Provides hints"], tools: ["Question Generator", "Distractor Builder"] },
  { id: "critic", title: "Critic Agent", subtitle: "Answer Checker", color: "red", icon: "fa-scale-balanced", description: "Evaluates student answers with constructive, encouraging feedback.", responsibilities: ["Scores answers", "Identifies right/wrong", "Suggests improvement", "Recommends retry"], tools: ["Answer Scorer", "Feedback Generator"] },
  { id: "learner_profile", title: "Learner Profile", subtitle: "Student Model", color: "indigo", icon: "fa-user-graduate", description: "Tracks knowledge state, progress, weak areas, and strong areas.", responsibilities: ["Records interactions", "Tracks mastery", "Identifies weak areas", "Determines next topic"], tools: ["Profile Database", "Mastery Tracker"] },
];

const quickPrompts = [
  "Teach me stacks and queues with examples.",
  "What is a binary search tree and how do I use it?",
  "Quiz me on recursion.",
  "Explain time complexity like I am new.",
];

function renderMarkdown(text: string): string {
  if (!text) return "";
  let html = text
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/```(\w*)\n([\s\S]*?)```/g, '<pre style="background:#0f172a;border:1px solid rgba(255,255,255,0.1);border-radius:8px;padding:12px;margin:8px 0;overflow-x:auto;font-family:JetBrains Mono,monospace;font-size:12px;line-height:1.5;color:#e2e8f0;"><code>$2</code></pre>')
    .replace(/`([^`]+)`/g, '<code style="background:rgba(255,255,255,0.08);padding:2px 6px;border-radius:4px;font-family:JetBrains Mono,monospace;font-size:12px;color:#fbbf24;">$1</code>')
    .replace(/\*\*([^*]+)\*\*/g, '<strong style="color:#e2e8f0;">$1</strong>')
    .replace(/\*([^*]+)\*/g, '<em style="color:#a78bfa;">$1</em>')
    .replace(/^#{1,2}\s+(.+)$/gm, '<h3 style="font-size:14px;font-weight:700;color:#60a5fa;margin:12px 0 6px;">$1</h3>')
    .replace(/^#{3,6}\s+(.+)$/gm, '<h4 style="font-size:13px;font-weight:600;color:#60a5fa;margin:10px 0 4px;">$1</h4>')
    .replace(/^>\s+(.+)$/gm, '<blockquote style="border-left:3px solid #3b82f6;padding-left:12px;margin:8px 0;color:#94a3b8;font-style:italic;">$1</blockquote>')
    .replace(/^-{3,}$/gm, '<hr style="border:0;border-top:1px solid rgba(255,255,255,0.1);margin:12px 0;" />')
    .replace(/^\s*[-*]\s+(.+)$/gm, '<li style="margin:4px 0;color:#cbd5e1;">$1</li>')
    .replace(/^\s*\d+\.\s+(.+)$/gm, '<li style="margin:4px 0;color:#cbd5e1;">$1</li>')
    .replace(/\n\n/g, '</p><p style="margin:8px 0;color:#cbd5e1;line-height:1.6;">')
    .replace(/\n/g, '<br />');
  if (!html.startsWith('<')) html = '<p style="margin:8px 0;color:#cbd5e1;line-height:1.6;">' + html + '</p>';
  return html;
}

export default function HomePage() {
  const [messages, setMessages] = useState<Message[]>([
    { role: "assistant", content: "Hi! I am your CS tutor. Ask me anything and I will plan a lesson, teach you with analogies, and give you a practice quiz!" },
  ]);
  const [input, setInput] = useState("");
  const [sessionId] = useState(() => {
    // Try to reuse existing session ID from localStorage
    const saved = typeof window !== "undefined" ? localStorage.getItem("tutor_session_id") : null;
    if (saved) return saved;
    const fresh = crypto.randomUUID();
    if (typeof window !== "undefined") localStorage.setItem("tutor_session_id", fresh);
    return fresh;
  });
  const [domain, setDomain] = useState("Data Structures and Algorithms");
  const [file, setFile] = useState<File | null>(null);

  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadStage, setUploadStage] = useState("");
  const [uploadMsg, setUploadMsg] = useState("");
  const [uploading, setUploading] = useState(false);
  const [uploaded, setUploaded] = useState<{ filename: string; chunk_count: number } | null>(null);
  const [uploadError, setUploadError] = useState("");
  const [documents, setDocuments] = useState<DocumentInfo[]>([]);
  const [docsLoading, setDocsLoading] = useState(false);
  const [docsError, setDocsError] = useState("");
  const [cleaning, setCleaning] = useState(false);

  const [teachLoading, setTeachLoading] = useState(false);
  const [teachError, setTeachError] = useState("");

  const [lastLesson, setLastLesson] = useState<TeachResponse | null>(null);
  const [quizAnswer, setQuizAnswer] = useState("");
  const [evaluation, setEvaluation] = useState<EvaluateResponse | null>(null);
  const [evaluating, setEvaluating] = useState(false);

  const [profile, setProfile] = useState<LearnerProfile | null>(null);
  const [profileLoading, setProfileLoading] = useState(false);
  const [showProfile, setShowProfile] = useState(true);

  const [showPipeline, setShowPipeline] = useState(false);
  const [logs, setLogs] = useState<{ time: string; agent: string | null; text: string }[]>([]);
  const [activeAgent, setActiveAgent] = useState<string | null>(null);
  const [agents, setAgents] = useState<AgentDetails[]>(defaultAgents);
  const [modalAgent, setModalAgent] = useState<AgentDetails | null>(null);
  const [showModal, setShowModal] = useState(false);

  const chatEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => { fetchAgentDetails().then(setAgents).catch(() => {}); }, []);
  useEffect(() => { chatEndRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages, lastLesson, evaluation]);

  const fetchDocs = async () => {
    setDocsLoading(true);
    setDocsError("");
    try {
      const result = await listDocuments();
      setDocuments(result.documents);
    } catch (e) {
      setDocsError(e instanceof Error ? e.message : "Failed to load documents");
    } finally {
      setDocsLoading(false);
    }
  };

  const fetchProfile = async () => {
    setProfileLoading(true);
    try {
      const result = await getProfile(sessionId);
      setProfile(result);
    } catch (e) {
      // silently fail - profile is optional
    } finally {
      setProfileLoading(false);
    }
  };

  useEffect(() => { fetchDocs(); }, []);
  useEffect(() => { fetchProfile(); }, []);

  const addLog = (text: string, agent: string | null = null) => {
    const time = new Date().toLocaleTimeString("en-US", { hour12: false, hour: "2-digit", minute: "2-digit", second: "2-digit" });
    setLogs((prev) => [...prev, { time, agent, text }]);
    if (agent) setActiveAgent(agent);
  };

  const handleUpload = async () => {
    if (!file) { setUploadError("Choose a file first."); return; }
    setUploading(true); setUploadProgress(0); setUploadStage(""); setUploadMsg(""); setUploadError(""); setUploaded(null);
    try {
      for await (const event of uploadDocumentStream(file, domain)) {
        setUploadProgress(event.progress); setUploadStage(event.stage); setUploadMsg(event.message);
        if (event.progress === 100 && event.document_id) {
          setUploaded({ filename: event.filename || file.name, chunk_count: event.chunk_count || 0 });
          fetchDocs();
        }
      }
    } catch (e) { setUploadError(e instanceof Error ? e.message : "Upload failed"); }
    finally { setUploading(false); }
  };

  const handleDeleteDoc = async (docId: string) => {
    try {
      await deleteDocument(docId);
      fetchDocs();
    } catch (e) {
      setDocsError(e instanceof Error ? e.message : "Delete failed");
    }
  };

  const handleCleanAll = async () => {
    if (!window.confirm("This will delete ALL uploaded documents, chunks, and data. This cannot be undone. Continue?")) return;
    setCleaning(true);
    try {
      const result = await cleanAllDocuments();
      alert(result.message);
      fetchDocs();
    } catch (e) {
      setDocsError(e instanceof Error ? e.message : "Clean failed");
    } finally {
      setCleaning(false);
    }
  };

  const handleTeach = async (event?: FormEvent) => {
    event?.preventDefault();
    const text = input.trim();
    if (!text) return;
    setMessages((p) => [...p, { role: "user", content: text }]);
    setInput("");
    setTeachLoading(true);
    setTeachError("");
    setShowPipeline(true);
    setLogs([]);
    setLastLesson(null);
    setEvaluation(null);
    setQuizAnswer("");
    addLog("Received student query", "planner");

    try {
      const response = await teach(text, sessionId, domain);
      setLastLesson(response);
      response.agent_trace?.forEach((trace) => {
        const agent = trace.match(/\[([a-z_]+)\]/)?.[1] || null;
        addLog(trace, agent);
      });
      setMessages((p) => [...p, { role: "assistant", content: response.explanation }]);
      fetchProfile(); // refresh profile after lesson
    } catch (e) {
      setTeachError(e instanceof Error ? e.message : "Teaching failed");
    } finally {
      setTeachLoading(false);
    }
  };

  const handleQuizSubmit = async () => {
    if (!lastLesson?.quiz_question || !quizAnswer.trim()) return;
    setEvaluating(true);
    try {
      const result = await evaluate({
        question: lastLesson.quiz_question.question,
        student_answer: quizAnswer.trim(),
        correct_answer: lastLesson.quiz_question.correct_answer,
        domain,
        session_id: sessionId,
      });
      setEvaluation(result);
      fetchProfile(); // refresh profile after quiz evaluation
    } catch (e) {
      setTeachError(e instanceof Error ? e.message : "Evaluation failed");
    } finally {
      setEvaluating(false);
    }
  };

  const openModal = (id: string) => {
    const a = agents.find((x) => x.id === id);
    if (a) { setModalAgent(a); setShowModal(true); }
  };

  const stageLabels: Record<string, string> = {
    saving: "Receiving", parsing: "Parsing", chunking: "Chunking", indexing: "Indexing", complete: "Complete",
  };

  return (
    <div style={{ background: "#0b1120", color: "#e2e8f0", minHeight: "100vh", fontFamily: "Inter, sans-serif" }}>
      {/* Header */}
      <header style={{ position: "fixed", top: 0, left: 0, right: 0, zIndex: 50, background: "rgba(11,17,32,0.85)", backdropFilter: "blur(12px)", borderBottom: "1px solid rgba(255,255,255,0.06)", height: 56, display: "flex", alignItems: "center", padding: "0 24px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12, maxWidth: 1200, width: "100%", margin: "0 auto" }}>
          <div style={{ width: 34, height: 34, borderRadius: 10, background: "linear-gradient(135deg,#3b82f6,#8b5cf6)", display: "flex", alignItems: "center", justifyContent: "center" }}>
            <i className="fas fa-network-wired" style={{ color: "#fff", fontSize: 14 }} />
          </div>
          <div>
            <div style={{ fontWeight: 700, fontSize: 15, background: "linear-gradient(90deg,#60a5fa,#a78bfa)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}>Multi-Agent Tutor</div>
            <div style={{ fontSize: 10, color: "#64748b" }}>Plan → Teach → Quiz → Learn</div>
          </div>
          <div style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 12 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 11, color: "#64748b" }}>
              <span style={{ width: 6, height: 6, borderRadius: "50%", background: "#34d399", display: "inline-block" }} />
              System Active
            </div>
          </div>
        </div>
      </header>

      {/* Main */}
      <main style={{ maxWidth: 1200, margin: "0 auto", padding: "80px 24px 40px" }}>
        {/* Welcome + Query */}
        {messages.length <= 1 && (
          <section style={{ textAlign: "center", padding: "40px 0 32px" }}>
            <h1 style={{ fontSize: "clamp(2rem,5vw,3.2rem)", fontWeight: 800, lineHeight: 1.1, background: "linear-gradient(90deg,#60a5fa,#a78bfa,#34d399)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent", marginBottom: 12 }}>
              Your Personal CS Tutor
            </h1>
            <p style={{ color: "#94a3b8", fontSize: 15, maxWidth: 560, margin: "0 auto 28px" }}>
              I plan lessons, teach with analogies, and quiz you to check understanding. Upload a textbook and let us start learning!
            </p>
            <div style={{ maxWidth: 640, margin: "0 auto", position: "relative" }}>
              <div style={{ position: "absolute", inset: -1, borderRadius: 16, background: "linear-gradient(135deg,#3b82f6,#8b5cf6)", opacity: 0.15, filter: "blur(8px)" }} />
              <div style={{ position: "relative", background: "rgba(30,41,59,0.7)", backdropFilter: "blur(12px)", border: "1px solid rgba(255,255,255,0.08)", borderRadius: 16, padding: 6, display: "flex", alignItems: "center", gap: 8 }}>
                <i className="fas fa-search" style={{ color: "#475569", marginLeft: 12, fontSize: 14 }} />
                <input type="text" value={input} onChange={(e) => setInput(e.target.value)} onKeyDown={(e) => e.key === "Enter" && handleTeach()} placeholder="Ask anything about your uploaded material..." style={{ flex: 1, background: "transparent", border: "none", outline: "none", color: "#e2e8f0", fontSize: 14, padding: "10px 4px" }} />
                <button onClick={() => handleTeach()} style={{ padding: "10px 20px", borderRadius: 12, border: "none", background: "linear-gradient(135deg,#3b82f6,#8b5cf6)", color: "#fff", fontWeight: 600, fontSize: 13, cursor: "pointer" }}>
                  <i className="fas fa-play" style={{ marginRight: 6 }} />Teach Me
                </button>
              </div>
            </div>
            <div style={{ display: "flex", flexWrap: "wrap", justifyContent: "center", gap: 8, marginTop: 16 }}>
              {quickPrompts.map((q) => (
                <button key={q} onClick={() => setInput(q)} style={{ padding: "8px 14px", borderRadius: 999, fontSize: 12, color: "#94a3b8", background: "rgba(255,255,255,0.04)", border: "1px solid rgba(255,255,255,0.08)", cursor: "pointer" }}>{q}</button>
              ))}
            </div>
          </section>
        )}

        {/* Workspace */}
        <div style={{ display: "grid", gridTemplateColumns: "320px 1fr", gap: 20 }}>
          {/* Left Sidebar */}
          <aside style={{ display: "flex", flexDirection: "column", gap: 16 }}>
            {/* Upload */}
            <div style={{ background: "rgba(30,41,59,0.6)", backdropFilter: "blur(12px)", border: "1px solid rgba(255,255,255,0.08)", borderRadius: 20, padding: 20 }}>
              <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 16 }}>
                <i className="fas fa-upload" style={{ color: "#a78bfa" }} />
                <h2 style={{ fontSize: 14, fontWeight: 600, margin: 0 }}>Upload Source</h2>
              </div>
              <label style={{ display: "block", fontSize: 12, color: "#94a3b8", marginBottom: 6 }}>Domain</label>
              <input value={domain} onChange={(e) => setDomain(e.target.value)} style={{ width: "100%", padding: "10px 12px", borderRadius: 12, border: "1px solid rgba(255,255,255,0.1)", background: "rgba(255,255,255,0.03)", color: "#e2e8f0", fontSize: 13, marginBottom: 12 }} />
              <label style={{ display: "block", fontSize: 12, color: "#94a3b8", marginBottom: 6 }}>File</label>
              <input type="file" accept=".pdf,.docx,.txt,.md,.markdown,.rst" onChange={(e) => { setFile(e.target.files?.[0] ?? null); setUploadError(""); }} style={{ width: "100%", padding: "10px 12px", borderRadius: 12, border: "1px solid rgba(255,255,255,0.1)", background: "rgba(255,255,255,0.03)", color: "#e2e8f0", fontSize: 13, marginBottom: 12 }} />
              <button onClick={handleUpload} disabled={uploading} style={{ width: "100%", padding: "12px", borderRadius: 12, border: "none", background: "linear-gradient(135deg,#3b82f6,#8b5cf6)", color: "#fff", fontWeight: 600, fontSize: 13, cursor: uploading ? "wait" : "pointer", opacity: uploading ? 0.7 : 1 }}>
                {uploading ? <span><i className="fas fa-spinner fa-spin" style={{ marginRight: 6 }} /> {stageLabels[uploadStage] || "Processing"}...</span> : "Upload & Ingest"}
              </button>
              {uploading && (
                <div style={{ marginTop: 14 }}>
                  <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, color: "#94a3b8", marginBottom: 4 }}>
                    <span>{stageLabels[uploadStage] || "Processing"}</span><span>{uploadProgress}%</span>
                  </div>
                  <div style={{ height: 6, borderRadius: 3, background: "rgba(255,255,255,0.06)", overflow: "hidden" }}>
                    <div style={{ height: "100%", width: `${uploadProgress}%`, borderRadius: 3, background: "linear-gradient(90deg,#3b82f6,#8b5cf6)", transition: "width 0.3s ease" }} />
                  </div>
                  <p style={{ fontSize: 11, color: "#64748b", marginTop: 4 }}>{uploadMsg}</p>
                </div>
              )}
              {uploaded && <div style={{ marginTop: 12, padding: 12, borderRadius: 12, background: "rgba(16,185,129,0.08)", border: "1px solid rgba(16,185,129,0.2)", color: "#d1fae5", fontSize: 12 }}><strong>{uploaded.filename}</strong> ingested into <strong>{uploaded.chunk_count}</strong> chunks.</div>}
              {uploadError && <div style={{ marginTop: 12, padding: 12, borderRadius: 12, background: "rgba(239,68,68,0.08)", border: "1px solid rgba(239,68,68,0.2)", color: "#fee2e2", fontSize: 12 }}>{uploadError}</div>}
            </div>

            {/* Document Library */}
            <div style={{ background: "rgba(30,41,59,0.6)", backdropFilter: "blur(12px)", border: "1px solid rgba(255,255,255,0.08)", borderRadius: 20, padding: 20 }}>
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 14 }}>
                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  <i className="fas fa-book" style={{ color: "#fbbf24" }} />
                  <h2 style={{ fontSize: 14, fontWeight: 600, margin: 0 }}>Your Library</h2>
                </div>
                <button onClick={fetchDocs} style={{ padding: "4px 10px", borderRadius: 8, border: "1px solid rgba(255,255,255,0.1)", background: "rgba(255,255,255,0.04)", color: "#94a3b8", fontSize: 11, cursor: "pointer" }}><i className="fas fa-rotate-right" /></button>
              </div>
              {docsLoading && <div style={{ color: "#64748b", fontSize: 12, textAlign: "center", padding: "12px 0" }}><i className="fas fa-spinner fa-spin" /> Loading...</div>}
              {docsError && <div style={{ color: "#f87171", fontSize: 11, padding: 8 }}>{docsError}</div>}
              {documents.length === 0 && !docsLoading ? (
                <div style={{ color: "#475569", fontSize: 12, textAlign: "center", padding: "12px 0" }}>No documents yet. Upload one above!</div>
              ) : (
                <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                  {documents.map((doc) => (
                    <div key={doc.document_id} style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "8px 10px", borderRadius: 10, background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.06)" }}>
                      <div style={{ overflow: "hidden" }}>
                        <div style={{ fontSize: 12, fontWeight: 500, color: "#e2e8f0", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{doc.filename}</div>
                        <div style={{ fontSize: 10, color: "#64748b" }}>{doc.chunk_count} chunks · {doc.domain}</div>
                      </div>
                      <button onClick={() => handleDeleteDoc(doc.document_id)} style={{ padding: "4px 8px", borderRadius: 6, border: "none", background: "rgba(239,68,68,0.1)", color: "#f87171", fontSize: 10, cursor: "pointer" }}><i className="fas fa-trash" /></button>
                    </div>
                  ))}
                </div>
              )}
              {documents.length > 0 && (
                <button onClick={handleCleanAll} disabled={cleaning} style={{ width: "100%", marginTop: 10, padding: "8px", borderRadius: 10, border: "1px solid rgba(239,68,68,0.2)", background: "rgba(239,68,68,0.05)", color: "#f87171", fontSize: 11, cursor: cleaning ? "wait" : "pointer" }}>
                  {cleaning ? <><i className="fas fa-spinner fa-spin" /> Cleaning...</> : <><i className="fas fa-broom" /> Clean All Documents</>}
                </button>
              )}
            </div>

            {/* Your Profile */}
            {profile && (
              <div style={{ background: "rgba(30,41,59,0.6)", backdropFilter: "blur(12px)", border: "1px solid rgba(255,255,255,0.08)", borderRadius: 20, padding: 20 }}>
                <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 14 }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                    <i className="fas fa-user-graduate" style={{ color: "#6366f1" }} />
                    <h2 style={{ fontSize: 14, fontWeight: 600, margin: 0 }}>Your Profile</h2>
                  </div>
                  <button onClick={fetchProfile} style={{ padding: "4px 10px", borderRadius: 8, border: "1px solid rgba(255,255,255,0.1)", background: "rgba(255,255,255,0.04)", color: "#94a3b8", fontSize: 11, cursor: "pointer" }}><i className="fas fa-rotate-right" /></button>
                </div>
                {profileLoading && <div style={{ color: "#64748b", fontSize: 12, textAlign: "center", padding: "8px 0" }}><i className="fas fa-spinner fa-spin" /> Loading...</div>}
                {!profileLoading && (
                  <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                    {profile.current_lesson && (
                      <div style={{ padding: "8px 10px", borderRadius: 8, background: "rgba(99,102,241,0.08)", border: "1px solid rgba(99,102,241,0.15)" }}>
                        <div style={{ fontSize: 10, color: "#64748b", marginBottom: 2 }}>Current Lesson</div>
                        <div style={{ fontSize: 12, fontWeight: 500, color: "#e2e8f0" }}>{profile.current_lesson}</div>
                      </div>
                    )}
                    {profile.weak_areas.length > 0 && (
                      <div>
                        <div style={{ fontSize: 10, color: "#64748b", marginBottom: 4 }}><i className="fas fa-triangle-exclamation" style={{ marginRight: 4, color: "#f87171" }} />Weak Areas</div>
                        <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
                          {profile.weak_areas.map((area) => (
                            <span key={area} style={{ fontSize: 10, padding: "3px 8px", borderRadius: 999, background: "rgba(239,68,68,0.1)", color: "#f87171", border: "1px solid rgba(239,68,68,0.2)" }}>{area}</span>
                          ))}
                        </div>
                      </div>
                    )}
                    {profile.strong_areas.length > 0 && (
                      <div>
                        <div style={{ fontSize: 10, color: "#64748b", marginBottom: 4 }}><i className="fas fa-check-circle" style={{ marginRight: 4, color: "#34d399" }} />Strong Areas</div>
                        <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
                          {profile.strong_areas.map((area) => (
                            <span key={area} style={{ fontSize: 10, padding: "3px 8px", borderRadius: 999, background: "rgba(16,185,129,0.1)", color: "#34d399", border: "1px solid rgba(16,185,129,0.2)" }}>{area}</span>
                          ))}
                        </div>
                      </div>
                    )}
                    {Object.keys(profile.topics).length > 0 && (
                      <div>
                        <div style={{ fontSize: 10, color: "#64748b", marginBottom: 4 }}>Topic Progress</div>
                        <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                          {Object.values(profile.topics).slice(0, 5).map((topic) => (
                            <div key={topic.topic}>
                              <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10, color: "#94a3b8", marginBottom: 2 }}>
                                <span>{topic.topic}</span>
                                <span>{Math.round(topic.accuracy * 100)}% ({topic.questions_correct}/{topic.questions_attempted})</span>
                              </div>
                              <div style={{ height: 4, borderRadius: 2, background: "rgba(255,255,255,0.06)", overflow: "hidden" }}>
                                <div style={{ height: "100%", width: `${topic.accuracy * 100}%`, borderRadius: 2, background: topic.accuracy > 0.7 ? "#34d399" : topic.accuracy > 0.4 ? "#fbbf24" : "#f87171", transition: "width 0.3s ease" }} />
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                    {profile.interactions.length > 0 && (
                      <div style={{ fontSize: 10, color: "#475569", textAlign: "right" }}>{profile.interactions.length} interaction{profile.interactions.length !== 1 ? "s" : ""}</div>
                    )}
                    <button onClick={() => { if (window.confirm("Start a new session? Your current progress will be saved but a new profile will be created.")) { localStorage.removeItem("tutor_session_id"); window.location.reload(); } }} style={{ width: "100%", marginTop: 8, padding: "6px", borderRadius: 8, border: "1px solid rgba(255,255,255,0.1)", background: "rgba(255,255,255,0.03)", color: "#64748b", fontSize: 10, cursor: "pointer" }}>
                      <i className="fas fa-rotate" style={{ marginRight: 4 }} />New Session
                    </button>
                    {profile.interactions.length === 0 && !profile.current_lesson && (
                      <div style={{ color: "#475569", fontSize: 12, textAlign: "center", padding: "8px 0" }}>No activity yet. Start learning!</div>
                    )}
                  </div>
                )}
              </div>
            )}

            {/* Agent Network */}
            <div style={{ background: "rgba(30,41,59,0.6)", backdropFilter: "blur(12px)", border: "1px solid rgba(255,255,255,0.08)", borderRadius: 20, padding: 20 }}>
              <h2 style={{ fontSize: 14, fontWeight: 600, margin: "0 0 14px" }}>Agent Network</h2>
              <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                {["planner", "retriever", "teacher", "quiz", "critic", "learner_profile"].map((id) => {
                  const color = agentColor[id];
                  const isActive = activeAgent === id;
                  return (
                    <div key={id} style={{ display: "flex", alignItems: "center", gap: 10, padding: "8px 10px", borderRadius: 10, cursor: "pointer", background: isActive ? `${color}15` : "transparent", border: isActive ? `1px solid ${color}40` : "1px solid transparent", transition: "all 0.2s" }} onClick={() => openModal(id)}>
                      <div style={{ width: 32, height: 32, borderRadius: 8, background: `linear-gradient(135deg, ${color}, ${color}cc)`, display: "flex", alignItems: "center", justifyContent: "center" }}>
                        <i className={`fas ${id === "planner" ? "fa-map" : id === "retriever" ? "fa-database" : id === "teacher" ? "fa-chalkboard-user" : id === "quiz" ? "fa-clipboard-question" : id === "critic" ? "fa-scale-balanced" : "fa-user-graduate"}`} style={{ color: "#fff", fontSize: 12 }} />
                      </div>
                      <div>
                        <div style={{ fontSize: 12, fontWeight: 600, color: isActive ? color : "#e2e8f0" }}>{agentLabel[id]}</div>
                        <div style={{ fontSize: 10, color: "#64748b" }}>{id === "planner" ? "Plans lessons" : id === "retriever" ? "Finds material" : id === "teacher" ? "Explains concepts" : id === "quiz" ? "Creates problems" : id === "critic" ? "Checks answers" : "Tracks progress"}</div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </aside>

          {/* Right: Lesson + Chat + Quiz */}
          <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
            {/* Pipeline Toggle */}
            <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
              <button onClick={() => setShowPipeline(!showPipeline)} style={{ padding: "8px 16px", borderRadius: 10, border: "1px solid rgba(255,255,255,0.1)", background: showPipeline ? "rgba(59,130,246,0.15)" : "rgba(255,255,255,0.04)", color: showPipeline ? "#60a5fa" : "#94a3b8", fontSize: 12, cursor: "pointer" }}>
                <i className={`fas ${showPipeline ? "fa-eye" : "fa-eye-slash"}`} style={{ marginRight: 6 }} />{showPipeline ? "Hide Pipeline" : "Show Pipeline"}
              </button>
              {teachLoading && <div style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 12, color: "#60a5fa" }}><i className="fas fa-spinner fa-spin" /> Planning lesson...</div>}
            </div>

            {/* Execution Logs */}
            {showPipeline && (
              <div style={{ background: "rgba(30,41,59,0.6)", backdropFilter: "blur(12px)", border: "1px solid rgba(255,255,255,0.08)", borderRadius: 20, padding: 20, maxHeight: 240, overflowY: "auto" }}>
                <h3 style={{ fontSize: 13, fontWeight: 600, marginBottom: 10 }}><i className="fas fa-terminal" style={{ marginRight: 6, color: "#10b981" }} />Execution Log</h3>
                {logs.length === 0 ? <div style={{ color: "#475569", fontSize: 12, textAlign: "center", padding: "20px 0" }}>Waiting for query...</div> : (
                  <div style={{ display: "flex", flexDirection: "column", gap: 6, fontFamily: "'JetBrains Mono', monospace", fontSize: 11 }}>
                    {logs.map((log, i) => (
                      <div key={i} style={{ display: "flex", gap: 8, padding: "4px 0", borderBottom: i < logs.length - 1 ? "1px solid rgba(255,255,255,0.04)" : "none" }}>
                        <span style={{ color: "#475569", whiteSpace: "nowrap" }}>[{log.time}]</span>
                        <span style={{ color: log.agent ? agentColor[log.agent] || "#64748b" : "#64748b", fontWeight: 600, minWidth: 90, fontSize: 10 }}>{log.agent ? `[${log.agent}]` : ""}</span>
                        <span style={{ color: "#cbd5e1" }}>{log.text}</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}

            {/* Chat / Lesson Display */}
            <div style={{ background: "rgba(30,41,59,0.6)", backdropFilter: "blur(12px)", border: "1px solid rgba(255,255,255,0.08)", borderRadius: 20, padding: 20, display: "flex", flexDirection: "column", minHeight: 400 }}>
              <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 16 }}>
                <i className="fas fa-graduation-cap" style={{ color: "#60a5fa" }} />
                <h2 style={{ fontSize: 14, fontWeight: 600, margin: 0 }}>Your Lesson</h2>
              </div>

              <div style={{ flex: 1, overflowY: "auto", display: "flex", flexDirection: "column", gap: 12, marginBottom: 16 }}>
                {messages.map((msg, i) => (
                  <div key={i} style={{ alignSelf: msg.role === "user" ? "flex-end" : "flex-start", maxWidth: "85%", padding: "12px 16px", borderRadius: 16, background: msg.role === "user" ? "rgba(59,130,246,0.15)" : "rgba(255,255,255,0.04)", border: "1px solid rgba(255,255,255,0.06)" }}>
                    <div style={{ fontSize: 10, textTransform: "uppercase", letterSpacing: "0.1em", color: "#64748b", marginBottom: 4 }}>{msg.role}</div>
                    <div className="md-answer" style={{ fontSize: 13, color: "#e2e8f0", lineHeight: 1.6 }} dangerouslySetInnerHTML={{ __html: renderMarkdown(msg.content) }} />
                  </div>
                ))}

                {/* Lesson Plan Card */}
                {lastLesson?.lesson_plan && (
                  <div style={{ padding: 16, borderRadius: 12, background: "rgba(59,130,246,0.08)", border: "1px solid rgba(59,130,246,0.2)" }}>
                    <div style={{ fontSize: 12, fontWeight: 700, color: "#60a5fa", marginBottom: 8 }}><i className="fas fa-map" style={{ marginRight: 6 }} />Lesson Plan</div>
                    <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 4 }}>{lastLesson.lesson_plan.topic}</div>
                    <div style={{ fontSize: 11, color: "#94a3b8", marginBottom: 8 }}>{lastLesson.lesson_plan.lesson_type} · {lastLesson.lesson_plan.difficulty} · {lastLesson.lesson_plan.estimated_time} min</div>
                    <div style={{ fontSize: 11, color: "#94a3b8" }}><strong>Objectives:</strong> {lastLesson.lesson_plan.objectives.join(", ")}</div>
                  </div>
                )}

                {/* Key Points */}
                {lastLesson && lastLesson.key_points.length > 0 && (
                  <div style={{ padding: 16, borderRadius: 12, background: "rgba(16,185,129,0.05)", border: "1px solid rgba(16,185,129,0.15)" }}>
                    <div style={{ fontSize: 12, fontWeight: 700, color: "#34d399", marginBottom: 8 }}><i className="fas fa-key" style={{ marginRight: 6 }} />Key Points</div>
                    <ul style={{ margin: 0, paddingLeft: 16, fontSize: 12, color: "#cbd5e1" }}>
                      {lastLesson.key_points.map((p, i) => <li key={i} style={{ margin: "4px 0" }}>{p}</li>)}
                    </ul>
                  </div>
                )}

                {/* Common Pitfalls */}
                {lastLesson && lastLesson.common_pitfalls.length > 0 && (
                  <div style={{ padding: 16, borderRadius: 12, background: "rgba(239,68,68,0.05)", border: "1px solid rgba(239,68,68,0.15)" }}>
                    <div style={{ fontSize: 12, fontWeight: 700, color: "#f87171", marginBottom: 8 }}><i className="fas fa-triangle-exclamation" style={{ marginRight: 6 }} />Common Pitfalls</div>
                    <ul style={{ margin: 0, paddingLeft: 16, fontSize: 12, color: "#cbd5e1" }}>
                      {lastLesson.common_pitfalls.map((p, i) => <li key={i} style={{ margin: "4px 0" }}>{p}</li>)}
                    </ul>
                  </div>
                )}

                {/* Practice Hint */}
                {lastLesson?.practice_hint && (
                  <div style={{ padding: 16, borderRadius: 12, background: "rgba(251,191,36,0.05)", border: "1px solid rgba(251,191,36,0.15)" }}>
                    <div style={{ fontSize: 12, fontWeight: 700, color: "#fbbf24", marginBottom: 8 }}><i className="fas fa-lightbulb" style={{ marginRight: 6 }} />Practice Hint</div>
                    <div style={{ fontSize: 12, color: "#cbd5e1" }} dangerouslySetInnerHTML={{ __html: renderMarkdown(lastLesson.practice_hint) }} />
                  </div>
                )}

                {/* Quiz Question */}
                {lastLesson?.quiz_question && (
                  <div style={{ padding: 16, borderRadius: 12, background: "rgba(139,92,246,0.08)", border: "1px solid rgba(139,92,246,0.2)" }}>
                    <div style={{ fontSize: 12, fontWeight: 700, color: "#a78bfa", marginBottom: 10 }}><i className="fas fa-clipboard-question" style={{ marginRight: 6 }} />Practice Question</div>
                    <div style={{ fontSize: 13, color: "#e2e8f0", marginBottom: 12, lineHeight: 1.6 }} dangerouslySetInnerHTML={{ __html: renderMarkdown(lastLesson.quiz_question.question) }} />
                    {lastLesson.quiz_question.options && lastLesson.quiz_question.options.length > 0 && (
                      <div style={{ display: "flex", flexDirection: "column", gap: 8, marginBottom: 12 }}>
                        {lastLesson.quiz_question.options.map((opt: string) => (
                          <button key={opt} onClick={() => setQuizAnswer(opt)} style={{ padding: "10px 12px", borderRadius: 10, border: `1px solid ${quizAnswer === opt ? "#a78bfa" : "rgba(255,255,255,0.1)"}`, background: quizAnswer === opt ? "rgba(139,92,246,0.15)" : "rgba(255,255,255,0.03)", color: "#e2e8f0", fontSize: 12, textAlign: "left", cursor: "pointer" }} dangerouslySetInnerHTML={{ __html: renderMarkdown(opt) }} />
                        ))}
                      </div>
                    )}
                    {!lastLesson.quiz_question.options && (
                      <textarea value={quizAnswer} onChange={(e) => setQuizAnswer(e.target.value)} placeholder="Type your answer here..." rows={3} style={{ width: "100%", padding: 10, borderRadius: 10, border: "1px solid rgba(255,255,255,0.1)", background: "rgba(255,255,255,0.03)", color: "#e2e8f0", fontSize: 12, marginBottom: 12, resize: "vertical" }} />
                    )}
                    <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
                      <button onClick={handleQuizSubmit} disabled={evaluating || !quizAnswer.trim()} style={{ padding: "8px 18px", borderRadius: 10, border: "none", background: "linear-gradient(135deg,#8b5cf6,#6366f1)", color: "#fff", fontWeight: 600, fontSize: 12, cursor: evaluating ? "wait" : "pointer", opacity: evaluating || !quizAnswer.trim() ? 0.6 : 1 }}>
                        {evaluating ? <><i className="fas fa-spinner fa-spin" style={{ marginRight: 6 }} />Checking...</> : "Check Answer"}
                      </button>
                      <span style={{ fontSize: 11, color: "#64748b" }}><i className="fas fa-circle-info" style={{ marginRight: 4 }} />{lastLesson.quiz_question.hint}</span>
                    </div>
                  </div>
                )}

                {/* Evaluation Result */}
                {evaluation && (
                  <div style={{ padding: 16, borderRadius: 12, background: evaluation.evaluation.is_correct ? "rgba(16,185,129,0.08)" : "rgba(239,68,68,0.08)", border: `1px solid ${evaluation.evaluation.is_correct ? "rgba(16,185,129,0.2)" : "rgba(239,68,68,0.2)"}` }}>
                    <div style={{ fontSize: 13, fontWeight: 700, color: evaluation.evaluation.is_correct ? "#34d399" : "#f87171", marginBottom: 10 }}>
                      <i className={`fas ${evaluation.evaluation.is_correct ? "fa-check-circle" : "fa-times-circle"}`} style={{ marginRight: 6 }} />
                      {evaluation.evaluation.is_correct ? "Correct!" : "Not quite — let us learn from this."} (Score: {Math.round(evaluation.evaluation.score * 100)}%)
                    </div>
                    <div style={{ fontSize: 12, color: "#cbd5e1", marginBottom: 8, lineHeight: 1.6 }} dangerouslySetInnerHTML={{ __html: renderMarkdown(evaluation.evaluation.feedback) }} />
                    {evaluation.evaluation.what_was_right && <div style={{ fontSize: 11, color: "#34d399", marginBottom: 4 }}><strong>What was right:</strong> {evaluation.evaluation.what_was_right}</div>}
                    {evaluation.evaluation.what_was_wrong && <div style={{ fontSize: 11, color: "#f87171", marginBottom: 4 }}><strong>What to fix:</strong> {evaluation.evaluation.what_was_wrong}</div>}
                    {evaluation.evaluation.how_to_improve && <div style={{ fontSize: 11, color: "#fbbf24", marginBottom: 4 }}><strong>How to improve:</strong> {evaluation.evaluation.how_to_improve}</div>}
                    {evaluation.evaluation.should_retry && <div style={{ fontSize: 11, color: "#94a3b8", marginTop: 8 }}><i className="fas fa-redo" style={{ marginRight: 4 }} />{evaluation.evaluation.next_question_hint}</div>}
                  </div>
                )}

                <div ref={chatEndRef} />
              </div>

              <form onSubmit={handleTeach} style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                <textarea value={input} onChange={(e) => setInput(e.target.value)} placeholder="Ask a question or type 'quiz me on X'..." rows={3} style={{ width: "100%", padding: 12, borderRadius: 12, border: "1px solid rgba(255,255,255,0.1)", background: "rgba(255,255,255,0.03)", color: "#e2e8f0", fontSize: 13, resize: "vertical" }} />
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                  <span style={{ fontSize: 11, color: "#64748b" }}>Press Enter to learn</span>
                  <button type="submit" disabled={teachLoading} style={{ padding: "10px 24px", borderRadius: 12, border: "none", background: "linear-gradient(135deg,#3b82f6,#8b5cf6)", color: "#fff", fontWeight: 600, fontSize: 13, cursor: teachLoading ? "wait" : "pointer", opacity: teachLoading ? 0.7 : 1 }}>
                    {teachLoading ? <><i className="fas fa-spinner fa-spin" style={{ marginRight: 6 }} />Teaching...</> : "Teach Me"}
                  </button>
                </div>
                {teachError && <div style={{ fontSize: 12, color: "#f87171", padding: 8 }}>{teachError}</div>}
              </form>
            </div>
          </div>
        </div>
      </main>

      {/* Agent Modal */}
      {showModal && modalAgent && (
        <div style={{ position: "fixed", inset: 0, zIndex: 100, background: "rgba(0,0,0,0.6)", backdropFilter: "blur(4px)", display: "flex", alignItems: "center", justifyContent: "center", padding: 24 }} onClick={(e) => { if (e.target === e.currentTarget) setShowModal(false); }}>
          <div style={{ background: "rgba(30,41,59,0.95)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 16, padding: 24, maxWidth: 440, width: "100%", maxHeight: "80vh", overflowY: "auto" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
              <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
                <div style={{ width: 44, height: 44, borderRadius: 12, background: "rgba(255,255,255,0.05)", display: "flex", alignItems: "center", justifyContent: "center" }}>
                  <i className={`fas ${modalAgent.icon}`} style={{ color: agentColor[modalAgent.id] || "#94a3b8", fontSize: 18 }} />
                </div>
                <div>
                  <div style={{ fontSize: 16, fontWeight: 700, color: agentColor[modalAgent.id] || "#e2e8f0" }}>{modalAgent.title}</div>
                  <div style={{ fontSize: 12, color: "#64748b" }}>{modalAgent.subtitle}</div>
                </div>
              </div>
              <button onClick={() => setShowModal(false)} style={{ width: 32, height: 32, borderRadius: 8, border: "none", background: "rgba(255,255,255,0.06)", color: "#94a3b8", cursor: "pointer" }}><i className="fas fa-xmark" /></button>
            </div>
            <p style={{ fontSize: 13, color: "#cbd5e1", lineHeight: 1.6, marginBottom: 16 }}>{modalAgent.description}</p>
            <div style={{ marginBottom: 16 }}>
              <div style={{ fontSize: 13, fontWeight: 600, color: "#e2e8f0", marginBottom: 8 }}>Responsibilities</div>
              <ul style={{ spaceY: 2, paddingLeft: 16, fontSize: 12, color: "#cbd5e1" }}>
                {modalAgent.responsibilities.map((r) => <li key={r} style={{ margin: "4px 0" }}>{r}</li>)}
              </ul>
            </div>
            <div>
              <div style={{ fontSize: 13, fontWeight: 600, color: "#e2e8f0", marginBottom: 8 }}>Tools</div>
              <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
                {modalAgent.tools.map((t) => <span key={t} style={{ fontSize: 11, padding: "4px 10px", borderRadius: 999, background: "rgba(255,255,255,0.05)", color: "#94a3b8", border: "1px solid rgba(255,255,255,0.08)" }}>{t}</span>)}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
