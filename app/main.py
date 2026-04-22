import os
import uuid
import json
import base64
import sqlite3
import logging
import re
from datetime import datetime
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ASR_URL = os.getenv("ASR_URL", "http://localhost:5092")
LLM_URL = os.getenv("LLM_URL", "http://localhost:11434")
TTS_URL = os.getenv("TTS_URL", "http://localhost:8880")
DB_PATH = os.getenv("DB_PATH", "/data/interview.db")

# Available models: display name -> ollama model tag
AVAILABLE_MODELS = {
    "Qwen2.5 3B": "qwen2.5:3b",
    "Gemma2 2B (Google)": "gemma2:2b",
}
DEFAULT_MODEL = "qwen2.5:3b"

sessions: dict = {}


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
def get_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            job_title TEXT,
            interview_type TEXT,
            difficulty TEXT,
            job_description TEXT,
            resume_text TEXT,
            model_name TEXT,
            created_at TEXT,
            ended_at TEXT,
            overall_score REAL
        );
        CREATE TABLE IF NOT EXISTS turns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            turn_number INTEGER,
            question TEXT,
            answer_text TEXT,
            score_completeness REAL,
            score_depth REAL,
            score_clarity REAL,
            score_structure REAL,
            overall_score REAL,
            feedback TEXT,
            created_at TEXT,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        );
    """)
    # Add model_name column if upgrading from old schema
    try:
        conn.execute("ALTER TABLE sessions ADD COLUMN model_name TEXT")
        conn.commit()
    except Exception:
        pass
    conn.close()


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a professional interviewer conducting a {interview_type} interview for a {job_title} position at {difficulty} difficulty level.

{job_context}

Your task:
1. Analyse the candidate's latest answer carefully — consider depth, completeness, and accuracy.
2. If the answer is shallow or incomplete, ask a targeted follow-up on the SAME topic.
3. If the answer is thorough, move to a NEW relevant topic.
4. Score this answer on four dimensions (0-10 each).
5. Provide brief, specific feedback.

Previous conversation so far:
{history}

Respond ONLY with a single valid JSON object — no markdown fences, no commentary:
{{
  "follow_up_question": "Your next question here",
  "scores": {{
    "content_completeness": 7,
    "professional_depth": 6,
    "clarity": 8,
    "structure": 7
  }},
  "overall_score": 7,
  "feedback": "Specific, actionable feedback for this answer"
}}"""

OPENING_PROMPT = """You are a professional interviewer. Generate the FIRST question for a {interview_type} interview for a {job_title} position at {difficulty} difficulty.

{job_context}

Respond ONLY with a single valid JSON object — no markdown, no commentary:
{{
  "question": "Your opening interview question here"
}}"""


def build_job_context(session: dict) -> str:
    parts = []
    if session.get("job_description"):
        parts.append(f"Job Description:\n{session['job_description'][:1500]}")
    if session.get("resume_text"):
        parts.append(f"Candidate Resume:\n{session['resume_text'][:1000]}")
    return "\n\n".join(parts) if parts else "No specific job description provided."


def build_history_text(turns: list) -> str:
    if not turns:
        return "(none)"
    lines = []
    for t in turns:
        lines.append(f"Q{t['turn']}: {t['question']}")
        lines.append(f"A{t['turn']}: {t['answer']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# AI service helpers
# ---------------------------------------------------------------------------
async def call_asr(audio_bytes: bytes, filename: str = "audio.webm") -> str:
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            f"{ASR_URL}/v1/audio/transcriptions",
            files={"file": (filename, audio_bytes, "audio/webm")},
            data={"model": "faster-whisper"},
        )
        resp.raise_for_status()
        return resp.json().get("text", "").strip()


async def call_llm(messages: list, model: str, temperature: float = 0.7) -> str:
    """Call Ollama's OpenAI-compatible endpoint."""
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "stream": False,
    }
    log.info("LLM request — model=%s", model)
    async with httpx.AsyncClient(timeout=180) as client:
        resp = await client.post(
            f"{LLM_URL}/v1/chat/completions",
            json=payload,
        )
        if not resp.is_success:
            log.error("LLM error %d: %s", resp.status_code, resp.text[:300])
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"].strip()
        log.info("LLM response (first 200): %s", content[:200])
        return content


async def call_tts(text: str) -> bytes:
    payload = {
        "model": "tts-1",
        "input": text,
        "voice": "af_heart",
        "lang_code": "a",
        "response_format": "mp3",
    }
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            f"{TTS_URL}/v1/audio/speech",
            json=payload,
            headers={"Accept": "audio/mpeg"},
        )
        resp.raise_for_status()
        return resp.content


def parse_llm_json(text: str) -> dict:
    # Strip Qwen3 thinking blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # Strip markdown fences
    text = re.sub(r"```(?:json)?", "", text).strip()
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(f"No JSON in LLM response: {text[:300]}")
    chunk = text[start:end]
    try:
        return json.loads(chunk)
    except json.JSONDecodeError as e:
        log.error("JSON parse failed: %s\nChunk: %s", e, chunk[:400])
        raise


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    log.info("App ready. LLM_URL=%s, available models: %s", LLM_URL, list(AVAILABLE_MODELS.values()))
    yield


app = FastAPI(lifespan=lifespan)


# ---------------------------------------------------------------------------
# API: Models list
# ---------------------------------------------------------------------------
@app.get("/api/models")
async def list_models():
    """Return available models and which ones are ready in Ollama."""
    ready = set()
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{LLM_URL}/api/tags")
            if r.is_success:
                for m in r.json().get("models", []):
                    ready.add(m["name"].split(":")[0] + ":" + m["name"].split(":")[1] if ":" in m["name"] else m["name"])
    except Exception:
        pass

    result = []
    for display, tag in AVAILABLE_MODELS.items():
        result.append({
            "display": display,
            "tag": tag,
            "ready": tag in ready or any(tag.split(":")[0] in r for r in ready),
        })
    return result


# ---------------------------------------------------------------------------
# API: Documents
# ---------------------------------------------------------------------------
@app.post("/api/documents/parse")
async def parse_document(file: UploadFile):
    content = await file.read()
    text = ""
    fname = (file.filename or "").lower()
    try:
        if fname.endswith(".pdf"):
            import PyPDF2, io
            reader = PyPDF2.PdfReader(io.BytesIO(content))
            text = "\n".join(p.extract_text() or "" for p in reader.pages)
        elif fname.endswith(".docx"):
            import docx, io
            doc = docx.Document(io.BytesIO(content))
            text = "\n".join(p.text for p in doc.paragraphs)
        else:
            text = content.decode("utf-8", errors="ignore")
    except Exception as e:
        raise HTTPException(400, f"Failed to parse document: {e}")
    return {"text": text[:3000]}


# ---------------------------------------------------------------------------
# API: Session Init
# ---------------------------------------------------------------------------
class InitRequest(BaseModel):
    job_title: str
    interview_type: str = "Technical"
    difficulty: str = "Intermediate"
    model: str = DEFAULT_MODEL
    job_description: Optional[str] = None
    resume_text: Optional[str] = None


@app.post("/api/session/init")
async def session_init(req: InitRequest):
    session_id = str(uuid.uuid4())
    session_data = {
        "job_title": req.job_title,
        "interview_type": req.interview_type,
        "difficulty": req.difficulty,
        "model": req.model,
        "job_description": req.job_description or "",
        "resume_text": req.resume_text or "",
    }
    job_ctx = build_job_context(session_data)

    opening_prompt = OPENING_PROMPT.format(
        interview_type=req.interview_type,
        job_title=req.job_title,
        difficulty=req.difficulty,
        job_context=job_ctx,
    )

    try:
        raw = await call_llm(
            [{"role": "user", "content": opening_prompt}],
            model=req.model,
            temperature=0.8,
        )
        data = parse_llm_json(raw)
        question = data.get("question", "").strip()
        if not question:
            raise ValueError("Empty question")
    except Exception as e:
        log.warning("Opening question failed: %s — using fallback", e)
        question = f"Welcome! Could you give me a brief overview of your background and why you're interested in the {req.job_title} role?"

    try:
        audio_bytes = await call_tts(question)
        audio_b64 = base64.b64encode(audio_bytes).decode()
    except Exception as e:
        log.warning("TTS failed: %s", e)
        audio_b64 = ""

    sessions[session_id] = {
        **session_data,
        "turns": [],
        "current_question": question,
        "turn_count": 0,
        "created_at": datetime.utcnow().isoformat(),
    }

    conn = get_db()
    conn.execute(
        """INSERT INTO sessions
           (id, job_title, interview_type, difficulty, job_description, resume_text, model_name, created_at)
           VALUES (?,?,?,?,?,?,?,?)""",
        (session_id, req.job_title, req.interview_type, req.difficulty,
         req.job_description or "", req.resume_text or "", req.model,
         sessions[session_id]["created_at"]),
    )
    conn.commit()
    conn.close()

    return {"session_id": session_id, "question": question, "audio_b64": audio_b64}


# ---------------------------------------------------------------------------
# API: Submit Answer
# ---------------------------------------------------------------------------
@app.post("/api/session/{session_id}/answer")
async def submit_answer(session_id: str, audio: UploadFile):
    if session_id not in sessions:
        raise HTTPException(404, "Session not found")
    session = sessions[session_id]
    model = session.get("model", DEFAULT_MODEL)

    # 1. ASR
    audio_bytes = await audio.read()
    try:
        transcript = await call_asr(audio_bytes, audio.filename or "audio.webm")
    except Exception as e:
        log.warning("ASR failed: %s", e)
        transcript = "(transcription failed)"
    if not transcript:
        transcript = "(no speech detected)"

    turn_number = session["turn_count"] + 1
    session["turn_count"] = turn_number

    # 2. LLM
    history_text = build_history_text(session["turns"])
    job_ctx = build_job_context(session)

    system_msg = SYSTEM_PROMPT.format(
        interview_type=session["interview_type"],
        job_title=session["job_title"],
        difficulty=session["difficulty"],
        job_context=job_ctx,
        history=history_text,
    )
    user_msg = f"Current question: {session['current_question']}\nCandidate's answer: {transcript}"

    llm_ok = False
    try:
        raw = await call_llm(
            [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
            model=model,
            temperature=0.7,
        )
        llm_data = parse_llm_json(raw)
        follow_up = llm_data.get("follow_up_question", "").strip()
        scores = llm_data.get("scores", {})
        overall_score = float(llm_data.get("overall_score", 6.0))
        feedback = llm_data.get("feedback", "")
        if not follow_up:
            follow_up = "Can you give me a concrete example of that from your experience?"
        llm_ok = True
    except Exception as e:
        log.error("LLM failed on turn %d: %s", turn_number, e)
        follow_up = "Interesting — could you elaborate on that with a specific example?"
        scores = {"content_completeness": 6, "professional_depth": 6, "clarity": 6, "structure": 6}
        overall_score = 6.0
        feedback = "LLM unavailable — score is estimated."

    # 3. TTS
    try:
        audio_bytes_out = await call_tts(follow_up)
        audio_b64_out = base64.b64encode(audio_bytes_out).decode()
    except Exception as e:
        log.warning("TTS failed: %s", e)
        audio_b64_out = ""

    # 4. Save turn
    turn_record = {
        "turn": turn_number,
        "question": session["current_question"],
        "answer": transcript,
        "scores": scores,
        "overall_score": overall_score,
        "feedback": feedback,
    }
    session["turns"].append(turn_record)
    session["current_question"] = follow_up

    conn = get_db()
    conn.execute(
        """INSERT INTO turns
           (session_id, turn_number, question, answer_text,
            score_completeness, score_depth, score_clarity, score_structure,
            overall_score, feedback, created_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
        (session_id, turn_number,
         turn_record["question"], transcript,
         scores.get("content_completeness", 0), scores.get("professional_depth", 0),
         scores.get("clarity", 0), scores.get("structure", 0),
         overall_score, feedback, datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()

    return {
        "transcript": transcript,
        "follow_up": follow_up,
        "audio_b64": audio_b64_out,
        "scores": scores,
        "overall_score": overall_score,
        "feedback": feedback,
        "turn_number": turn_number,
        "llm_ok": llm_ok,
        "model": model,
    }


# ---------------------------------------------------------------------------
# API: End Session
# ---------------------------------------------------------------------------
@app.post("/api/session/{session_id}/end")
async def end_session(session_id: str):
    if session_id not in sessions:
        raise HTTPException(404, "Session not found")
    session = sessions[session_id]
    turns = session["turns"]
    avg_score = sum(t["overall_score"] for t in turns) / len(turns) if turns else 0.0
    ended_at = datetime.utcnow().isoformat()
    conn = get_db()
    conn.execute("UPDATE sessions SET ended_at=?, overall_score=? WHERE id=?",
                 (ended_at, avg_score, session_id))
    conn.commit()
    conn.close()
    sessions.pop(session_id, None)
    return {"session_id": session_id, "overall_score": avg_score, "total_turns": len(turns)}


# ---------------------------------------------------------------------------
# API: History
# ---------------------------------------------------------------------------
@app.get("/api/history")
async def get_history():
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM sessions WHERE ended_at IS NOT NULL ORDER BY created_at DESC LIMIT 50"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


@app.get("/api/history/{session_id}")
async def get_session_detail(session_id: str):
    conn = get_db()
    session = conn.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
    if not session:
        raise HTTPException(404, "Session not found")
    turns = conn.execute(
        "SELECT * FROM turns WHERE session_id=? ORDER BY turn_number", (session_id,)
    ).fetchall()
    conn.close()
    return {"session": dict(session), "turns": [dict(t) for t in turns]}


# ---------------------------------------------------------------------------
# Debug
# ---------------------------------------------------------------------------
@app.get("/api/debug/llm")
async def debug_llm():
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{LLM_URL}/api/tags")
            return {"status": "ok", "ollama_url": LLM_URL, "models": r.json()}
    except Exception as e:
        return {"status": "error", "error": str(e), "ollama_url": LLM_URL}


# ---------------------------------------------------------------------------
# Static files
# ---------------------------------------------------------------------------
app.mount("/static", StaticFiles(directory="/app/static"), name="static")


@app.get("/")
async def index():
    return FileResponse("/app/static/index.html")
