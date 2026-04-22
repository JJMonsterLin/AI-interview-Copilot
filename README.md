# 🎯 Local AI Interview Copilot

A fully **local, offline** AI-powered interview practice system built for **NVIDIA Jetson Orin NX**. It uses on-device speech recognition, a local LLM, and text-to-speech to simulate a realistic interview experience — no cloud services required.

---

## Features

- 🎤 **Voice interaction** — speak your answers, hear the AI's questions
- 🤖 **Intelligent follow-up questions** — the LLM adapts based on your answer quality
- 📊 **Real-time scoring** — each answer is scored on Completeness, Depth, Clarity, and Structure (0–10)
- 💡 **Instant feedback** — actionable suggestions after every answer
- 📁 **Session history** — review past interviews with score trend charts
- 🗂️ **JD / Resume upload** — personalise questions from a PDF, DOCX, or pasted text
- 🔒 **100% local** — all inference runs on-device via Ollama, Whisper, and Kokoro TTS

---

## System Architecture

```
Browser ──► FastAPI (port 3000)
               ├── ASR  ──► faster-whisper  (port 5092)
               ├── LLM  ──► Ollama          (port 11434)
               └── TTS  ──► Kokoro TTS      (port 8880)
```

All services communicate over `localhost` using Docker `network_mode: host`.

---

## Hardware Requirements

| Component | Minimum |
|-----------|---------|
| Platform  | NVIDIA Jetson Orin NX (or equivalent Jetson with JetPack 6) |
| RAM       | 16 GB unified memory |
| Storage   | 20 GB free (for Docker images + model weights) |
| GPU       | CUDA-capable (Jetson integrated GPU) |

---

## Services

| Service | Image | Port |
|---------|-------|------|
| Ollama (LLM) | `dustynv/ollama:r36.4.0-cu128-24.04` | 11434 |
| ASR (Whisper) | `faster-whisper:fastapi` | 5092 |
| TTS (Kokoro) | `dustynv/kokoro-tts:fastapi-r36.4.0-cu128-24.04` | 8880 |
| Web App | Built from `Dockerfile` | 3000 |

### Supported LLM Models

| Model | Tag | Notes |
|-------|-----|-------|
| Qwen2.5 3B | `qwen2.5:3b` | Default, fast on Jetson |
| Gemma2 2B | `gemma2:2b` | Google, compact |

Models are pulled automatically on first startup.

---

## Quick Start

### Prerequisites

- Docker and Docker Compose installed
- NVIDIA Container Runtime configured (`nvidia-docker2`)
- The `faster-whisper:fastapi` image built locally (see Lab 3)

### 1. Clone the repository

```bash
git clone https://github.com/JJMonsterLin/AI-interview-Copilot.git
cd AI-interview-Copilot
```

### 2. (First run only) Build the faster-whisper ASR image

Follow the instructions in Lab 3 to build `faster-whisper:fastapi`, or run:

```bash
# Example — adjust path to your Lab3 Dockerfile
docker build -t faster-whisper:fastapi ./asr
```

### 3. Start all services

```bash
docker compose up --build
```

> **Note:** The first run downloads Qwen2.5 3B and Gemma2 2B (~5 GB total). This may take 10–20 minutes depending on your network speed. The app will start automatically once models are ready.

### 4. Open the web interface

Navigate to [http://localhost:3000](http://localhost:3000) in your browser.

---

## Usage

1. **Setup tab** — choose job title, interview type (Technical / Behavioural / Comprehensive), difficulty, and optionally upload a job description or resume.
2. **Start Interview** — the AI generates an opening question and reads it aloud.
3. **Record Answer** — click the microphone button to record your answer, then click Stop.
4. The system transcribes your speech, scores your answer, and asks a follow-up question.
5. **End Interview** — click ⏹ End to finish. The session is saved to History.
6. **History tab** — review all past sessions with per-turn scores and a trend chart.

---

## Project Structure

```
Project/
├── app/
│   ├── main.py          # FastAPI backend
│   └── static/
│       └── index.html   # Single-page frontend
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ASR_URL` | `http://localhost:5092` | Faster-Whisper service URL |
| `LLM_URL` | `http://localhost:11434` | Ollama service URL |
| `TTS_URL` | `http://localhost:8880` | Kokoro TTS service URL |
| `DB_PATH` | `/data/interview.db` | SQLite database path |
| `MAX_TURNS` | `6` | Max turns per interview session |
| `APP_PORT` | `3000` | Web app listening port |

---

## Tech Stack

- **Backend**: Python 3.11, FastAPI, httpx, SQLite
- **Frontend**: Vanilla JS, Chart.js 4
- **ASR**: OpenAI Whisper (faster-whisper) via OpenAI-compatible API
- **LLM**: Ollama with Qwen2.5 / Gemma2
- **TTS**: Kokoro TTS (Jetson-optimised)
- **Deployment**: Docker Compose on NVIDIA Jetson Orin NX

---

## Course

IEMS 5709 — Edge Computing, CUHK, 2025 Spring R2
