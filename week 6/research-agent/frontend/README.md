# Research Agent
**Week 10 — Streaming GenAI API + React Frontend**
*FastAPI · SSE streaming · React + Vite · Docker · Railway · Vercel*

A ReAct research agent wrapped in a production FastAPI backend with a streaming React chat UI. Ask a question, watch the agent decompose it into sub-questions, research each one live, and stream back a structured report with clickable source citations.

---

## One-command local run

```bash
# 1. Clone and enter the project
cd research-agent

# 2. Backend setup
cp .env.example .env
# Edit .env — set OPENAI_API_KEY and ALLOWED_API_KEYS

# 3. Start backend (Docker)
docker compose up --build

# 4. Frontend setup (new terminal)
cd frontend
cp .env.example .env.local
# .env.local already points to http://localhost:8000 — no edits needed
npm install
npm run dev
```

Open `http://localhost:5173` — type a research question and watch tokens stream in.

---

## Local run without Docker

```bash
# Backend
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # fill in OPENAI_API_KEY + ALLOWED_API_KEYS
python3 -m uvicorn api.main:app --reload

# Frontend (separate terminal)
cd frontend
npm install && npm run dev
```

---

## Deploy to production

### Backend → Railway

1. Push this repo to GitHub
2. Go to [railway.app](https://railway.app) → New Project → Deploy from GitHub repo
3. Select the `research-agent` folder as the root
4. Railway auto-detects `railway.json` and `Dockerfile`
5. In Railway dashboard → Variables, set:

| Variable | Value |
|---|---|
| `OPENAI_API_KEY` | your OpenAI key |
| `ALLOWED_API_KEYS` | `prod-key-1,prod-key-2` |
| `OPENAI_MODEL` | `gpt-4o-mini` |
| `FRONTEND_URL` | *(set after frontend deploy — step below)* |

6. Deploy. Copy the Railway public URL (e.g. `https://research-agent.up.railway.app`)
7. Verify: `curl https://your-backend.up.railway.app/health`

### Backend → Render (alternative)

1. Go to [render.com](https://render.com) → New → Web Service → Connect repo
2. Render detects `render.yaml` automatically
3. Set `OPENAI_API_KEY` and `ALLOWED_API_KEYS` in the Render dashboard under Environment
4. Deploy. Copy the Render URL.

### Frontend → Vercel

1. Go to [vercel.com](https://vercel.com) → New Project → Import your GitHub repo
2. Set the **Root Directory** to `frontend`
3. In Environment Variables, set:

| Variable | Value |
|---|---|
| `VITE_API_BASE_URL` | `https://your-backend.up.railway.app` |
| `VITE_API_KEY` | a key from `ALLOWED_API_KEYS` on the backend |

4. Deploy. Copy the Vercel URL (e.g. `https://research-agent.vercel.app`)

### Final step — wire CORS

Go back to Railway → Variables, set:
```
FRONTEND_URL=https://research-agent.vercel.app
```
Redeploy the backend. CORS is now locked to your frontend domain only.

---

## Project structure

```
research-agent/
├── agent/                    ← Week 6 ReAct agent (unchanged)
│   ├── decomposer.py         question → sub-questions
│   ├── react_loop.py         think → search → observe → answer
│   ├── synthesizer.py        sub-answers → structured markdown report
│   └── error_handler.py
├── tools/
│   ├── search.py             DuckDuckGo (no API key needed)
│   └── fetch.py              page content fetcher
├── api/                      ← FastAPI layer
│   ├── main.py               app + CORS config
│   ├── models.py             Pydantic schemas
│   ├── routers/
│   │   ├── health.py         GET  /health
│   │   ├── chat.py           POST /chat · GET /chat/stream (SSE)
│   │   └── history.py        GET/DELETE /history/{session_id}
│   ├── middleware/
│   │   ├── auth.py           X-API-Key → 401
│   │   └── rate_limiter.py   sliding window 10 req/hr → 429
│   └── services/
│       ├── agent_runner.py   FastAPI → agent bridge
│       └── cost_logger.py    token + cost tracking
├── frontend/                 ← React + Vite chat UI
│   └── src/
│       ├── App.tsx           layout + sidebar wiring
│       ├── hooks/
│       │   └── useChatStream.ts   SSE client, session tracking
│       ├── components/
│       │   ├── MessageBubble.tsx  answer rendering + citations
│       │   ├── CitationChips.tsx  clickable source chips
│       │   ├── HistorySidebar.tsx past sessions
│       │   ├── MessageSkeleton.tsx shimmer loading state
│       │   ├── MessageList.tsx
│       │   └── ChatInput.tsx
│       └── lib/
│           ├── parseCitations.ts  splits ## Sources → Citation[]
│           └── config.ts          env vars
├── scripts/
│   └── test_stream.py        terminal SSE demo
├── tests/
│   └── test_api.py
├── Dockerfile                multi-stage, supports $PORT
├── docker-compose.yml        local one-command setup
├── railway.json              Railway deploy config
├── render.yaml               Render deploy config
├── ARCHITECTURE.md           Mermaid system diagram
├── requirements.txt
├── .env.example              backend env template (no real keys)
└── .gitignore
```

---

## API reference

All endpoints except `/health` require:
```
X-API-Key: <your-key>
```

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Status check, no auth |
| `POST` | `/chat` | Sync — waits for full answer |
| `GET` | `/chat/stream` | SSE — streams tokens live |
| `GET` | `/history/{id}` | Past Q&A for a session |
| `DELETE` | `/history/{id}` | Clear session history |
| `GET` | `/docs` | Swagger UI |

### SSE event types (`/chat/stream`)

| type | payload | when |
|---|---|---|
| `status` | `content: str` | each pipeline stage |
| `chunk` | `content: str` | each word group of the answer |
| `done` | `session_id, success, usage` | pipeline complete |
| `error` | `content: str` | something failed |

---

## Environment variables

### Backend (`.env`)

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENAI_API_KEY` | ✅ | — | OpenAI key |
| `OPENAI_MODEL` | | `gpt-4o-mini` | Model name |
| `ALLOWED_API_KEYS` | ✅ | — | Comma-separated valid keys |
| `RATE_LIMIT_REQUESTS` | | `10` | Max requests per window |
| `RATE_LIMIT_WINDOW_SECONDS` | | `3600` | Window size in seconds |
| `FRONTEND_URL` | prod only | `""` | Locks CORS to this origin |
| `MAX_STEPS` | | `6` | ReAct loop steps per sub-question |
| `REQUEST_TIMEOUT` | | `5` | HTTP timeout for search/fetch |

### Frontend (`.env.local`)

| Variable | Required | Description |
|---|---|---|
| `VITE_API_BASE_URL` | ✅ | Backend URL, no trailing slash |
| `VITE_API_KEY` | ✅ | Must match a value in `ALLOWED_API_KEYS` |

---

## Demo script (terminal)

```bash
# With backend running:
python3 scripts/test_stream.py "What caused the 2008 financial crisis?"
```

Runs auth rejection → rate limit checks → full streaming research pipeline with live output.

---

## Running tests

```bash
pytest tests/test_api.py -v
# 26 tests, no real API or network calls
```

---

## Cost logging

Every request appends one JSON line to `cost_log.jsonl`:
```json
{"ts": "2026-06-23T10:00:00", "session_id": "abc...", "question": "...", "tokens": 4241, "cost_usd": 0.001197}
```

---

## Architecture

See [ARCHITECTURE.md](./ARCHITECTURE.md) for the full Mermaid system diagram.

```
User → React UI → useChatStream (SSE fetch)
                       ↓
              FastAPI /chat/stream
                  auth → rate limit
                       ↓
              agent_runner.stream_agent
                  decompose → ReAct×N → synthesize
                  yield SSE: status | chunk | done
                       ↓
              parseCitations → citation chips
                       ↓
              Rendered answer in MessageBubble
```
