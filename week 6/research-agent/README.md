# Research Agent API
**Week 10 Demo — Streaming GenAI API**
*FastAPI · async Python · SSE streaming · Docker · Auth · Rate limiting*

A production-structured FastAPI backend wrapping the Week 6 ReAct research agent.
Streaming enabled. Dockerized. Runnable with one command. Demoed from the terminal.

---

## Quickstart (one command)

```bash
# 1. Copy and fill in your keys
cp .env.example .env
# Edit .env: set OPENAI_API_KEY and ALLOWED_API_KEYS

# 2. Start
docker compose up --build

# 3. Hit the API
curl -H "X-API-Key: your-secret-key-here" http://localhost:8000/health
```

---

## Running locally (without Docker)

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # fill in OPENAI_API_KEY and ALLOWED_API_KEYS
python3 -m uvicorn api.main:app --reload
```

---

## Project structure

```
research-agent/
├── agent/                  ← Week 6 ReAct agent (unchanged)
│   ├── config.py
│   ├── client.py
│   ├── decomposer.py       decompose question → sub-questions
│   ├── react_loop.py       ReAct loop (think → search → observe → answer)
│   ├── synthesizer.py      combine sub-answers → structured report
│   ├── error_handler.py    graceful failure on search/LLM errors
│   ├── models.py
│   └── tracer.py
├── tools/
│   ├── search.py           DuckDuckGo search (no API key needed)
│   └── fetch.py            page fetcher
├── api/                    ← Week 10: FastAPI layer
│   ├── main.py             app entry point, router registration
│   ├── models.py           Pydantic request/response schemas
│   ├── routers/
│   │   ├── health.py       GET  /health
│   │   ├── chat.py         POST /chat  ·  GET /chat/stream
│   │   └── history.py      GET  /history/{session_id}
│   ├── middleware/
│   │   ├── auth.py         X-API-Key header check → 401
│   │   └── rate_limiter.py sliding window, 10 req/hr per key → 429
│   └── services/
│       ├── agent_runner.py bridge: FastAPI → research agent pipeline
│       └── cost_logger.py  token + cost estimation, writes cost_log.jsonl
├── scripts/
│   └── test_stream.py      terminal SSE demo + auth/rate-limit checks
├── tests/
│   └── test_api.py         26 pytest tests (no real API calls)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

## API reference

All endpoints except `/health` require the header:
```
X-API-Key: <your-key>
```

### `GET /health`
No auth required.
```json
{"status": "ok", "version": "1.0.0", "model": "gpt-4o-mini", "agent": "research-agent-v6"}
```

### `POST /chat`
Synchronous. Waits for full answer before returning.
```bash
curl -X POST http://localhost:8000/chat \
  -H "X-API-Key: your-key" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is quantum computing?"}'
```
Response:
```json
{
  "session_id": "abc123...",
  "question": "What is quantum computing?",
  "answer": "## Summary\n...",
  "success": true,
  "usage": {"total_tokens": 4200, "estimated_cost_usd": 0.0012},
  "timestamp": "2026-06-06T10:00:00"
}
```

### `GET /chat/stream`
Server-Sent Events. Tokens arrive in real time.
```bash
curl -N "http://localhost:8000/chat/stream?question=What+is+JWST" \
  -H "X-API-Key: your-key"
```
Each event: `data: {"type": "status"|"chunk"|"done"|"error", ...}`

| type | payload | meaning |
|------|---------|---------|
| `status` | `content: str` | pipeline stage update |
| `chunk` | `content: str` | partial answer text |
| `done` | `session_id, success, usage` | pipeline finished |
| `error` | `content: str` | something failed |

### `GET /history/{session_id}`
```bash
curl http://localhost:8000/history/abc123 -H "X-API-Key: your-key"
```

### `DELETE /history/{session_id}`
Clears history for a session.

### `GET /docs`
Interactive Swagger UI — try every endpoint in the browser.

---

## Auth

Set `ALLOWED_API_KEYS` in `.env` as a comma-separated list:
```
ALLOWED_API_KEYS=key-one,key-two,key-three
```
Missing or invalid key → **401 Unauthorized**.
Leave empty to accept any key (dev mode only).

---

## Rate limiting

Per API key, sliding window. Defaults: **10 requests / hour**.
Exceeded → **429 Too Many Requests** with `Retry-After` header.

Configure in `.env`:
```
RATE_LIMIT_REQUESTS=10
RATE_LIMIT_WINDOW_SECONDS=3600
```

---

## Running tests

```bash
pytest tests/test_api.py -v
# 26 passed — no real OpenAI or network calls
```

---

## Demo from terminal

```bash
# Start server
python3 -m uvicorn api.main:app --reload

# New terminal — stream a full research query live
python3 scripts/test_stream.py "What caused the 2008 financial crisis?"
```

The script runs auth rejection + rate limit checks automatically, then streams the full research pipeline with live status updates and token-by-token output.

---

## Cost logging

Every request appends one line to `cost_log.jsonl`:
```json
{"ts": "2026-06-06T10:00:00", "session_id": "abc...", "question": "...", "tokens": 4241, "cost_usd": 0.001197}
```

---

## Architecture

```
Client
  │  POST /chat  or  GET /chat/stream
  ▼
FastAPI  ──► auth middleware (X-API-Key)
         ──► rate limiter (sliding window)
         ──► agent_runner
               │
               ├─ decompose(question)     → 3–5 sub-questions
               ├─ ReAct loop × N          → search + synthesize per sub-q
               └─ safe_synthesize(...)    → structured markdown report
                        │
                        ▼
               StreamingResponse (SSE)  or  ChatResponse (JSON)
```