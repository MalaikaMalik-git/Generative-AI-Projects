# Personal Avatar Assistant

A 3D AI avatar embedded on a web page that answers questions about
**Agentix System** (agentixsystem.com) by voice or text, grounded in the
company's real website content — plus a camera mode where it can look at
and describe whatever you show it.

Built as a one-day MVP: complete end-to-end experience, not a
production-grade avatar.

## Features
- 🗣️ **Voice in** — hold the mic button, ask a question, release
- 🔊 **Voice out** — answers are read aloud, avatar animates while talking
- 📚 **RAG-grounded chat** — answers come from real scraped content from
  agentixsystem.com, not invented facts
- 📷 **Camera vision** — show it something, it describes what it sees
- 💬 **Text fallback** — always available, works even if mic/camera
  permissions are denied
- 🤖 **3D avatar** — idles, rotates, and animates while responding

## Architecture

```
┌─────────────┐         ┌──────────────┐         ┌─────────────────┐
│   Browser   │  HTTP   │   FastAPI    │         │  ChromaDB        │
│  (React +   │────────▶│   Backend    │────────▶│  (local vector   │
│   R3F +     │         │              │         │   store)         │
│  Web Speech │◀────────│  /chat       │         └─────────────────┘
│    APIs)    │  JSON   │  /vision     │                 ▲
└─────────────┘         │  /health     │                 │
                         └──────┬───────┘         ┌───────┴────────┐
                                │                  │  scripts/      │
                                ▼                  │  ingest.py     │
                         ┌──────────────┐          │  (chunks +     │
                         │   OpenAI     │          │  embeds site   │
                         │  gpt-4o-mini │          │  content once) │
                         │ (chat+vision)│          └────────────────┘
                         └──────────────┘
```

**Frontend** (React + Vite): renders the 3D avatar (Three.js /
React Three Fiber), the chat panel, and drives the browser's native
speech-to-text and text-to-speech APIs directly — no server round-trip
needed for either.

**Backend** (FastAPI): three endpoints.
- `GET /health` — sanity check
- `POST /chat` — embeds the question, retrieves the top matching chunks
  from ChromaDB, sends them + the question to `gpt-4o-mini`, returns a
  grounded answer
- `POST /vision` — takes a captured camera frame (base64 data URL) +
  optional question, sends both directly to `gpt-4o-mini`'s vision
  capability, returns a description

**One AI provider for both jobs:** `gpt-4o-mini` handles chat and vision,
so there's exactly one API key to manage.

**RAG data source:** `agentixsystem.com` only. `shammarianas.com` was
scraped and found to be a pure JavaScript-rendered shell with no static
text to extract — see `backend/SESSION2_README.md` for the full story.
Rather than spend session time on a headless-browser scraper for one
thin page, the project ships with one well-covered source.

## Setup (from scratch)

**Prerequisites:** Python 3.10+, Node.js 18+, an OpenAI API key
([get one here](https://platform.openai.com/api-keys)).

```bash
# 1. Backend
cd backend
python3 -m venv .venv
source .venv/bin/activate          
pip install -r requirements.txt
cp .env.example .env
# Edit .env and set OPENAI_API_KEY=sk-...

# 2. Build the RAG index (one-time, needs internet to download the
#    embedding model on first run)
python scripts/ingest.py

# 3. Start the backend
uvicorn main:app --reload --port 8000
```

In a second terminal:
```bash
cd frontend
npm install
npm run dev
```

Open **http://localhost:5173**. Allow microphone and camera permissions
when prompted (or don't — the app degrades gracefully to text-only, see
Limitations below).


## Limitations
Being upfront about what this MVP does and doesn't do:

- **Single knowledge source.** Only `agentixsystem.com` is in the RAG
  index — `shammarianas.com` is JS-rendered and wasn't scraped (see
  Architecture above). Questions specifically about Sham Marianas FZC
  beyond its one-line mention as Agentix's parent company won't be
  answered.
- **No real lip-sync.** The avatar's "speaking" animation is a gesture
  (Wave) triggered while audio plays, not synced mouth movement — this
  was an explicit scope cut in the project plan to keep the timeline
  realistic.
- **No literal blinking.** The chosen avatar model (RobotExpressive, a
  robot) has no eyelids; a small head-tic substitutes for it.
- **Browser-dependent voice input.** Speech-to-text is reliable in
  Chrome/Edge, partial in Safari, and off by default in Firefox. Text
  input always works regardless — **demo in Chrome**.
- **Single camera frame, not continuous video.** By design — the plan
  explicitly scoped out real-time video analysis.
- **Free-tier hosting caveats.** If deployed per `DEPLOYMENT.md`, the
  free backend tier sleeps after inactivity (~30-60s cold start) and the
  RAG index rebuilds from scratch on every deploy rather than persisting.
- **No conversation memory.** Each question is answered independently;
  the assistant doesn't recall earlier turns in the same session.
- **No automated test suite ships with the repo.** Backend logic was
  verified through manual test scripts during development (documented in
  each session's README) rather than a checked-in `pytest` suite — a
  reasonable cut for a one-day timeline, but worth knowing if this goes
  beyond MVP.

## Project structure
```
avatar-assistant/
├── README.md              ← you are here
├── DEPLOYMENT.md           deployment steps (Render + Vercel)
├── backend/
│   ├── main.py             FastAPI app: /health, /chat, /vision
│   ├── rag.py               ChromaDB retrieval
│   ├── llm.py                OpenAI chat wrapper
│   ├── vision.py              OpenAI vision wrapper
│   ├── requirements.txt
│   ├── .env.example
│   ├── data/raw/            scraped agentixsystem.com content
│   ├── data/chroma_db/      vector store (built by ingest.py)
│   ├── scripts/             scrape.py, ingest.py, test_retrieval.py
│ 
└── frontend/
    ├── src/
    │   ├── App.jsx                     top-level layout
    │   ├── AvatarViewer.jsx             3D canvas + lighting
    │   ├── RobotAvatar.jsx              avatar model + animation logic
    │   ├── ChatPanel.jsx                chat UI, mic, camera trigger
    │   ├── CameraCapture.jsx            camera permission + capture
    │   ├── useSpeechRecognition.js      voice input hook
    │   └── useTextToSpeech.js           voice output hook
    ├── public/models/RobotExpressive.glb
    ├── .env.example
```
