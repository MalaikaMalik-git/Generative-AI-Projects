# Architecture

```mermaid
flowchart TD
    User(["👤 User\n(Browser)"])

    subgraph Frontend ["Frontend — Vercel"]
        UI["React + Vite\nChat UI"]
        Hook["useChatStream hook\nSSE client"]
        Parser["parseCitations\n## Sources → chips"]
    end

    subgraph Backend ["Backend — Railway"]
        Auth["Auth middleware\nX-API-Key → 401"]
        Rate["Rate limiter\n10 req/hr/key → 429"]

        subgraph API ["FastAPI"]
            Health["GET /health"]
            Stream["GET /chat/stream\nStreamingResponse SSE"]
            Sync["POST /chat\nJSON response"]
            History["GET /history/:id"]
        end

        subgraph Agent ["Research Agent (Week 6)"]
            Decomp["decompose()\nquestion → sub-questions"]
            React["ReAct loop × N\nthink → search → observe"]
            Synth["safe_synthesize()\nsub-answers → report"]
        end

        CostLog["cost_log.jsonl\ntoken + cost audit"]
    end

    subgraph External ["External"]
        OpenAI(["OpenAI API\ngpt-4o-mini"])
        DDG(["DuckDuckGo\nSearch"])
    end

    User -->|"question (text)"| UI
    UI --> Hook
    Hook -->|"GET /chat/stream\nX-API-Key header"| Auth
    Auth --> Rate
    Rate --> Stream
    Stream --> Decomp
    Decomp -->|"3-5 sub-questions"| React
    React -->|"search query"| DDG
    React -->|"LLM call"| OpenAI
    React --> Synth
    Synth -->|"SSE chunks"| Hook
    Hook --> Parser
    Parser -->|"body + citation chips"| UI
    UI -->|"rendered answer"| User
    Stream --> CostLog
```

## Request lifecycle

1. User types a question → `useChatStream` opens a `fetch` SSE connection to `GET /chat/stream`
2. Auth middleware checks `X-API-Key` header → 401 if missing/invalid
3. Rate limiter checks sliding window per key → 429 if exceeded
4. `agent_runner.stream_agent` starts yielding SSE events:
   - `status` events for each pipeline stage (shown as status line in UI)
   - `chunk` events for each word group of the final answer (streamed token-by-token)
   - `done` event with `session_id` + `usage` when complete
5. On `done`, `parseCitations` splits `## Sources` into clickable chips
6. Cost is appended to `cost_log.jsonl`
