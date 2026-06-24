# Week 6 Demo Script — Research Agent
**Total time: 25 minutes**

---

## Part 1 — What the agent does and how the ReAct loop works (3 min)

> Open your slides. Talk through this, don't read it word for word.

**What the agent does (30 sec)**

"This is a Research Agent. You give it any question, and it:
1. Breaks it into sub-questions using an LLM
2. Searches the web for each one using the ReAct pattern
3. Synthesizes all the answers into a structured, cited report"

**How the ReAct loop works (2 min)**

Draw or point to this flow on your slides:

```
Question → Thought → Action (web_search) → Observation → Thought → ... → Final Answer
```

"ReAct stands for Reasoning + Acting. The agent alternates between:
- **Thought** — deciding what to do next
- **Action** — calling a tool (in our case, web search)
- **Observation** — reading what the tool returned

It repeats this loop until it has enough information to write a Final Answer.
The key insight is that the LLM is not just generating text — it's making decisions
about which tools to call and when to stop."

**Tools available (30 sec)**

"The agent has two tools:
- `web_search` — queries DuckDuckGo, returns titles, URLs and snippets
- `fetch_page` — fetches full text of a page (used as a fallback)

No API key needed for search — we use the open DuckDuckGo library."

---

## Part 2 — Live run on a simple question with trace visible (8 min)

> Switch to your terminal. Make the font size big (Cmd + = a few times).

**Run this command:**
```bash
python3 main.py "What is the James Webb Space Telescope?"
```

**While it runs, narrate each step out loud:**

When you see **Step 1 — Decomposing question**:
> "First the agent decomposes the question. Watch — it breaks one question into 5 focused sub-questions. This is important because each sub-question becomes a targeted search query."

When you see the first **💭 Thought** panel:
> "This is the Thought step. The LLM is deciding what to search for."

When you see the first **⚡ Action** panel:
> "This is the Action — it's calling web_search with a specific query."

When you see the first **🔍 Observation** panel:
> "This is the Observation — real search results coming back from DuckDuckGo. No mock data, this is live."

When you see the first **✅ Answer** panel:
> "The agent decided it has enough information and wrote its Final Answer for this sub-question. Notice it took just 2 steps — search once, answer immediately. That's efficient."

When the **Research Report** appears:
> "Finally the synthesizer combines all 5 sub-answers into one structured report —
> Summary, Key Findings, Detailed Analysis, and a numbered source list.
> This is production-quality output from a single command."

---

## Part 3 — Live run on a complex question showing multi-step reasoning (8 min)

> Keep terminal open. Run this command:

```bash
python3 main.py "Compare the pros and cons of electric vehicles vs petrol cars for long distance travel"
```

**Key things to highlight while it runs:**

When decomposition appears:
> "Notice the decomposition is smarter here — it generated 5 distinct sub-questions covering pros, cons, and a specific comparison angle. That's multi-faceted reasoning."

When a sub-question takes 3+ steps:
> "Watch this sub-question — it's taking more steps. The first search returned poor results, so the agent automatically tried a different query. That's the ReAct loop doing its job — reasoning about what went wrong and adapting."

When the report appears:
> "9 sources cited. The agent found information across multiple domains — EV reviews, environmental studies, consumer reports — and synthesized them into a coherent comparison. That's the power of multi-step reasoning."

**Compare to Part 2:**
> "The simple question took 2 steps per sub-question. The complex one took up to 5. The agent scaled its effort to the difficulty of the question automatically."

---

## Part 4 — One interesting failure case and how it was handled (6 min)

> Keep terminal open. Explain before running:

"Every real system needs to handle failures gracefully. Let me show you the most interesting failure case we encountered and how we handle it."

**Explain the failure (1 min)**

"DuckDuckGo sometimes returns an SSL protocol error — `Unsupported protocol version 0x304`. This happens randomly due to rate limiting. In a naive implementation, this would crash the entire agent."

**Show the error handling in code (1 min)**

Open `agent/error_handler.py` in VS Code and point to `safe_search()`:
> "Our `safe_search` function wraps every search call. When it detects a failure:
> 1. It logs a warning in yellow
> 2. Simplifies the query by removing the last word
> 3. Retries with a delay
> If all retries fail, it returns a graceful 'Search unavailable' message instead of crashing."

**Show it live (2 min)**

Run the EV question again — the SSL errors are visible in the output:
```bash
python3 main.py "Compare the pros and cons of electric vehicles vs petrol cars for long distance travel"
```

Point to the yellow warning lines:
> "See these yellow warnings? That's the error handler catching a live SSL failure, retrying automatically, and recovering. The agent kept running — the user got a complete report despite the errors."

**Show the test (1 min)**

```bash
python3 -m pytest tests/test_error_handler.py -v
```

> "We have 8 tests covering every failure case — search failures, network errors, max steps exceeded, and synthesizer failures. All mocked, all passing. 38 tests total in the project."

**Close (1 min)**

> "The 4 failure cases we handle are:
> 1. Search returns no results → auto-retry with simplified query
> 2. SSL/network error → retry with delay, graceful fallback
> 3. Max steps exceeded → return partial answer from observations
> 4. Synthesizer fails → format raw answers as fallback report
>
> None of these crash the agent. The user always gets an answer."

---

## Backup commands (in case something goes wrong)

```bash
# If DuckDuckGo is completely down — run tests to show the logic still works
python3 -m pytest tests/ -v

# Alternative simple question that works well
python3 main.py "What is Python programming language?"

# Alternative complex question
python3 main.py "What are the causes and effects of climate change?"

# Show the architecture quickly
cat README.md
```

---

## Questions you might get

**"Why DuckDuckGo and not Google?"**
> "DuckDuckGo has a free Python library with no API key needed — perfect for a demo project. In production you'd swap in the Google Custom Search API or Serper.dev with one line change in `tools/search.py`."

**"Why gpt-4o-mini and not gpt-4o?"**
> "Cost. gpt-4o-mini is 15x cheaper and fast enough for development. You'd switch to gpt-4o for a production deployment — it's one env variable change."

**"Can it handle questions where the answer keeps changing?"**
> "Yes — every run does fresh searches, so the agent always gets current results. It's not cached."

**"How many API calls does one question make?"**
> "Roughly 8-12 OpenAI calls per full run — one for decomposition, one or two per sub-question, one for synthesis. At gpt-4o-mini pricing that's a fraction of a cent per question."

**"What would you add if you had more time?"**
> "A Gradio UI so non-technical users can use it from a browser, persistent caching to avoid re-searching the same queries, and streaming output so the report appears word by word instead of all at once."