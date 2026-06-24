#!/usr/bin/env python3
"""
scripts/test_stream.py

Hits GET /chat/stream and pretty-prints each SSE event live in the terminal.
Run with:
    python3 scripts/test_stream.py
    python3 scripts/test_stream.py "Your question here"

Requires: requests  (already in requirements.txt)
Server must be running: python3 -m uvicorn api.main:app --reload
"""
import sys
import json
import requests

BASE_URL = "http://localhost:8000"
API_KEY  = "dev-key-1234"          # must match ALLOWED_API_KEYS in .env

RESET  = "\033[0m"
BOLD   = "\033[1m"
CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
DIM    = "\033[2m"


def stream(question: str) -> None:
    print(f"\n{BOLD}{'─'*60}{RESET}")
    print(f"{BOLD}Question:{RESET} {question}")
    print(f"{BOLD}{'─'*60}{RESET}\n")

    url = f"{BASE_URL}/chat/stream"
    params = {"question": question}
    headers = {"X-API-Key": API_KEY}

    answer_buf = []

    try:
        with requests.get(url, params=params, headers=headers, stream=True, timeout=120) as r:
            r.raise_for_status()
            for raw_line in r.iter_lines():
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8")
                if not line.startswith("data: "):
                    continue

                try:
                    event = json.loads(line[6:])
                except json.JSONDecodeError:
                    continue

                etype = event.get("type")

                if etype == "status":
                    print(f"{CYAN}{event['content']}{RESET}")

                elif etype == "chunk":
                    text = event["content"]
                    print(text, end="", flush=True)
                    answer_buf.append(text)

                elif etype == "done":
                    print(f"\n\n{BOLD}{'─'*60}{RESET}")
                    if event.get("success"):
                        usage = event.get("usage", {})
                        print(f"{GREEN}✅ Done  |  session: {event['session_id'][:8]}...{RESET}")
                        print(
                            f"{DIM}tokens: {usage.get('total_tokens', '?')}  "
                            f"cost: ${usage.get('estimated_cost_usd', 0):.6f}{RESET}"
                        )
                    else:
                        print(f"{RED}❌ Pipeline failed{RESET}")

                elif etype == "error":
                    print(f"\n{RED}Error: {event['content']}{RESET}")

    except requests.exceptions.ConnectionError:
        print(f"{RED}Cannot connect to {BASE_URL}. Is the server running?{RESET}")
        print(f"  Start it with:  python3 -m uvicorn api.main:app --reload")
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            print(f"{RED}401 Unauthorized — check API_KEY in this script matches your .env{RESET}")
        elif e.response.status_code == 429:
            print(f"{YELLOW}429 Rate limited — too many requests{RESET}")
        else:
            print(f"{RED}HTTP {e.response.status_code}: {e.response.text}{RESET}")
        sys.exit(1)


def test_auth_rejection() -> None:
    """Verify that a bad API key gets a 401."""
    print(f"\n{BOLD}[Auth Test] Bad key should return 401...{RESET}", end=" ")
    try:
        r = requests.get(
            f"{BASE_URL}/chat/stream",
            params={"question": "test"},
            headers={"X-API-Key": "wrong-key-xyz"},
            stream=True,
            timeout=5,
        )
        if r.status_code == 401:
            print(f"{GREEN}✅ 401 received — auth works{RESET}")
        else:
            print(f"{RED}❌ Expected 401, got {r.status_code}{RESET}")
    except Exception as e:
        print(f"{RED}❌ {e}{RESET}")


def test_rate_limit() -> None:
    """
    Fires 11 rapid requests (limit is 10/hr) and checks the 11th gets 429.
    Uses GET /chat/stream with stream=True so FastAPI evaluates the auth and
    rate-limit dependencies and returns the 429 status header *before* the
    agent generator runs — response arrives in milliseconds, no OpenAI calls.
    We read only the status code then immediately close the connection.
    """
    print(f"\n{BOLD}[Rate Limit Test] Sending 11 rapid requests...{RESET}")
    headers = {"X-API-Key": "rate-test-key-999"}
    hit_429 = False
    for i in range(11):
        try:
            r = requests.get(
                f"{BASE_URL}/chat/stream",
                params={"question": "rate-limit-probe"},
                headers=headers,
                stream=True,     # grab status line only; don't buffer the body
                timeout=(3, 3),  # (connect_timeout, read_timeout)
            )
            status = r.status_code
            r.close()            # drop immediately — don't run the agent
        except requests.exceptions.ReadTimeout:
            # Timed out reading body — means we passed auth+rate-limit and the
            # SSE stream started (server is working); count as 200.
            status = 200
        if status == 429:
            print(f"  Request {i+1}: {GREEN}✅ 429 Rate Limited — works!{RESET}")
            hit_429 = True
            break
        else:
            print(f"  Request {i+1}: {DIM}{status}{RESET}")
    if not hit_429:
        print(f"{RED}  ❌ Never hit 429 — rate limiter may not be working{RESET}")


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "What is the James Webb Space Telescope?"

    # Run auth + rate limit checks first (no OpenAI calls)
    test_auth_rejection()
    test_rate_limit()

    # Then do a real stream
    print(f"\n{BOLD}[Stream Test] Running full research pipeline...{RESET}")
    stream(question)