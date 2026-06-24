// src/hooks/useChatStream.ts

import { useCallback, useRef, useState } from "react";
import { API_BASE_URL, API_KEY } from "../lib/config";
import type { ChatMessage, SSEEvent, SessionSummary } from "../types/api";
import { parseCitations } from "../lib/parseCitations";

function newId(): string {
  return Date.now().toString(36) + Math.random().toString(36).slice(2, 8);
}

function nowISO(): string {
  return new Date().toISOString();
}

interface UseChatStreamResult {
  messages: ChatMessage[];
  isStreaming: boolean;
  sessionId: string | null;
  sessions: SessionSummary[];
  sendQuestion: (question: string) => void;
  clearConversation: () => void;
  loadSession: (sessionId: string) => Promise<void>;
}

export function useChatStream(): UseChatStreamResult {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const sessionIdRef = useRef<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const updateAssistantMessage = useCallback(
    (id: string, updater: (msg: ChatMessage) => ChatMessage) => {
      setMessages((prev) => prev.map((m) => (m.id === id ? updater(m) : m)));
    },
    []
  );

  const upsertSession = useCallback(
    (sessionId: string, question: string, timestamp: string) => {
      setSessions((prev) => {
        const existing = prev.find((s) => s.sessionId === sessionId);
        if (existing) {
          return prev.map((s) =>
            s.sessionId === sessionId
              ? { ...s, entryCount: s.entryCount + 1 }
              : s
          );
        }
        return [
          {
            sessionId,
            firstQuestion: question,
            timestamp,
            entryCount: 1,
          },
          ...prev,
        ];
      });
    },
    []
  );

  const sendQuestion = useCallback(
    (question: string) => {
      const trimmed = question.trim();
      if (!trimmed || isStreaming) return;

      const userMsg: ChatMessage = {
        id: newId(),
        role: "user",
        content: trimmed,
        rawContent: trimmed,
        citations: [],
        status: "done",
        timestamp: nowISO(),
      };

      const assistantId = newId();
      const assistantMsg: ChatMessage = {
        id: assistantId,
        role: "assistant",
        content: "",
        rawContent: "",
        citations: [],
        status: "streaming",
        statusLine: "Connecting...",
        timestamp: nowISO(),
      };

      setMessages((prev) => [...prev, userMsg, assistantMsg]);
      setIsStreaming(true);

      const params = new URLSearchParams({ question: trimmed });
      if (sessionIdRef.current) {
        params.set("session_id", sessionIdRef.current);
      }

      const controller = new AbortController();
      abortRef.current = controller;

      // Accumulate raw content so we can parse citations at "done"
      let accumulated = "";

      fetch(`${API_BASE_URL}/chat/stream?${params.toString()}`, {
        method: "GET",
        headers: { "X-API-Key": API_KEY },
        signal: controller.signal,
      })
        .then(async (res) => {
          if (!res.ok) {
            const status = res.status;
            let detail = `HTTP ${status}`;
            try {
              const body = await res.json();
              detail = body.detail || detail;
            } catch { /* ignore */ }
            throw new HttpError(status, detail);
          }
          if (!res.body) throw new Error("No response body");

          const reader = res.body.getReader();
          const decoder = new TextDecoder();
          let buffer = "";

          while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });
            const events = buffer.split("\n\n");
            buffer = events.pop() ?? "";

            for (const raw of events) {
              const line = raw.trim();
              if (!line.startsWith("data:")) continue;
              const jsonStr = line.slice(5).trim();
              if (!jsonStr) continue;

              let parsed: SSEEvent;
              try { parsed = JSON.parse(jsonStr); } catch { continue; }

              if (parsed.type === "status") {
                updateAssistantMessage(assistantId, (m) => ({
                  ...m,
                  statusLine: parsed.content,
                }));
              } else if (parsed.type === "chunk") {
                accumulated += parsed.content;
                updateAssistantMessage(assistantId, (m) => ({
                  ...m,
                  content: m.content + parsed.content,
                  rawContent: m.rawContent + parsed.content,
                  statusLine: undefined,
                }));
              } else if (parsed.type === "done") {
                if (parsed.session_id) {
                  sessionIdRef.current = parsed.session_id;
                  upsertSession(parsed.session_id, trimmed, nowISO());
                }
                // Parse citations from the fully accumulated text
                const { body, citations } = parseCitations(accumulated);
                updateAssistantMessage(assistantId, (m) => ({
                  ...m,
                  status: parsed.success ? "done" : "error",
                  content: body,           // body without ## Sources
                  rawContent: accumulated, // keep original
                  citations,
                  sessionId: parsed.session_id,
                  usage: parsed.usage,
                  statusLine: undefined,
                  errorMessage: parsed.success ? undefined : "The research pipeline failed.",
                }));
              } else if (parsed.type === "error") {
                updateAssistantMessage(assistantId, (m) => ({
                  ...m,
                  status: "error",
                  errorMessage: parsed.content,
                  statusLine: undefined,
                }));
              }
            }
          }
        })
        .catch((err: unknown) => {
          if (controller.signal.aborted) return;
          const message =
            err instanceof HttpError
              ? friendlyErrorMessage(err.status, err.detail)
              : err instanceof Error
              ? err.message
              : "Unknown error";

          updateAssistantMessage(assistantId, (m) => ({
            ...m,
            status: "error",
            errorMessage: message,
            statusLine: undefined,
          }));
        })
        .finally(() => {
          setIsStreaming(false);
        });
    },
    [isStreaming, updateAssistantMessage, upsertSession]
  );

  const loadSession = useCallback(async (sid: string) => {
    try {
      const res = await fetch(`${API_BASE_URL}/history/${sid}`, {
        headers: { "X-API-Key": API_KEY },
      });
      if (!res.ok) return;
      const data = await res.json();

      const loaded: ChatMessage[] = [];
      for (const entry of data.entries) {
        loaded.push({
          id: newId(),
          role: "user",
          content: entry.question,
          rawContent: entry.question,
          citations: [],
          status: "done",
          timestamp: entry.timestamp,
        });
        const { body, citations } = parseCitations(entry.answer);
        loaded.push({
          id: newId(),
          role: "assistant",
          content: body,
          rawContent: entry.answer,
          citations,
          status: entry.success ? "done" : "error",
          sessionId: sid,
          timestamp: entry.timestamp,
        });
      }

      sessionIdRef.current = sid;
      setMessages(loaded);
    } catch { /* silent */ }
  }, []);

  const clearConversation = useCallback(() => {
    abortRef.current?.abort();
    setMessages([]);
    setIsStreaming(false);
    sessionIdRef.current = null;
  }, []);

  return {
    messages,
    isStreaming,
    sessionId: sessionIdRef.current,
    sessions,
    sendQuestion,
    clearConversation,
    loadSession,
  };
}

class HttpError extends Error {
  status: number;
  detail: string;
  constructor(status: number, detail: string) {
    super(detail);
    this.status = status;
    this.detail = detail;
  }
}

function friendlyErrorMessage(status: number, detail: string): string {
  if (status === 401) return "Authentication failed — check your API key in .env.local.";
  if (status === 429) return detail || "Rate limit hit. Wait a moment and try again.";
  if (status >= 500) return "Server error. Make sure the backend is running.";
  return detail || `Request failed (${status}).`;
}
