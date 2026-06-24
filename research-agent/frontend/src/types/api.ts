// src/types/api.ts

export interface ChatRequest {
  question: string;
  session_id?: string;
}

export interface TokenUsage {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  estimated_cost_usd: number;
}

export interface ChatResponse {
  session_id: string;
  question: string;
  answer: string;
  success: boolean;
  error?: string;
  usage?: TokenUsage;
  timestamp: string;
}

export interface HistoryEntry {
  session_id: string;
  question: string;
  answer: string;
  success: boolean;
  timestamp: string;
}

export interface HistoryResponse {
  session_id: string;
  entries: HistoryEntry[];
  total: number;
}

// ── SSE event shapes ──────────────────────────────────────────────────────────

export type SSEEventType = "status" | "chunk" | "done" | "error";

export interface SSEStatusEvent {
  type: "status";
  content: string;
}

export interface SSEChunkEvent {
  type: "chunk";
  content: string;
}

export interface SSEDoneEvent {
  type: "done";
  session_id: string;
  success: boolean;
  usage?: TokenUsage;
}

export interface SSEErrorEvent {
  type: "error";
  content: string;
}

export type SSEEvent = SSEStatusEvent | SSEChunkEvent | SSEDoneEvent | SSEErrorEvent;

// ── Citation chip ─────────────────────────────────────────────────────────────

export interface Citation {
  index: number;
  url: string;
  domain: string;
}

// ── Chat message ─────────────────────────────────────────────────────────────

export type MessageRole = "user" | "assistant";
export type MessageStatus = "streaming" | "done" | "error";

export interface ChatMessage {
  id: string;
  role: MessageRole;
  content: string;          // answer body (without ## Sources)
  rawContent: string;       // full original text including Sources
  citations: Citation[];    // parsed from ## Sources
  status: MessageStatus;
  statusLine?: string;
  sessionId?: string;
  usage?: TokenUsage;
  errorMessage?: string;
  timestamp: string;
}

// ── Session history sidebar ───────────────────────────────────────────────────

export interface SessionSummary {
  sessionId: string;
  firstQuestion: string;
  timestamp: string;
  entryCount: number;
}