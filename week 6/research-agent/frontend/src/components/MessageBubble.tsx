// src/components/MessageBubble.tsx
import type { ChatMessage } from "../types/api";
import { CitationChips } from "./CitationChips";
import { MessageSkeleton } from "./MessageSkeleton";

interface Props {
  message: ChatMessage;
}

export function MessageBubble({ message }: Props) {
  const isUser = message.role === "user";
  const isWaitingForFirstToken =
    message.status === "streaming" && !message.content;

  return (
    <div className={`msg-row ${isUser ? "msg-row--user" : "msg-row--assistant"}`}>
      <div className="msg-meta">
        <span className="msg-role">{isUser ? "YOU" : "AGENT"}</span>
      </div>

      <div className={`msg-bubble ${isUser ? "msg-bubble--user" : "msg-bubble--assistant"}`}>
        {/* Status line — shown while agent stages run */}
        {message.status === "streaming" && message.statusLine && (
          <div className="status-line">
            <span className="status-dot" />
            {message.statusLine}
          </div>
        )}

        {/* Loading skeleton — before first token */}
        {isWaitingForFirstToken && <MessageSkeleton />}

        {/* Answer body */}
        {message.content && (
          <p className="msg-text">
            {message.content}
            {message.status === "streaming" && <span className="cursor" />}
          </p>
        )}

        {/* Citation chips */}
        {message.status === "done" && (
          <CitationChips citations={message.citations} />
        )}

        {/* Error state */}
        {message.status === "error" && (
          <div className="error-box">
            <span className="error-icon">!</span>
            <span>{message.errorMessage ?? "Something went wrong."}</span>
          </div>
        )}

        {/* Token usage */}
        {message.status === "done" && message.usage && (
          <div className="usage-line">
            {message.usage.total_tokens.toLocaleString()} tokens · $
            {message.usage.estimated_cost_usd.toFixed(6)}
          </div>
        )}
      </div>
    </div>
  );
}
