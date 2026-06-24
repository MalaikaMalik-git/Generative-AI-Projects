// src/components/MessageList.tsx
import { useEffect, useRef } from "react";
import type { ChatMessage } from "../types/api";
import { MessageBubble } from "./MessageBubble";

interface Props {
  messages: ChatMessage[];
}

export function MessageList({ messages }: Props) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages]);

  if (messages.length === 0) {
    return (
      <div className="empty-state">
        <div className="empty-mark">⌁</div>
        <p>Ask a research question to get started.</p>
        <p className="empty-sub">
          The agent decomposes it, researches each part, and synthesizes a report — live.
        </p>
      </div>
    );
  }

  return (
    <div className="message-list">
      {messages.map((m) => (
        <MessageBubble key={m.id} message={m} />
      ))}
      <div ref={bottomRef} />
    </div>
  );
}
