// src/components/ChatInput.tsx
import { useState, type KeyboardEvent } from "react";

interface Props {
  onSend: (question: string) => void;
  disabled: boolean;
}

export function ChatInput({ onSend, disabled }: Props) {
  const [value, setValue] = useState("");

  const submit = () => {
    if (!value.trim() || disabled) return;
    onSend(value);
    setValue("");
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  };

  return (
    <div className="chat-input-bar">
      <textarea
        className="chat-input"
        placeholder="Ask a research question..."
        value={value}
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={handleKeyDown}
        disabled={disabled}
        rows={1}
        maxLength={1000}
      />
      <button
        className="send-btn"
        onClick={submit}
        disabled={disabled || !value.trim()}
        aria-label="Send"
      >
        {disabled ? "···" : "→"}
      </button>
    </div>
  );
}
