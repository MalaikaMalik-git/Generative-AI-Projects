// src/components/HistorySidebar.tsx
import type { SessionSummary } from "../types/api";

interface Props {
  sessions: SessionSummary[];
  currentSessionId: string | null;
  onSelect: (sessionId: string) => void;
  onNew: () => void;
}

function relativeTime(iso: string): string {
  const diff = Date.now() - new Date(iso).getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  return `${Math.floor(hrs / 24)}d ago`;
}

export function HistorySidebar({ sessions, currentSessionId, onSelect, onNew }: Props) {
  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <span className="sidebar-title">Sessions</span>
        <button className="new-session-btn" onClick={onNew}>
          + New
        </button>
      </div>

      {sessions.length === 0 ? (
        <p className="sidebar-empty">No sessions yet.</p>
      ) : (
        <ul className="session-list">
          {sessions.map((s) => (
            <li
              key={s.sessionId}
              className={`session-item ${
                s.sessionId === currentSessionId ? "session-item--active" : ""
              }`}
              onClick={() => onSelect(s.sessionId)}
            >
              <span className="session-q">{s.firstQuestion}</span>
              <span className="session-meta">
                {relativeTime(s.timestamp)} · {s.entryCount}{" "}
                {s.entryCount === 1 ? "msg" : "msgs"}
              </span>
            </li>
          ))}
        </ul>
      )}
    </aside>
  );
}
