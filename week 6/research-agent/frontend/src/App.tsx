// src/App.tsx
import { useState } from "react";
import { useChatStream } from "./hooks/useChatStream";
import { MessageList } from "./components/MessageList";
import { ChatInput } from "./components/ChatInput";
import { HistorySidebar } from "./components/HistorySidebar";
import "./App.css";

function App() {
  const {
    messages,
    isStreaming,
    sessionId,
    sessions,
    sendQuestion,
    clearConversation,
    loadSession,
  } = useChatStream();

  const [sidebarOpen, setSidebarOpen] = useState(false);

  const handleSelectSession = async (sid: string) => {
    await loadSession(sid);
    setSidebarOpen(false);
  };

  return (
    <div className="app-layout">
      {/* Sidebar overlay on mobile */}
      {sidebarOpen && (
        <div className="sidebar-overlay" onClick={() => setSidebarOpen(false)} />
      )}

      {/* History sidebar */}
      <div className={`sidebar-container ${sidebarOpen ? "sidebar-container--open" : ""}`}>
        <HistorySidebar
          sessions={sessions}
          currentSessionId={sessionId}
          onSelect={handleSelectSession}
          onNew={() => { clearConversation(); setSidebarOpen(false); }}
        />
      </div>

      {/* Main chat panel */}
      <div className="app-shell">
        <header className="app-header">
          <button
            className="sidebar-toggle"
            onClick={() => setSidebarOpen((o) => !o)}
            aria-label="Toggle history"
            title="Session history"
          >
            ☰
          </button>
          <div className="brand">
            <span className="brand-mark">⌁</span>
            <span className="brand-name">Research Agent</span>
          </div>
          <button
            className="clear-btn"
            onClick={clearConversation}
            disabled={messages.length === 0}
          >
            New session
          </button>
        </header>

        <main className="app-main">
          <MessageList messages={messages} />
        </main>

        <footer className="app-footer">
          <ChatInput onSend={sendQuestion} disabled={isStreaming} />
        </footer>
      </div>
    </div>
  );
}

export default App;
