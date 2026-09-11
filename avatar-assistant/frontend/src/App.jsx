import { useEffect, useState } from 'react'
import AvatarViewer from './AvatarViewer.jsx'
import ChatPanel from './ChatPanel.jsx'
import { useTextToSpeech } from './useTextToSpeech.js'
import './theme.css'
import './App.css'
import './ChatPanel.css'

// Configurable so the same build works locally and once deployed — set
// VITE_BACKEND_URL in a .env file (see .env.example) for production.
const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || 'http://localhost:8000'

const STATUS_LABEL = {
  connected: 'Backend connected',
  error: 'Backend unreachable',
  checking: 'Checking backend…',
}

function App() {
  const [status, setStatus] = useState('checking') // checking | connected | error
  const { speak, isSpeaking, isSupported: ttsSupported } = useTextToSpeech()

  const checkHealth = () => {
    setStatus('checking')
    fetch(`${BACKEND_URL}/health`)
      .then((res) => res.json())
      .then(() => setStatus('connected'))
      .catch(() => setStatus('error'))
  }

  useEffect(() => {
    checkHealth()
  }, [])

  return (
    <div className="console">
      <div className="console-frame">
        <div className="console-header">
          <div className="console-brand">
            <span className="mark">Agentix</span>
            <span className="sub">Avatar Assistant</span>
          </div>
          <button
            type="button"
            className={`status-pill ${status}`}
            onClick={checkHealth}
            title="Click to re-check"
          >
            <span className="status-dot" />
            {STATUS_LABEL[status]}
          </button>
        </div>

        <div className="console-body">
          <div className="avatar-col">
            <div className="avatar-frame">
              <div className="avatar-canvas-wrap">
                <AvatarViewer isSpeaking={isSpeaking} />
              </div>
            </div>
            {!ttsSupported && (
              <div className="tts-note">
                Text-to-speech isn't supported in this browser — answers will show as text only.
              </div>
            )}
          </div>

          <div className="chat-col">
            <ChatPanel onAssistantReply={speak} isAvatarSpeaking={isSpeaking} />
          </div>
        </div>
      </div>
    </div>
  )
}

export default App
