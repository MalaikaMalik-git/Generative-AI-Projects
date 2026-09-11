import { useEffect, useRef, useState } from 'react'
import { useSpeechRecognition } from './useSpeechRecognition.js'
import CameraCapture from './CameraCapture.jsx'

// Configurable so the same build works locally and once deployed — set
// VITE_BACKEND_URL in a .env file (see .env.example) for production.
const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || 'http://localhost:8000'
const DEFAULT_VISION_QUESTION = 'What do you see in this image?'

function MicIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
      <rect x="9" y="2" width="6" height="12" rx="3" />
      <path d="M5 11a7 7 0 0 0 14 0" strokeLinecap="round" />
      <path d="M12 18v3" strokeLinecap="round" />
    </svg>
  )
}

function CameraIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
      <path d="M4 8h3l1.5-2h7L17 8h3a1 1 0 0 1 1 1v9a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V9a1 1 0 0 1 1-1Z" />
      <circle cx="12" cy="13.5" r="3.3" />
    </svg>
  )
}

function SendIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
      <path d="M4 12h15M13 6l6 6-6 6" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  )
}

function AlertIcon() {
  return (
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
      <path d="M12 3 2 20h20L12 3Z" strokeLinejoin="round" />
      <path d="M12 10v4" strokeLinecap="round" />
      <circle cx="12" cy="17" r="0.5" fill="currentColor" />
    </svg>
  )
}

function ChatPanel({ onAssistantReply, isAvatarSpeaking }) {
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      text: "Hi! Ask me anything about Agentix System — what they do, their AI agents, or how to reach them. You can type, hold the mic to talk, or use the camera to show me something.",
      sources: [],
    },
  ])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const bottomRef = useRef(null)

  // --- Camera / vision state ---
  const [cameraOpen, setCameraOpen] = useState(false)
  const [capturedImage, setCapturedImage] = useState(null) // data URL, awaiting a question
  const [visionQuestion, setVisionQuestion] = useState(DEFAULT_VISION_QUESTION)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading, capturedImage])

  const submitQuestion = async (question) => {
    const trimmed = question.trim()
    if (!trimmed || loading) return

    setMessages((prev) => [...prev, { role: 'user', text: trimmed }])
    setInput('')
    setLoading(true)
    setError(null)

    try {
      const res = await fetch(`${BACKEND_URL}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: trimmed }),
      })

      const data = await res.json()
      if (!res.ok) throw new Error(data.detail || `Request failed (${res.status})`)

      setMessages((prev) => [
        ...prev,
        { role: 'assistant', text: data.answer, sources: data.sources || [] },
      ])
      onAssistantReply?.(data.answer)
    } catch (err) {
      setError(err.message || 'Something went wrong talking to the backend.')
    } finally {
      setLoading(false)
    }
  }

  const submitVisionQuestion = async (imageDataUrl, question) => {
    const trimmedQuestion = (question || DEFAULT_VISION_QUESTION).trim()
    if (loading) return

    setMessages((prev) => [
      ...prev,
      { role: 'user', text: trimmedQuestion, image: imageDataUrl },
    ])
    setCapturedImage(null)
    setVisionQuestion(DEFAULT_VISION_QUESTION)
    setLoading(true)
    setError(null)

    try {
      const res = await fetch(`${BACKEND_URL}/vision`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: imageDataUrl, question: trimmedQuestion }),
      })

      const data = await res.json()
      if (!res.ok) throw new Error(data.detail || `Request failed (${res.status})`)

      setMessages((prev) => [
        ...prev,
        { role: 'assistant', text: data.answer, sources: [] },
      ])
      onAssistantReply?.(data.answer)
    } catch (err) {
      setError(err.message || 'Something went wrong analyzing the image.')
    } finally {
      setLoading(false)
    }
  }

  const handleFormSubmit = (e) => {
    e.preventDefault()
    submitQuestion(input)
  }

  // --- Voice input (push-to-talk) ---
  const { start, stop, isListening, isSupported: micSupported } = useSpeechRecognition({
    onResult: (transcript) => {
      setInput(transcript)
      submitQuestion(transcript)
    },
    onError: (err) => {
      if (err === 'no-speech') {
        setError("Didn't catch that — try holding the mic button a little longer.")
      } else if (err === 'not-allowed') {
        setError("Microphone access was denied. Check your browser's site permissions.")
      } else {
        setError(`Voice input error: ${err}`)
      }
    },
  })

  const inputDisabled = loading || isAvatarSpeaking
  const micDisabled = inputDisabled

  const handleMicDown = (e) => {
    e.preventDefault()
    if (micDisabled) return
    start()
  }
  const handleMicUp = (e) => {
    e.preventDefault()
    if (isListening) stop()
  }

  return (
    <div className="chat-panel">
      <div className="chat-messages">
        {messages.map((msg, i) => (
          <div key={i} className={`chat-bubble ${msg.role}`}>
            {msg.image && (
              <img src={msg.image} alt="Captured from camera" className="chat-image" />
            )}
            <p>{msg.text}</p>
            {msg.sources && msg.sources.length > 0 && (
              <div className="chat-sources">Source: {msg.sources.join(', ')}</div>
            )}
          </div>
        ))}

        {loading && (
          <div className="chat-bubble assistant loading">
            <span className="dot" />
            <span className="dot" />
            <span className="dot" />
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      {error && (
        <div className="chat-error">
          <span className="chat-error-text">
            <AlertIcon /> {error}
          </span>
          <button onClick={() => setError(null)}>Dismiss</button>
        </div>
      )}

      {capturedImage && (
        <div className="vision-review">
          <img src={capturedImage} alt="Captured preview" className="vision-review-thumb" />
          <input
            type="text"
            value={visionQuestion}
            onChange={(e) => setVisionQuestion(e.target.value)}
            placeholder="What do you want to know about this?"
          />
          <div className="vision-review-actions">
            <button
              className="vision-retake"
              onClick={() => {
                setCapturedImage(null)
                setCameraOpen(true)
              }}
            >
              Retake
            </button>
            <button
              className="vision-send"
              onClick={() => submitVisionQuestion(capturedImage, visionQuestion)}
              disabled={loading}
            >
              Ask about this
            </button>
          </div>
        </div>
      )}

      <form className="chat-input-row" onSubmit={handleFormSubmit}>
        {micSupported && (
          <button
            type="button"
            className={`mic-button ${isListening ? 'listening' : ''}`}
            onMouseDown={handleMicDown}
            onMouseUp={handleMicUp}
            onMouseLeave={handleMicUp}
            onTouchStart={handleMicDown}
            onTouchEnd={handleMicUp}
            disabled={micDisabled}
            title={isAvatarSpeaking ? 'Wait for the avatar to finish speaking' : 'Hold to talk'}
          >
            <MicIcon />
          </button>
        )}

        <button
          type="button"
          className="camera-button"
          onClick={() => setCameraOpen(true)}
          disabled={inputDisabled || !!capturedImage}
          title="Show me something"
        >
          <CameraIcon />
        </button>

        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder={isListening ? 'Listening…' : 'Ask about Agentix System…'}
          disabled={inputDisabled}
        />
        <button type="submit" className="send-button" disabled={inputDisabled || !input.trim()}>
          {loading ? 'Sending…' : (
            <>
              Send <SendIcon />
            </>
          )}
        </button>
      </form>

      {!micSupported && (
        <div className="chat-note">
          Voice input isn't supported in this browser — Chrome, Edge, or Safari work best.
          Text input above always works.
        </div>
      )}

      {cameraOpen && (
        <CameraCapture
          onCapture={(dataUrl) => {
            setCapturedImage(dataUrl)
            setCameraOpen(false)
          }}
          onClose={() => setCameraOpen(false)}
        />
      )}
    </div>
  )
}

export default ChatPanel
