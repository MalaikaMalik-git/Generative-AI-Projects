import { useCallback, useRef, useState } from 'react'

/**
 * Push-to-talk speech recognition using the browser's Web Speech API.
 * Reliable in Chrome, Edge, and other Chromium browsers via the
 * vendor-prefixed `webkitSpeechRecognition`. Safari has partial/version-
 * dependent support (14.1+ on macOS, via the same prefix); Firefox does
 * not support it by default. `isSupported` reflects real-time feature
 * detection, and text input is always available as a fallback regardless.
 */
export function useSpeechRecognition({ onResult, onError } = {}) {
  const recognitionRef = useRef(null)
  const [isListening, setIsListening] = useState(false)

  const SpeechRecognitionCtor =
    typeof window !== 'undefined'
      ? window.SpeechRecognition || window.webkitSpeechRecognition
      : null
  const isSupported = !!SpeechRecognitionCtor

  const start = useCallback(() => {
    if (!isSupported || isListening) return

    const recognition = new SpeechRecognitionCtor()
    recognition.lang = 'en-US'
    recognition.interimResults = false
    recognition.maxAlternatives = 1
    recognition.continuous = false

    recognition.onresult = (event) => {
      const transcript = event.results?.[0]?.[0]?.transcript?.trim()
      if (transcript) onResult?.(transcript)
    }

    recognition.onerror = (event) => {
      // Common values: 'no-speech', 'audio-capture', 'not-allowed'
      onError?.(event.error)
    }

    recognition.onend = () => {
      setIsListening(false)
      recognitionRef.current = null
    }

    recognitionRef.current = recognition
    setIsListening(true)
    try {
      recognition.start()
    } catch {
      // start() throws if called while already started — safe to ignore
      setIsListening(false)
    }
  }, [SpeechRecognitionCtor, isSupported, isListening, onResult, onError])

  const stop = useCallback(() => {
    recognitionRef.current?.stop()
  }, [])

  return { start, stop, isListening, isSupported }
}
