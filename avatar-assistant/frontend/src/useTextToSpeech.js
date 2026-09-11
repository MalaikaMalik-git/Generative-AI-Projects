import { useCallback, useState } from 'react'

/**
 * Browser text-to-speech via the Web Speech Synthesis API. Widely
 * supported (Chrome, Edge, Safari, Firefox). `isSpeaking` is meant to be
 * fed straight into the avatar's speaking animation.
 */
export function useTextToSpeech() {
  const [isSpeaking, setIsSpeaking] = useState(false)
  const isSupported = typeof window !== 'undefined' && 'speechSynthesis' in window

  const speak = useCallback(
    (text) => {
      if (!isSupported || !text) return

      // Cancel anything currently playing so responses don't overlap.
      window.speechSynthesis.cancel()

      const utterance = new SpeechSynthesisUtterance(text)
      utterance.rate = 1
      utterance.pitch = 1
      utterance.onstart = () => setIsSpeaking(true)
      utterance.onend = () => setIsSpeaking(false)
      utterance.onerror = () => setIsSpeaking(false)

      window.speechSynthesis.speak(utterance)
    },
    [isSupported]
  )

  const cancel = useCallback(() => {
    if (!isSupported) return
    window.speechSynthesis.cancel()
    setIsSpeaking(false)
  }, [isSupported])

  return { speak, cancel, isSpeaking, isSupported }
}
