import { useEffect, useRef, useState } from 'react'

/**
 * Camera capture flow: request permission -> show live preview ->
 * capture a single frame to a data URL -> stop the stream (release the
 * camera). Deliberately does NOT process continuous video, per the
 * project plan's own scope decision.
 */
function CameraCapture({ onCapture, onClose }) {
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const streamRef = useRef(null)
  const [error, setError] = useState(null)
  const [ready, setReady] = useState(false)

  useEffect(() => {
    let cancelled = false

    async function startCamera() {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: 'environment' },
          audio: false,
        })
        if (cancelled) {
          stream.getTracks().forEach((t) => t.stop())
          return
        }
        streamRef.current = stream
        if (videoRef.current) {
          videoRef.current.srcObject = stream
          await videoRef.current.play()
        }
        setReady(true)
      } catch (err) {
        if (err.name === 'NotAllowedError') {
          setError('Camera access was denied. Check your browser\'s site permissions.')
        } else if (err.name === 'NotFoundError') {
          setError('No camera was found on this device.')
        } else {
          setError(`Could not access the camera: ${err.message || err.name}`)
        }
      }
    }

    startCamera()

    return () => {
      cancelled = true
      streamRef.current?.getTracks().forEach((t) => t.stop())
    }
  }, [])

  const stopStream = () => {
    streamRef.current?.getTracks().forEach((t) => t.stop())
    streamRef.current = null
  }

  const handleCapture = () => {
    const video = videoRef.current
    const canvas = canvasRef.current
    if (!video || !canvas) return

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d')
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height)

    const dataUrl = canvas.toDataURL('image/jpeg', 0.85)

    stopStream() // release the camera immediately after capturing one frame
    onCapture(dataUrl)
  }

  const handleClose = () => {
    stopStream()
    onClose()
  }

  return (
    <div className="camera-overlay">
      <div className="camera-box">
        {error ? (
          <div className="camera-error">
            <p>{error}</p>
            <button onClick={handleClose}>Close</button>
          </div>
        ) : (
          <>
            <video ref={videoRef} className="camera-video" playsInline muted />
            {!ready && <p className="camera-status">Requesting camera access...</p>}
            <div className="camera-controls">
              <button className="camera-cancel" onClick={handleClose}>
                Cancel
              </button>
              <button className="camera-capture" onClick={handleCapture} disabled={!ready}>
                Capture
              </button>
            </div>
          </>
        )}
        <canvas ref={canvasRef} style={{ display: 'none' }} />
      </div>
    </div>
  )
}

export default CameraCapture
