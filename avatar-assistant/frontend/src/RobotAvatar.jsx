import { useEffect, useRef } from 'react'
import { useGLTF, useAnimations } from '@react-three/drei'
import { useFrame } from '@react-three/fiber'

const MODEL_URL = '/models/RobotExpressive.glb'

// Animation clip to play while the avatar is "talking." The model has no
// dedicated talk-cycle animation, so Wave reads as an active, responsive
// gesture without looking like it's about to walk off-screen.
const SPEAKING_CLIP = 'Wave'
const IDLE_CLIP = 'Idle'

function RobotAvatar({ isSpeaking }) {
  const group = useRef()
  const headRef = useRef()
  const blinkTimer = useRef(0)
  const nextBlinkAt = useRef(2 + Math.random() * 3)

  const { scene, animations } = useGLTF(MODEL_URL)
  const { actions } = useAnimations(animations, group)

  // Find the Head bone once the model is loaded, so we can fake a "blink"
  // by giving it a quick vertical squash — this model has no eyelids, so a
  // literal blink isn't possible, but this reads as a small idle tic.
  useEffect(() => {
    headRef.current = scene.getObjectByName('Head') || null
  }, [scene])

  // Crossfade between Idle and the speaking gesture whenever isSpeaking flips.
  useEffect(() => {
    const idle = actions[IDLE_CLIP]
    const speak = actions[SPEAKING_CLIP]
    if (!idle || !speak) return

    if (isSpeaking) {
      idle.fadeOut(0.3)
      speak.reset().fadeIn(0.3).play()
    } else {
      speak.fadeOut(0.3)
      idle.reset().fadeIn(0.3).play()
    }

    return () => {
      idle.fadeOut(0.2)
      speak.fadeOut(0.2)
    }
  }, [isSpeaking, actions])

  // Start on Idle as soon as actions exist.
  useEffect(() => {
    actions[IDLE_CLIP]?.reset().fadeIn(0.3).play()
  }, [actions])

  useFrame((state, delta) => {
    // Occasional quick head-scale "blink" tic, only while idle (not speaking).
    if (headRef.current && !isSpeaking) {
      blinkTimer.current += delta
      const t = blinkTimer.current
      if (t > nextBlinkAt.current && t < nextBlinkAt.current + 0.15) {
        const phase = (t - nextBlinkAt.current) / 0.15
        const squash = 1 - Math.sin(phase * Math.PI) * 0.4
        headRef.current.scale.y = squash
      } else if (t >= nextBlinkAt.current + 0.15) {
        headRef.current.scale.y = 1
        blinkTimer.current = 0
        nextBlinkAt.current = 2 + Math.random() * 3
      }
    }
  })

  // No position/scale adjustment on the model itself — this matches
  // three.js's own official example for this exact asset, which places
  // it at the origin with scale 1. The model's real-world units are
  // large (~4.5 units tall), which is why the camera in AvatarViewer is
  // positioned much further back than you'd expect for a "human-scale"
  // character — matching a scaled-down model would need different
  // numbers, but modifying the model's own transform caused more
  // problems than it solved (see git history / session notes).
  return <primitive ref={group} object={scene} />
}

useGLTF.preload(MODEL_URL)

export default RobotAvatar
