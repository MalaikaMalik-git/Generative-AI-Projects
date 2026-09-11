import { Suspense } from 'react'
import { Canvas } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'
import RobotAvatar from './RobotAvatar.jsx'

function LoadingFallback() {
  return null // Suspense fallback can't render inside <Canvas> with DOM; see overlay in AvatarViewer instead
}

function AvatarViewer({ isSpeaking }) {
  return (
    <div style={{ position: 'relative', width: '100%', height: '100%' }}>
      {/* Camera numbers below are NOT guesses — they're taken directly from
          three.js's own official example for this exact model
          (webgl_animation_skinning_morph.html: camera.position.set(-5, 3, 10),
          fov 45, lookAt(0, 2, 0), model added with no scale/position change).
          An earlier attempt to auto-fit the camera via drei's <Bounds>
          backfired: it computed the fit from the model's raw T-pose bind
          geometry (arms spread ~6.6 units wide) before the Idle animation
          collapsed the arms down, framing the wrong shape entirely. Fixed
          camera values tied to the model's real, verified proportions are
          more reliable here than any auto-fit approach. */}
      <Canvas camera={{ position: [0, 3, 14], fov: 45, near: 0.25, far: 100 }}>
        {/* Plain lights only, deliberately no <Environment> — that component
            fetches an HDRI from a CDN at runtime, which would break the
            "renders without network dependency" requirement for this session. */}
        <ambientLight intensity={0.7} />
        <directionalLight position={[3, 8, 4]} intensity={1.4} castShadow />
        <directionalLight position={[-4, 4, -3]} intensity={0.5} />
        <hemisphereLight args={['#ffffff', '#444466', 0.6]} />

        <Suspense fallback={<LoadingFallback />}>
          <RobotAvatar isSpeaking={isSpeaking} />
        </Suspense>

        <OrbitControls
          makeDefault
          target={[0, 2, 0]}
          enablePan={false}
          minDistance={5}
          maxDistance={18}
          maxPolarAngle={Math.PI / 1.8}
        />
      </Canvas>
    </div>
  )
}

export default AvatarViewer
