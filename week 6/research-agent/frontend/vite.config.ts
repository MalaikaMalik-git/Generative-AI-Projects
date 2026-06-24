import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    // Local dev proxy — avoids CORS issues when backend runs on :8000
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
  build: {
    // Warn if any chunk exceeds 500kb
    chunkSizeWarningLimit: 500,
  },
})
