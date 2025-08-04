import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/chat': 'http://localhost:5001',
      '/metadata': 'http://localhost:5001',
      '/api': 'http://localhost:5001',
      '/ml': 'http://localhost:5001'
      // Add more endpoints as needed
    },
  },
})
