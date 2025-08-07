import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
          '/chat': 'http://localhost:5002',
    '/metadata': 'http://localhost:5002',
    '/api': 'http://localhost:5002',
    '/ml': 'http://localhost:5002'
      // Add more endpoints as needed
    },
  },
})
