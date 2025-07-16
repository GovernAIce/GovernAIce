import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
<<<<<<< HEAD
=======
  server: {
    proxy: {
      '/chat': 'http://localhost:5001',
      '/metadata': 'http://localhost:5001',
      '/upload-and-analyze': 'http://localhost:5001',
      '/api': 'http://localhost:5001',
      // Add more endpoints as needed
    },
  },
>>>>>>> dfab21b9ab606bef9aae393aba8cefb32de97b9f
})
