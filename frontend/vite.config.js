import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
<<<<<<< HEAD
<<<<<<< HEAD
=======
=======
>>>>>>> cd4bb56c766d7bca6f05cf4fb692e6bd07dec2ce
  server: {
    proxy: {
      '/chat': 'http://localhost:5001',
      '/metadata': 'http://localhost:5001',
      '/upload-and-analyze': 'http://localhost:5001',
      '/api': 'http://localhost:5001',
      // Add more endpoints as needed
    },
  },
<<<<<<< HEAD
>>>>>>> dfab21b9ab606bef9aae393aba8cefb32de97b9f
=======
>>>>>>> cd4bb56c766d7bca6f05cf4fb692e6bd07dec2ce
})
