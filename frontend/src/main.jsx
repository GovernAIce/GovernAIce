<<<<<<< HEAD
<<<<<<< HEAD
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
=======
import React from 'react';
import { createRoot } from 'react-dom/client';
import './index.css';
import App from './App.jsx';

=======
import React from 'react';
import { createRoot } from 'react-dom/client';
import './index.css';
import App from './App.jsx';

>>>>>>> cd4bb56c766d7bca6f05cf4fb692e6bd07dec2ce
const root = document.getElementById('root');
if (root) {
  createRoot(root).render(
    <React.StrictMode>
      <App />
    </React.StrictMode>
  );
}
<<<<<<< HEAD
>>>>>>> dfab21b9ab606bef9aae393aba8cefb32de97b9f
=======
>>>>>>> cd4bb56c766d7bca6f05cf4fb692e6bd07dec2ce
