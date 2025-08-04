# Quick Setup Guide

## 🚀 Get Started in 5 Minutes

### 1. Clone & Setup
```bash
git clone https://github.com/GovernAIce/GovernAIce.git
cd GovernAIce
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Create Environment Files ⚠️ **REQUIRED**

**Backend** (`backend/.env`):
```bash
MONGO_URI=mongodb://localhost:27017/governaice
SECRET_KEY=your_secret_key_here
VOYAGE_API_KEY=your_voyage_api_key_here
```

**Frontend** (`frontend/.env`):
```bash
VITE_API_BASE_URL=http://localhost:8000
```

**ML** (`ml/.env`):
```bash
MONGO_URI=mongodb://localhost:27017/governaice
VOYAGE_API_KEY=your_voyage_api_key_here
```

### 3. Start Services
```bash
# Terminal 1 - Backend
cd backend && python app.py

# Terminal 2 - Frontend  
cd frontend && npm install && npm run dev

# Terminal 3 - ML (optional)
cd ml && python -m ml.models.compliance_checker
```

### 4. Access Application
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000

## 🔧 Common Issues

| Problem | Solution |
|---------|----------|
| "Module not found" | `pip install -r requirements.txt` |
| "Port already in use" | Change port in `.env` files |
| "API key error" | Check your `.env` files exist |
| "Database connection failed" | Start MongoDB or check `MONGO_URI` |

## 📝 Notes
- `.env` files are **NOT** in Git (security)
- You **MUST** create them manually
- See `SETUP.md` for detailed instructions 
