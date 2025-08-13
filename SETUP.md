# GovernAIce Setup Guide

## Prerequisites
- Python 3.8+
- Node.js 16+
- Git
- Docker (optional, for containerized setup)

## Initial Setup

### 1. Clone the Repository
```bash
git clone https://github.com/GovernAIce/GovernAIce.git
cd GovernAIce
```

### 2. Environment Files Setup

**Important**: The `.env` files are not tracked in Git for security reasons. You need to create them manually.

#### Backend Environment (.env)
Create `backend/.env`:
```bash
# Database Configuration
DATABASE_URL=mongodb://localhost:27017/governaice
MONGO_URI=mongodb://localhost:27017/governaice

# API Configuration
API_KEY=your_api_key_here
SECRET_KEY=your_secret_key_here

# ML Service Configuration
ML_SERVICE_URL=http://localhost:8001
VOYAGE_API_KEY=your_voyage_api_key_here

# Logging
LOG_LEVEL=INFO
```

#### Frontend Environment (.env)
Create `frontend/.env`:
```bash
# API Configuration
VITE_API_BASE_URL=http://localhost:8000
VITE_ML_SERVICE_URL=http://localhost:8001

# Feature Flags
VITE_ENABLE_ML_FEATURES=true
VITE_ENABLE_CHAT=true
```

#### ML Environment (.env)
Create `ml/.env`:
```bash
# Database
MONGO_URI=mongodb://localhost:27017/governaice

# API Keys
VOYAGE_API_KEY=your_voyage_api_key_here
OPENAI_API_KEY=your_openai_api_key_here

# Model Configuration
EMBEDDING_MODEL=voyage-3-large
LLM_MODEL=gpt-4
```

#### Docker Environment (.env)
Create `docker/.env`:
```bash
# Database
MONGO_URI=mongodb://mongo:27017/governaice

# Services
BACKEND_PORT=8000
FRONTEND_PORT=3000
ML_SERVICE_PORT=8001

# API Keys
VOYAGE_API_KEY=your_voyage_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. Python Setup

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 4. Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

### 5. Backend Setup

```bash
cd backend

# Install dependencies (if not already done)
pip install -r requirements.txt

# Start the backend server
python app.py
```

### 6. ML Service Setup

```bash
cd ml

# Install dependencies (if not already done)
pip install -r requirements.txt

# Start the ML service
python -m ml.models.compliance_checker
```

## Docker Setup (Alternative)

If you prefer using Docker:

```bash
# Copy environment file
cp docker/.env.example docker/.env
# Edit docker/.env with your configuration

# Start services
cd docker
docker compose -f docker-compose.local.yml up --build

```

## Development Workflow

### 1. Starting Development
```bash
# Terminal 1 - Backend
cd backend
source ../venv/bin/activate
python app.py

# Terminal 2 - Frontend
cd frontend
npm run dev

# Terminal 3 - ML Service (if needed)
cd ml
source ../venv/bin/activate
python -m ml.models.compliance_checker
```

### 2. Making Changes
- Create a new branch: `git checkout -b your-feature-branch`
- Make your changes
- Commit: `git commit -m "Your commit message"`
- Push: `git push origin your-feature-branch`
- Create a PR to main

### 3. Pulling Latest Changes
```bash
git checkout main
git pull origin main

# Update dependencies if needed
pip install -r requirements.txt
cd frontend && npm install
```

## Troubleshooting

### Common Issues

1. **Environment files missing**
   - Solution: Create the `.env` files as shown above

2. **Port conflicts**
   - Check if ports 3000, 8000, 8001 are available
   - Change ports in `.env` files if needed

3. **Database connection issues**
   - Ensure MongoDB is running
   - Check `MONGO_URI` in your `.env` files

4. **API key errors**
   - Verify your API keys are correctly set in `.env` files
   - Check if keys have proper permissions

### Getting Help
- Check the `docs/` directory for detailed documentation
- Review `docs/TROUBLESHOOTING.md` for common issues
- Create an issue on GitHub for bugs

## Security Notes

- Never commit `.env` files to Git
- Keep your API keys secure
- Use different keys for development and production
- Regularly rotate your API keys

## Environment Variables Reference

| Variable | Description | Required |
|----------|-------------|----------|
| `MONGO_URI` | MongoDB connection string | Yes |
| `VOYAGE_API_KEY` | Voyage AI API key | Yes |
| `OPENAI_API_KEY` | OpenAI API key | Optional |
| `SECRET_KEY` | Flask secret key | Yes |
| `VITE_API_BASE_URL` | Frontend API base URL | Yes |
| `LOG_LEVEL` | Logging level | No |

## Next Steps

1. Set up your environment files
2. Install dependencies
3. Start the development servers
4. Visit `http://localhost:3000` to see the application
5. Check the API at `http://localhost:8000` for backend functionality 
