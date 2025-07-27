# GovernAIce Project Structure

## 📁 Overview

This document provides a detailed breakdown of the GovernAIce repository structure, explaining the purpose and contents of each directory and key file.

## 🏗️ Root Directory Structure

```
GovernAIce/
├── 📁 backend/           # Flask API backend
├── 📁 frontend/          # React/TypeScript frontend
├── 📁 ml/               # ML pipeline and models
├── 📁 data/             # Data storage
├── 📁 docs/             # Documentation
├── 📁 scripts/          # Utility scripts
├── 📁 docker/           # Docker configurations
├── 📁 use_cases/        # Use case implementations
├── 📁 tests/            # Test suite
├── 📄 main.py           # Main entry point
├── 📄 requirements.txt  # Python dependencies
├── 📄 README.md         # Main documentation
└── 📄 .gitignore        # Git ignore rules
```

## 📁 Detailed Directory Breakdown

### 🔧 Backend (`backend/`)

**Purpose**: Flask-based REST API that serves the frontend and provides ML model endpoints.

**Key Files**:
- `app.py` - Main Flask application with all API endpoints
- `llm_utils.py` - LLM integration utilities (Together AI, OpenAI, etc.)
- `main.py` - FastAPI alternative entry point
- `requirements.txt` - Backend-specific dependencies

**API Endpoints**:
- `/upload-and-analyze/` - Document upload and compliance analysis
- `/documents/` - Document management
- `/metadata/countries/` - Available countries
- `/metadata/policies/` - Available policies
- `/reports/` - Report generation and management

### 🎨 Frontend (`frontend/`)

**Purpose**: React/TypeScript web application providing the user interface.

**Key Files**:
- `src/components/` - React components
- `src/api/` - API integration layer
- `src/contexts/` - React contexts for state management
- `package.json` - Node.js dependencies
- `vite.config.js` - Vite build configuration

**Key Components**:
- `ComplianceAnalysisWidget.tsx` - Main compliance analysis interface
- `UploadProjectWidget.tsx` - Document upload interface
- `PolicyAnalysisResults.tsx` - Results display
- `OECDScoreWidget.tsx` - OECD framework visualization
- `NISTAILifestyleWidget.tsx` - NIST lifecycle visualization

### 🤖 ML Pipeline (`ml/`)

**Purpose**: Core machine learning models and utilities for AI policy analysis.

**Structure**:
```
ml/
├── 📁 models/           # ML models
│   ├── compliance_checker.py    # Compliance analysis model
│   ├── policy_comparator.py     # Policy comparison model
│   └── principle_assessor.py    # Principle-based assessment
├── 📁 embeddings/       # Text embedding system
│   ├── embedding_system.py      # Main embedding system
│   └── text_extractor.py       # Text extraction utilities
├── 📁 config/           # Configuration
│   └── settings.py      # ML configuration settings
└── 📁 utils/            # Utilities
    └── db_connection.py # Database connection utilities
```

**Key Models**:
- **Compliance Checker**: Analyzes documents against country-specific policies
- **Policy Comparator**: Compares policies across different frameworks
- **Principle Assessor**: Evaluates compliance with AI ethics principles

### 📊 Data (`data/`)

**Purpose**: Storage for policy documents, processed data, and embeddings.

**Structure**:
```
data/
├── 📁 raw/              # Raw policy documents (PDFs, etc.)
├── 📁 processed/        # Processed and cleaned data
└── 📁 embeddings/       # Generated embeddings
```

**Note**: This directory is gitignored to prevent large files from being committed.

### 📚 Documentation (`docs/`)

**Purpose**: Project documentation and guides.

**Files**:
- `PROJECT_STRUCTURE.md` - This file
- `COMPLIANCE_WIDGET_IMPROVEMENTS.md` - Frontend widget improvements
- `TEST_COMPLIANCE_WIDGET.md` - Testing documentation
- `DOCKER_SETUP.md` - Docker documentation
- `ML_INTEGRATION_PLAN.md` - ML Architecture
- `ML_INTEGRATION_TASK_ASSIGNMENT.md` - Tasks

### 🔧 Scripts (`scripts/`)

**Purpose**: Utility scripts for development and deployment.

**Files**:
- `setup.sh` - Automated setup script for development environment

### 🐳 Docker (`docker/`)

**Purpose**: Docker configurations for different deployment scenarios.

**Files**:
- `docker-compose.dev.yml` - Development environment
- `docker-compose.local.yml` - Local deployment
- `Dockerfile.dev` - Development Dockerfile
- `Dockerfile.local` - Local Dockerfile
- `dev-docker.sh` - Development setup script
- `local-docker.sh` - Local setup script

### 🧪 Use Cases (`use_cases/`)

**Purpose**: Implementation examples and use case demonstrations.

**Files**:
- `uc1.py` - Use case 1 implementation
- `uc3.py` - Use case 3 implementation
- `compliance_analysis/` - Compliance analysis examples
- `policy_comparison/` - Policy comparison examples
- `principle_assessment/` - Principle assessment examples

### 🧪 Tests (`tests/`)

**Purpose**: Test suite for the application.

**Files**:
- `test_imports.py` - Import tests for all modules
- `__init__.py` - Test package initialization

## 📄 Key Root Files

### `main.py`
**Purpose**: Main entry point for the ML pipeline.

**Features**:
- Interactive menu for selecting ML modules
- Error handling and dependency checking
- Module import management

### `requirements.txt`
**Purpose**: Comprehensive Python dependencies for the entire platform.

**Categories**:
- Core dependencies (numpy, pandas, requests)
- Web framework (Flask, FastAPI)
- AI/ML (PyTorch, transformers, sentence-transformers)
- Database (pymongo)
- Text processing (OpenCV, PyPDF2, etc.)
- Visualization (matplotlib, seaborn, plotly)

### `README.md`
**Purpose**: Main project documentation with installation and usage instructions.

**Sections**:
- Project overview and features
- Installation instructions
- Usage examples
- API documentation
- Contributing guidelines

### `.gitignore`
**Purpose**: Comprehensive git ignore rules for the entire platform.

**Categories**:
- Environment files (.env)
- Python cache and virtual environments
- Node.js dependencies
- Large files and binaries
- IDE and OS files
- Logs and temporary files

## 🔄 Data Flow

1. **Document Upload**: Frontend → Backend → ML Pipeline
2. **Policy Retrieval**: ML Pipeline → MongoDB → Policy Data
3. **Analysis**: ML Models → Embeddings → Similarity Analysis
4. **Results**: Analysis → Backend → Frontend → Visualization

## 🛠️ Development Workflow

1. **Setup**: Run `./scripts/setup.sh`
2. **Backend**: `python main.py` or `cd backend && python app.py`
3. **Frontend**: `cd frontend && npm run dev`
4. **Testing**: `python -m pytest tests/`
5. **Docker**: `cd docker && ./dev-docker.sh`

## 📊 Key Integrations

- **MongoDB**: Policy document storage and vector search
- **Together AI**: LLM integration for analysis
- **Sentence Transformers**: Text embedding generation
- **React**: Frontend user interface
- **Flask**: Backend API server

## 🔒 Security Considerations

- API keys stored in `.env` (gitignored)
- MongoDB connection strings secured
- Large files excluded from git
- Virtual environments isolated

## 🚀 Deployment Options

1. **Development**: Local setup with virtual environment
2. **Docker**: Containerized deployment
3. **Production**: Cloud deployment with proper environment variables

This structure ensures modularity, maintainability, and scalability while keeping the codebase organized and functional. 
