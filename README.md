# GovernAIce - AI Policy Compliance Assessment Tool

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![React](https://img.shields.io/badge/React-18.0+-blue.svg)](https://reactjs.org/)

## 🎯 Overview

GovernAIce is a comprehensive AI policy compliance assessment platform that helps organizations evaluate their AI systems against global regulatory frameworks. The platform provides automated analysis, risk assessment, and strategic recommendations for AI governance.

## 🏗️ Architecture

```
GovernAIce/
├── 📁 backend/           # Flask API backend
│   ├── app.py           # Main Flask application
│   ├── llm_utils.py     # LLM integration utilities
│   ├── main.py          # FastAPI alternative
│   └── requirements.txt # Backend dependencies
├── 📁 frontend/          # React/TypeScript frontend
│   ├── src/             # React components
│   ├── public/          # Static assets
│   └── package.json     # Frontend dependencies
├── 📁 ml/               # ML pipeline and models
│   ├── models/          # ML models (compliance, policy, principle)
│   ├── embeddings/      # Text embedding system
│   ├── config/          # Configuration settings
│   └── utils/           # Database and utility functions
├── 📁 data/             # Data storage
│   ├── raw/             # Raw policy documents
│   ├── processed/       # Processed data
│   └── embeddings/      # Generated embeddings
├── 📁 docs/             # Documentation
├── 📁 scripts/          # Utility scripts
├── 📁 docker/           # Docker configurations
├── 📁 use_cases/        # Use case implementations
├── 📁 tests/            # Test suite
└── 📄 main.py           # Main entry point
```

## 🚀 Features

### 🤖 ML Pipeline
- **Compliance Checker**: Automated AI policy compliance analysis
- **Policy Comparator**: Cross-framework policy comparison
- **Principle Assessor**: Principle-based assessment tools
- **Embedding System**: Advanced text embedding and similarity analysis

### 🌍 Global Policy Coverage
- **USA**: NIST AI Framework, CCPA, sector-specific regulations
- **EU**: AI Act, GDPR, Digital Services Act
- **UK**: AI Safety Framework, Data Protection Act
- **Singapore**: AI Governance Framework
- **Canada**: AIDA (Artificial Intelligence and Data Act)
- **Australia**: AI Ethics Principles
- **Japan**: AI Safety Guidelines
- **China**: AI Governance Regulations

### 📊 Analysis Capabilities
- **Risk Assessment**: Multi-dimensional risk analysis
- **Gap Analysis**: Identify compliance gaps and opportunities
- **Strategic Recommendations**: AI-powered improvement strategies
- **Visualization**: Interactive charts and reports

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- Node.js 16+
- MongoDB (for policy database)
- Docker (optional)

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/GovernAIce/GovernAIce.git
   cd GovernAIce
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install frontend dependencies**
   ```bash
   cd frontend
   npm install
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and database settings
   ```

5. **Run the application**
   ```bash
   # Backend
   python main.py
   
   # Frontend (in another terminal)
   cd frontend
   npm run dev
   ```

## 🐳 Docker Deployment

### Development
```bash
cd docker
./dev-docker.sh
```

### Production
```bash
cd docker
./local-docker.sh
```

## 📚 Usage

### ML Pipeline

#### 1. Compliance Analysis
```python
from ml.models.compliance_checker import AIComplianceChecker

checker = AIComplianceChecker()
result = checker.run_compliance_check(
    document_input="path/to/document.txt",
    country="USA"
)
print(f"Compliance Score: {result.overall_score}/10")
```

#### 2. Policy Comparison
```python
from ml.models.policy_comparator import AIPolicyComparator

comparator = AIPolicyComparator(api_key="your_api_key")
result = comparator.compare_policy_document(
    user_document_text="Your policy text",
    reference_embeddings_file="path/to/embeddings.json",
    target_country="EU"
)
```

#### 3. Principle Assessment
```python
from ml.models.principle_assessor import PrincipleBasedAIAssessment

assessor = PrincipleBasedAIAssessment(api_key="your_api_key")
assessment = assessor.assess_customer_document(
    customer_document_path="path/to/document.txt",
    embeddings_file="path/to/embeddings.json"
)
```

### Web Interface

1. **Upload Document**: Upload your AI policy or system documentation
2. **Select Countries**: Choose target jurisdictions for analysis
3. **Run Analysis**: Get comprehensive compliance assessment
4. **Review Results**: Interactive dashboard with detailed insights
5. **Export Reports**: Generate PDF or JSON reports

## 🔧 Configuration

### Environment Variables
```bash
# API Keys
TOGETHER_API_KEY=your_together_ai_key
MONGODB_URI=your_mongodb_connection_string

# Model Settings
DEFAULT_EMBEDDING_MODEL=all-MiniLM-L6-v2
LEGAL_EMBEDDING_MODEL=joelniklaus/legal-xlm-roberta-large

# Database Settings
MONGODB_DB_NAME=govai-xlm-r-v2
MONGODB_COLLECTION=global_chunks
```

### ML Configuration
Edit `ml/config/settings.py` to customize:
- Risk categories and weights
- Available countries
- Model parameters
- Processing settings

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/

# Run specific test modules
python -m pytest tests/test_imports.py
python -m pytest tests/test_compliance.py
```

## 📊 API Documentation

### Backend Endpoints

#### Compliance Analysis
- `POST /upload-and-analyze/` - Upload and analyze document
- `GET /documents/` - List analyzed documents
- `GET /documents/<doc_id>/` - Get document analysis results

#### Policy Management
- `GET /metadata/countries/` - Get available countries
- `GET /metadata/policies/` - Get available policies
- `GET /metadata/domains/` - Get policy domains

#### Reports and Analytics
- `GET /reports/` - List generated reports
- `POST /reports/` - Generate new report
- `GET /reports/<report_id>/` - Get specific report

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 for Python code
- Use TypeScript for frontend components
- Write tests for new features
- Update documentation for API changes

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Policy Data**: Global AI policy frameworks and regulations
- **ML Models**: Sentence Transformers, Together AI, DeepSeek
- **Frontend**: React, TypeScript, Tailwind CSS
- **Backend**: Flask, FastAPI, MongoDB

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/GovernAIce/GovernAIce/issues)
- **Documentation**: [Wiki](https://github.com/GovernAIce/GovernAIce/wiki)
- **Email**: support@govai-ce.com

---

**GovernAIce** - Making AI governance accessible and actionable for organizations worldwide.
