import os
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class MLConfig:
    """Configuration settings for ML components"""
    
    # API Keys
    TOGETHER_API_KEY = os.getenv('TOGETHER_API_KEY')
    MONGODB_URI = os.getenv('MONGODB_URI', "mongodb+srv://smaranrbbtech22:aimd0MyDfx25MmSdL@govai-xlm-r-v1.xx5wl1d.mongodb.net/?retryWrites=true&w=majority&appName=govai-xlm-r-v1")
    
    # Model Settings
    DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    LEGAL_EMBEDDING_MODEL = "joelniklaus/legal-xlm-roberta-large"
    DEEPSEEK_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
    
    # Database Settings
    MONGODB_DB_NAME = "govai-xlm-r-v2"
    MONGODB_COLLECTION = "global_chunks"
    VECTOR_INDEX_NAME = "global_vector_index"
    
    # File Paths
    DATA_DIR = "data"
    RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
    PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
    EMBEDDINGS_DIR = os.path.join(DATA_DIR, "embeddings")
    
    # Processing Settings
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50
    MAX_TOKENS = 2048
    
    # Available Countries
    AVAILABLE_COUNTRIES = [
        "USA", "UK", "EU", "SINGAPORE", "CANADA", 
        "AUSTRALIA", "JAPAN", "CHINA"
    ]
    
    # Risk Categories
    RISK_CATEGORIES = {
        "data_privacy": {
            "keywords": ["data protection", "privacy", "personal data", "consent", "data subject rights", "gdpr", "ccpa"],
            "weight": 0.15
        },
        "algorithmic_bias": {
            "keywords": ["bias", "fairness", "discrimination", "equitable", "inclusive", "diverse", "prejudice"],
            "weight": 0.14
        },
        "transparency": {
            "keywords": ["transparent", "explainable", "interpretable", "disclosure", "communication", "accountability"],
            "weight": 0.13
        },
        "safety_security": {
            "keywords": ["safety", "security", "protection", "harm", "risk", "vulnerability", "cybersecurity"],
            "weight": 0.12
        },
        "human_oversight": {
            "keywords": ["human oversight", "human control", "human in the loop", "supervision", "intervention", "monitoring"],
            "weight": 0.11
        },
        "compliance_governance": {
            "keywords": ["compliance", "governance", "regulation", "legal", "standards", "certification", "audit"],
            "weight": 0.10
        },
        "risk_management": {
            "keywords": ["risk assessment", "risk management", "mitigation", "evaluation", "impact assessment"],
            "weight": 0.10
        },
        "ethical_considerations": {
            "keywords": ["ethics", "ethical", "moral", "responsible", "integrity", "values", "principles"],
            "weight": 0.09
        },
        "liability_accountability": {
            "keywords": ["liability", "accountability", "responsibility", "ownership", "legal responsibility"],
            "weight": 0.08
        },
        "innovation_development": {
            "keywords": ["innovation", "development", "research", "advancement", "technology", "progress"],
            "weight": 0.08
        }
    }
