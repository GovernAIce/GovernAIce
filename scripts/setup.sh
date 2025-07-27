#!/bin/bash

# GovernAIce - AI Policy Compliance Assessment Tool
# Setup script for development environment

set -e  # Exit on any error

echo "🚀 GovernAIce Setup Script"
echo "=========================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python is installed
check_python() {
    print_status "Checking Python installation..."
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
        print_success "Python $PYTHON_VERSION found"
    else
        print_error "Python 3 is not installed. Please install Python 3.8+ first."
        exit 1
    fi
}

# Check if Node.js is installed
check_node() {
    print_status "Checking Node.js installation..."
    if command -v node &> /dev/null; then
        NODE_VERSION=$(node --version)
        print_success "Node.js $NODE_VERSION found"
    else
        print_warning "Node.js not found. Frontend development will not be available."
        print_status "To install Node.js, visit: https://nodejs.org/"
    fi
}

# Create virtual environment
setup_python_env() {
    print_status "Setting up Python virtual environment..."
    
    if [ -d "venv" ]; then
        print_warning "Virtual environment already exists. Removing old one..."
        rm -rf venv
    fi
    
    python3 -m venv venv
    source venv/bin/activate
    
    print_status "Upgrading pip..."
    pip install --upgrade pip
    
    print_success "Virtual environment created and activated"
}

# Install Python dependencies
install_python_deps() {
    print_status "Installing Python dependencies..."
    
    if [ ! -f "requirements.txt" ]; then
        print_error "requirements.txt not found!"
        exit 1
    fi
    
    pip install -r requirements.txt
    
    print_status "Installing development dependencies..."
    pip install pytest jupyter ipykernel
    
    print_success "Python dependencies installed"
}

# Setup frontend
setup_frontend() {
    if command -v node &> /dev/null; then
        print_status "Setting up frontend..."
        
        cd frontend
        
        if [ -d "node_modules" ]; then
            print_warning "node_modules already exists. Removing..."
            rm -rf node_modules
        fi
        
        npm install
        
        print_success "Frontend dependencies installed"
        cd ..
    else
        print_warning "Skipping frontend setup (Node.js not available)"
    fi
}

# Create environment file
setup_env() {
    print_status "Setting up environment configuration..."
    
    if [ ! -f ".env" ]; then
        cat > .env << EOF
# GovernAIce Environment Configuration

# API Keys (Required)
TOGETHER_API_KEY=your_together_ai_key_here
MONGODB_URI=your_mongodb_connection_string_here

# Optional API Keys
OPENAI_API_KEY=your_openai_key_here
GOOGLE_API_KEY=your_google_key_here

# Database Settings
MONGODB_DB_NAME=govai-xlm-r-v2
MONGODB_COLLECTION=global_chunks

# Model Settings
DEFAULT_EMBEDDING_MODEL=all-MiniLM-L6-v2
LEGAL_EMBEDDING_MODEL=joelniklaus/legal-xlm-roberta-large

# Development Settings
DEBUG=True
FLASK_ENV=development
EOF
        print_success "Environment file created (.env)"
        print_warning "Please edit .env with your actual API keys and settings"
    else
        print_warning ".env file already exists"
    fi
}

# Create data directories
setup_directories() {
    print_status "Creating data directories..."
    
    mkdir -p data/raw
    mkdir -p data/processed
    mkdir -p data/embeddings
    mkdir -p logs
    mkdir -p reports
    
    print_success "Data directories created"
}

# Run tests
run_tests() {
    print_status "Running basic tests..."
    
    if [ -d "tests" ]; then
        python -m pytest tests/ -v
        print_success "Tests completed"
    else
        print_warning "No tests directory found"
    fi
}

# Main setup function
main() {
    echo ""
    print_status "Starting GovernAIce setup..."
    echo ""
    
    # Check prerequisites
    check_python
    check_node
    echo ""
    
    # Setup environment
    setup_python_env
    install_python_deps
    setup_frontend
    setup_env
    setup_directories
    echo ""
    
    # Run tests
    run_tests
    echo ""
    
    print_success "🎉 GovernAIce setup completed successfully!"
    echo ""
    print_status "Next steps:"
    echo "  1. Edit .env file with your API keys"
    echo "  2. Activate virtual environment: source venv/bin/activate"
    echo "  3. Run the application: python main.py"
    echo "  4. For frontend: cd frontend && npm run dev"
    echo ""
    print_status "For more information, see README.md"
}

# Run main function
main "$@" 
