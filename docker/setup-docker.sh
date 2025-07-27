#!/bin/bash

# GovernAIce - Docker Setup Script
# Handles Docker deployment with the new organized structure

set -e  # Exit on any error

echo "🐳 GovernAIce Docker Setup"
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

# Check if Docker is installed
check_docker() {
    print_status "Checking Docker installation..."
    if command -v docker &> /dev/null; then
        DOCKER_VERSION=$(docker --version)
        print_success "Docker $DOCKER_VERSION found"
    else
        print_error "Docker is not installed. Please install Docker first."
        print_status "Visit: https://docs.docker.com/get-docker/"
        exit 1
    fi
}

# Check if Docker Compose is available
check_docker_compose() {
    print_status "Checking Docker Compose..."
    if command -v docker-compose &> /dev/null || docker compose version &> /dev/null; then
        print_success "Docker Compose found"
    else
        print_error "Docker Compose is not available."
        print_status "Please install Docker Compose or use Docker Desktop."
        exit 1
    fi
}

# Create environment file for Docker
setup_docker_env() {
    print_status "Setting up Docker environment..."
    
    if [ ! -f ".env" ]; then
        cat > .env << EOF
# GovernAIce Docker Environment Configuration

# API Keys (Required)
TOGETHER_API_KEY=your_together_ai_key_here
MONGODB_URI=your_mongodb_connection_string_here

# Optional API Keys
OPENAI_API_KEY=your_openai_key_here
GOOGLE_API_KEY=your_google_key_here
GEMINI_API_KEY=your_gemini_key_here

# Database Settings
MONGO_DB_NAME=govai-xlm-r-v2
MONGO_DB_COLLECTION=global_chunks
DB_NAME=Training

# Model Settings
DEFAULT_EMBEDDING_MODEL=all-MiniLM-L6-v2
LEGAL_EMBEDDING_MODEL=joelniklaus/legal-xlm-roberta-large

# Development Settings
DEBUG=True
FLASK_ENV=development
EOF
        print_success "Docker environment file created (.env)"
        print_warning "Please edit .env with your actual API keys and settings"
    else
        print_warning ".env file already exists"
    fi
}

# Build and run development environment
run_dev_environment() {
    print_status "Building and starting development environment..."
    
    # Stop any existing containers
    docker compose -f docker-compose.dev.yml down -v 2>/dev/null || true
    
    # Build and start containers
    docker compose -f docker-compose.dev.yml up --build -d
    
    print_success "Development environment started!"
    print_status "Services available at:"
    echo "  - Backend API: http://localhost:5001"
    echo "  - Frontend: http://localhost:5173"
    echo "  - MongoDB: localhost:27017"
}

# Build and run local environment
run_local_environment() {
    print_status "Building and starting local environment..."
    
    # Stop any existing containers
    docker compose -f docker-compose.local.yml down -v 2>/dev/null || true
    
    # Build and start containers
    docker compose -f docker-compose.local.yml up --build -d
    
    print_success "Local environment started!"
    print_status "Services available at:"
    echo "  - Backend API: http://localhost:5001"
    echo "  - Frontend: http://localhost:5173"
    echo "  - MongoDB: localhost:27017"
}

# Show logs
show_logs() {
    print_status "Showing container logs..."
    docker compose -f docker-compose.dev.yml logs -f
}

# Stop all containers
stop_containers() {
    print_status "Stopping all containers..."
    docker compose -f docker-compose.dev.yml down -v
    docker compose -f docker-compose.local.yml down -v
    print_success "All containers stopped"
}

# Clean up Docker resources
cleanup() {
    print_status "Cleaning up Docker resources..."
    docker system prune -f
    print_success "Docker cleanup completed"
}

# Show help
show_help() {
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  dev     - Start development environment"
    echo "  local   - Start local environment"
    echo "  logs    - Show container logs"
    echo "  stop    - Stop all containers"
    echo "  cleanup - Clean up Docker resources"
    echo "  help    - Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 dev     # Start development environment"
    echo "  $0 local   # Start local environment"
    echo "  $0 logs    # Show logs"
}

# Main function
main() {
    case "${1:-help}" in
        "dev")
            check_docker
            check_docker_compose
            setup_docker_env
            run_dev_environment
            ;;
        "local")
            check_docker
            check_docker_compose
            setup_docker_env
            run_local_environment
            ;;
        "logs")
            show_logs
            ;;
        "stop")
            stop_containers
            ;;
        "cleanup")
            cleanup
            ;;
        "help"|*)
            show_help
            ;;
    esac
}

# Run main function
main "$@" 
