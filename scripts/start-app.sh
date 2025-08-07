#!/bin/bash

# GovernAIce Auto-Start Script
echo "🎯 GovernAIce Auto-Start Script"
echo "================================"

# Function to detect backend
detect_backend() {
    if curl -s http://localhost:5001/metadata/country/ > /dev/null 2>&1; then
        echo "docker"
    elif curl -s http://localhost:5002/metadata/country/ > /dev/null 2>&1; then
        echo "direct"
    else
        echo "none"
    fi
}

# Function to start backend
start_backend() {
    local mode=$1
    if [ "$mode" = "docker" ]; then
        echo "🐳 Starting Docker backend..."
        docker-compose -f docker/docker-compose.local.yml up -d backend
        sleep 10
    elif [ "$mode" = "direct" ]; then
        echo "🐍 Starting direct backend..."
        cd backend && python app.py &
        sleep 5
    fi
}

# Function to configure and start frontend
start_frontend() {
    echo "⚙️  Auto-configuring frontend..."
    ./scripts/auto-config.sh
    
    echo "🎨 Starting frontend..."
    cd frontend && npm run dev
}

# Main logic
echo "🔍 Detecting current setup..."

# Check if Docker containers are running
if docker ps | grep -q "docker-backend"; then
    echo "✅ Docker containers detected"
    BACKEND_MODE="docker"
elif pgrep -f "python app.py" > /dev/null; then
    echo "✅ Direct backend detected"
    BACKEND_MODE="direct"
else
    echo "❓ No backend detected"
    echo "Choose your setup:"
    echo "1) Docker (recommended for production)"
    echo "2) Direct Python/NPM (recommended for development)"
    read -p "Enter choice (1 or 2): " choice
    
    case $choice in
        1)
            BACKEND_MODE="docker"
            start_backend "docker"
            ;;
        2)
            BACKEND_MODE="direct"
            start_backend "direct"
            ;;
        *)
            echo "Invalid choice. Exiting."
            exit 1
            ;;
    esac
fi

# Wait for backend to be ready
echo "⏳ Waiting for backend to be ready..."
for i in {1..30}; do
    if [ "$(detect_backend)" != "none" ]; then
        echo "✅ Backend is ready!"
        break
    fi
    echo "⏳ Waiting... ($i/30)"
    sleep 2
done

# Start frontend
start_frontend 
