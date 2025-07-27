#!/bin/bash

# GovernAIce - Development Docker Setup
# Updated for new organized structure

echo "🚀 Starting GovernAIce Development Environment..."

# Stop containers and remove volumes
docker compose -f docker-compose.dev.yml down -v

# Build and start containers
docker compose -f docker-compose.dev.yml up --build

echo "✅ Development environment started!"
echo "📊 Services available at:"
echo "  - Backend API: http://localhost:5001"
echo "  - Frontend: http://localhost:5173"
echo "  - MongoDB: localhost:27017" 
