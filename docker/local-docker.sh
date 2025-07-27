#!/bin/bash

# GovernAIce - Local Docker Setup
# Updated for new organized structure

echo "🏠 Starting GovernAIce Local Environment..."

# Stop containers and remove volumes
docker compose -f docker-compose.local.yml down -v

# Build and start containers
docker compose -f docker-compose.local.yml up --build

echo "✅ Local environment started!"
echo "📊 Services available at:"
echo "  - Backend API: http://localhost:5001"
echo "  - Frontend: http://localhost:5173"
echo "  - MongoDB: localhost:27017" 
