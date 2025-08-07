#!/bin/bash

# Quick Start Script for GovernAIce
echo "🚀 GovernAIce Quick Start"
echo "========================"
echo ""
echo "Available commands:"
echo "1) ./scripts/auto-config.sh     - Auto-configure frontend for current backend"
echo "2) ./scripts/start-frontend.sh   - Auto-configure and start frontend"
echo "3) ./scripts/start-app.sh        - Full auto-start (backend + frontend)"
echo ""
echo "Manual commands:"
echo "Docker:    docker-compose -f docker/docker-compose.local.yml up"
echo "Direct:    cd backend && python app.py"
echo "Frontend:  cd frontend && npm run dev"
echo ""
echo "Current backend status:"

# Check backend status
if curl -s http://localhost:5001/metadata/country/ > /dev/null 2>&1; then
    echo "✅ Docker backend running on port 5001"
elif curl -s http://localhost:5002/metadata/country/ > /dev/null 2>&1; then
    echo "✅ Direct backend running on port 5002"
else
    echo "❌ No backend detected"
fi

# Check frontend status
if lsof -i :5173 > /dev/null 2>&1; then
    echo "✅ Frontend running on port 5173"
else
    echo "❌ Frontend not running"
fi 
