#!/bin/bash

# Auto-configure and start frontend
echo "🚀 Starting frontend with auto-configuration..."

# Run auto-config first
./scripts/auto-config.sh

# Kill any existing frontend process
echo "🔄 Restarting frontend..."
pkill -f "npm run dev" 2>/dev/null || true
pkill -f "vite" 2>/dev/null || true

# Wait a moment for processes to stop
sleep 2

# Start frontend
echo "🎯 Starting frontend..."
cd frontend && npm run dev 
