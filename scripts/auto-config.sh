#!/bin/bash

# Auto-configure frontend based on backend availability
echo "🔍 Detecting backend configuration..."

# Check if Docker backend is running on port 5001
if curl -s http://localhost:5001/metadata/country/ > /dev/null 2>&1; then
    echo "✅ Docker backend detected on port 5001"
    BACKEND_URL="http://localhost:5001"
    BACKEND_TYPE="docker"
elif curl -s http://localhost:5002/metadata/country/ > /dev/null 2>&1; then
    echo "✅ Direct backend detected on port 5002"
    BACKEND_URL="http://localhost:5002"
    BACKEND_TYPE="direct"
else
    echo "⚠️  No backend detected. Using default port 5002"
    BACKEND_URL="http://localhost:5002"
    BACKEND_TYPE="default"
fi

# Update frontend .env file
echo "📝 Updating frontend configuration..."
cat > frontend/.env << EOF
VITE_API_BASE_URL=${BACKEND_URL}
VITE_ML_SERVICE_URL=http://localhost:8001
VITE_ENABLE_ML_FEATURES=true
VITE_ENABLE_CHAT=true
EOF

echo "✅ Frontend configured for ${BACKEND_TYPE} backend (${BACKEND_URL})"
echo "🔄 Restart frontend to apply changes: cd frontend && npm run dev" 
