#!/bin/bash

# GovernAIce Environment Setup Script
# This script helps you set up the required environment variables

echo "🔧 GovernAIce Environment Setup"
echo "================================"

# Check if .env file exists
if [ ! -f .env ]; then
    echo "Creating .env file..."
    touch .env
fi

# Function to prompt for API key
prompt_for_key() {
    local key_name=$1
    local description=$2
    local current_value=$(grep "^${key_name}=" .env 2>/dev/null | cut -d'=' -f2)
    
    echo ""
    echo "🔑 ${description}"
    if [ ! -z "$current_value" ]; then
        echo "Current value: ${current_value}"
        read -p "Enter new value (or press Enter to keep current): " new_value
        if [ ! -z "$new_value" ]; then
            # Update .env file
            if grep -q "^${key_name}=" .env; then
                sed -i.bak "s/^${key_name}=.*/${key_name}=${new_value}/" .env
            else
                echo "${key_name}=${new_value}" >> .env
            fi
        fi
    else
        read -p "Enter your ${key_name}: " new_value
        if [ ! -z "$new_value" ]; then
            echo "${key_name}=${new_value}" >> .env
        fi
    fi
}

echo "📝 Setting up environment variables..."
echo "You'll need to provide API keys for the services you want to use."

# Gemini API Key
prompt_for_key "GEMINI_API_KEY" "Google Gemini API Key (for LLM functionality)"

# Together AI API Key
prompt_for_key "TOGETHER_API_KEY" "Together AI API Key (for ML models)"

# MongoDB URI
prompt_for_key "MONGODB_URI" "MongoDB Connection URI"

echo ""
echo "✅ Environment setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. If you're using Docker, update the docker-compose.dev.yml file with your API keys"
echo "2. Run 'docker-compose -f docker/docker-compose.dev.yml up' to start the services"
echo "3. Or run 'python main.py' to start the application locally"
echo ""
echo "⚠️  Note: Make sure to keep your API keys secure and never commit them to version control!" 
