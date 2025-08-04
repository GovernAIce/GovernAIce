# Troubleshooting Guide

## Common Issues and Solutions

### 1. Invalid Gemini API Key Error

**Error:**
```
ERROR:llm_utils:Gemini API call failed: 400 API key not valid. Please pass a valid API key.
```

**Solution:**
1. Get a valid Gemini API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Update the Docker compose file or set environment variable:

**Option A: Update Docker Compose**
Edit `docker/docker-compose.dev.yml`:
```yaml
environment:
  - GEMINI_API_KEY=your_actual_api_key_here
```

**Option B: Use Environment File**
Create a `.env` file in the project root:
```
GEMINI_API_KEY=your_actual_api_key_here
```

**Option C: Use the Setup Script**
```bash
./scripts/setup_env.sh
```

### 2. ML Models Import Error

**Error:**
```
WARNING:root:ML models not available: cannot import name 'search_global_chunks' from 'ml.utils.db_connection'
```

**Solution:**
This has been fixed by uncommenting the `search_global_chunks` function in `ml/utils/db_connection.py`. The function is now available for import.

### 3. MongoDB Connection Issues

**Error:**
```
MONGODB_URI environment variable not found
```

**Solution:**
1. Set the `MONGODB_URI` environment variable in your Docker compose file
2. Make sure the MongoDB connection string is correct
3. Verify that the MongoDB instance is accessible

### 4. Missing Dependencies

**Error:**
```
ModuleNotFoundError: No module named 'transformers'
```

**Solution:**
Install the required dependencies:
```bash
pip install transformers torch pymongo sentence-transformers
```

Or rebuild the Docker container:
```bash
docker-compose -f docker/docker-compose.dev.yml build
```

## Quick Fix Steps

1. **Get API Keys:**
   - [Google Gemini API Key](https://makersuite.google.com/app/apikey)
   - [Together AI API Key](https://together.ai/)

2. **Update Environment Variables:**
   ```bash
   ./scripts/setup_env.sh
   ```

3. **Update Docker Compose:**
   Edit `docker/docker-compose.dev.yml` and replace `YOUR_GEMINI_API_KEY_HERE` with your actual API key.

4. **Restart Services:**
   ```bash
   docker-compose -f docker/docker-compose.dev.yml down
   docker-compose -f docker/docker-compose.dev.yml up
   ```

## Environment Variables Reference

| Variable | Description | Required |
|----------|-------------|----------|
| `GEMINI_API_KEY` | Google Gemini API key for LLM functionality | Yes |
| `TOGETHER_API_KEY` | Together AI API key for ML models | Yes |
| `MONGODB_URI` | MongoDB connection string | Yes |
| `MONGO_DB_NAME` | MongoDB database name | No (defaults to "Training") |

## Testing Your Setup

After fixing the issues, test your setup:

1. **Check API Keys:**
   ```bash
   curl -X GET "http://localhost:5001/health"
   ```

2. **Test ML Models:**
   ```bash
   curl -X POST "http://localhost:5001/api/ml/status"
   ```

3. **Test LLM Functionality:**
   ```bash
   curl -X POST "http://localhost:5001/api/analyze" \
     -H "Content-Type: application/json" \
     -d '{"text": "test", "country": "USA"}'
   ```

## Getting Help

If you're still experiencing issues:

1. Check the logs: `docker-compose -f docker/docker-compose.dev.yml logs`
2. Verify your API keys are valid
3. Ensure all environment variables are set correctly
4. Check that MongoDB is accessible 
