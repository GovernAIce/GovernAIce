# GovernAIce Docker Setup Guide

## 🐳 Overview

This guide explains how to run GovernAIce using Docker containers. The Docker setup includes the backend API, frontend application, and MongoDB database.

## 📋 Prerequisites

- **Docker**: Install Docker Desktop or Docker Engine
- **Docker Compose**: Usually included with Docker Desktop
- **Git**: To clone the repository

## 🚀 Quick Start

### Option 1: Using the Setup Script (Recommended)

```bash
# Navigate to the docker directory
cd docker

# Run the comprehensive setup script
./setup-docker.sh dev     # For development environment
./setup-docker.sh local   # For local environment
```

### Option 2: Manual Setup

```bash
# Navigate to the docker directory
cd docker

# Start development environment
./dev-docker.sh

# Or start local environment
./local-docker.sh
```

## 🏗️ Docker Architecture

```
GovernAIce Docker Setup
├── 📦 Backend Container (Python/Flask)
│   ├── Flask API server
│   ├── ML pipeline modules
│   └── Environment configuration
├── 📦 Frontend Container (Node.js/React)
│   ├── React application
│   ├── TypeScript compilation
│   └── Vite development server
└── 📦 MongoDB Container
    ├── Policy document storage
    └── Vector search capabilities
```

## 📁 Docker Files Structure

```
docker/
├── 📄 Dockerfile.dev          # Development backend container
├── 📄 Dockerfile.local        # Local backend container
├── 📄 Dockerfile.frontend     # Frontend container
├── 📄 docker-compose.dev.yml  # Development environment
├── 📄 docker-compose.local.yml # Local environment
├── 📄 setup-docker.sh         # Comprehensive setup script
├── 📄 dev-docker.sh           # Development startup script
└── 📄 local-docker.sh         # Local startup script
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the root directory:

```bash
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
```

### Port Configuration

- **Backend API**: `http://localhost:5001`
- **Frontend**: `http://localhost:5173`
- **MongoDB**: `localhost:27017`

## 🛠️ Available Commands

### Setup Script Commands

```bash
# Development environment
./setup-docker.sh dev

# Local environment
./setup-docker.sh local

# Show logs
./setup-docker.sh logs

# Stop containers
./setup-docker.sh stop

# Clean up Docker resources
./setup-docker.sh cleanup

# Show help
./setup-docker.sh help
```

### Manual Docker Commands

```bash
# Start development environment
docker compose -f docker-compose.dev.yml up --build

# Start local environment
docker compose -f docker-compose.local.yml up --build

# Stop containers
docker compose -f docker-compose.dev.yml down -v

# View logs
docker compose -f docker-compose.dev.yml logs -f

# Rebuild containers
docker compose -f docker-compose.dev.yml up --build --force-recreate
```

## 🔍 Troubleshooting

### Common Issues

1. **Port Already in Use**
   ```bash
   # Check what's using the port
   lsof -i :5001
   lsof -i :5173
   
   # Stop the process or change ports in docker-compose files
   ```

2. **Build Failures**
   ```bash
   # Clean Docker cache
   docker system prune -f
   
   # Rebuild without cache
   docker compose -f docker-compose.dev.yml build --no-cache
   ```

3. **Environment Variables Not Loading**
   ```bash
   # Check if .env file exists
   ls -la .env
   
   # Verify environment variables in container
   docker exec -it governaice-backend-1 env
   ```

4. **MongoDB Connection Issues**
   ```bash
   # Check MongoDB container status
   docker ps | grep mongo
   
   # View MongoDB logs
   docker logs governaice-mongodb-1
   ```

### Debug Commands

```bash
# Check container status
docker ps

# View container logs
docker logs <container_name>

# Access container shell
docker exec -it <container_name> /bin/bash

# Check network connectivity
docker network ls
docker network inspect governaice_app-network
```

## 📊 Monitoring

### Container Health Checks

```bash
# Check all containers
docker ps

# Monitor resource usage
docker stats

# View container logs
docker logs -f governaice-backend-1
docker logs -f governaice-frontend-1
docker logs -f governaice-mongodb-1
```

### Application Health

- **Backend API**: `http://localhost:5001/health` (if implemented)
- **Frontend**: `http://localhost:5173`
- **MongoDB**: Connect via MongoDB Compass to `localhost:27017`

## 🔒 Security Considerations

1. **Environment Variables**: Never commit `.env` files to git
2. **API Keys**: Use environment variables for all sensitive data
3. **Network Security**: Containers are isolated in Docker network
4. **Volume Mounts**: Only necessary directories are mounted

## 🚀 Production Deployment

For production deployment, consider:

1. **Use production Dockerfiles** (not included in this setup)
2. **Configure proper environment variables**
3. **Set up reverse proxy (nginx)**
4. **Use external MongoDB instance**
5. **Implement health checks**
6. **Set up monitoring and logging**

## 📚 Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [MongoDB Docker Image](https://hub.docker.com/_/mongo)
- [Node.js Docker Image](https://hub.docker.com/_/node)
- [Python Docker Image](https://hub.docker.com/_/python)

## 🤝 Support

If you encounter issues:

1. Check the troubleshooting section above
2. Review container logs: `docker logs <container_name>`
3. Verify environment variables are set correctly
4. Ensure Docker and Docker Compose are up to date
5. Check the [GitHub Issues](https://github.com/GovernAIce/GovernAIce/issues) for known problems

---

**Note**: This Docker setup is optimized for development and local testing. For production deployment, additional security and performance configurations should be implemented. 
