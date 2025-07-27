#!/bin/bash

# Stop containers and remove volumes
docker compose -f docker-compose.dev.yml down -v

# Build and start containers
docker compose -f docker-compose.dev.yml up --build 
