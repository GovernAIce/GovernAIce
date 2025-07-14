#!/bin/bash

# Stop containers and remove volumes
docker compose -f docker-compose.local.yml down -v

# Build and start containers
docker compose -f docker-compose.local.yml up --build 
