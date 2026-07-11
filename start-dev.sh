#!/bin/bash
echo "[RAG Project] Tearing down old instances..."
docker-compose down

echo "[RAG Project] Starting up optimized stack..."
docker-compose up --build