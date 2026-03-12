#!/bin/bash
# Quick Start Script for Veneer App
# This starts both backend and frontend

set -e

echo "================================================"
echo "  Veneer Vision AI - Quick Start"
echo "================================================"
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if we're in the right directory
if [ ! -f "services/veneer-preview/api_server.py" ]; then
    echo "Error: Please run this script from the VeneerLoungeApp root directory"
    exit 1
fi

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "Shutting down services..."
    kill $BACKEND_PID 2>/dev/null || true
    kill $FRONTEND_PID 2>/dev/null || true
    exit 0
}

trap cleanup SIGINT SIGTERM

mkdir -p logs
echo "Logs will be saved to:"
echo "  Backend:  logs/backend.log"  
echo "  Frontend: logs/frontend.log"
echo ""


echo -e "${BLUE}[1/2] Starting Backend API Server...${NC}"
echo "      Using pretrained ControlNet from Hugging Face"
echo "      Port: 8000"
echo ""

cd services/veneer-preview

# Activate virtual environment
if [ -f "../../ext/veneer_generation/venv/bin/activate" ]; then
    source ../../ext/veneer_generation/venv/bin/activate
else
    echo "Warning: Virtual environment not found. Using system Python."
fi

# Start backend in background
VENEER_MODEL_TYPE=sdxl python api_server.py --model sdxl --port 8000 > ../../logs/backend.log 2>&1 &
BACKEND_PID=$!

echo -e "${GREEN}✓ Backend starting (PID: $BACKEND_PID)${NC}"
echo ""

# Wait for backend to be ready
echo "Waiting for backend to be ready..."
sleep 5

# Check if backend is running
if ! kill -0 $BACKEND_PID 2>/dev/null; then
    echo "Error: Backend failed to start. Check logs above."
    exit 1
fi

# Test backend health
for i in {1..10}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Backend is ready!${NC}"
        break
    fi
    if [ $i -eq 10 ]; then
        echo "Warning: Backend health check failed, but continuing..."
    fi
    sleep 2
done

echo ""

# Start Frontend
echo -e "${BLUE}[2/2] Starting Frontend...${NC}"
echo "      Port: 3000"
echo ""

cd ../../VeneerApp-main

# Check if .env.local exists
if [ ! -f ".env.local" ]; then
    echo "Creating .env.local..."
    echo "BACKEND_URL=http://localhost:8000" > .env.local
fi

# Install dependencies if needed
if [ ! -d "node_modules" ]; then
    echo "Installing frontend dependencies (this may take a few minutes)..."
    npm install
fi

# Start frontend in background
npm run dev > ../logs/frontend.log 2>&1 &
FRONTEND_PID=$!

echo -e "${GREEN}✓ Frontend starting (PID: $FRONTEND_PID)${NC}"
echo ""

# Wait a bit for frontend to start
sleep 3

echo "================================================"
echo -e "${GREEN}✓ Veneer Vision AI is Running!${NC}"
echo "================================================"
echo ""
echo "  Backend API:  http://localhost:8000"
echo "  Frontend App: http://localhost:3000"
echo ""
echo "  Health Check: curl http://localhost:8000/health"
echo ""
echo "  Press Ctrl+C to stop both services"
echo ""
echo "================================================"

# Keep script running
wait $BACKEND_PID $FRONTEND_PID
