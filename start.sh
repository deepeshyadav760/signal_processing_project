#!/bin/bash

echo "🚀 Starting AI Speech Intelligence MVP..."

# Kill any existing processes
pkill -f "uvicorn main:app" 2>/dev/null
pkill -f "http.server 8080" 2>/dev/null

# Start backend
cd backend
source venv/bin/activate
echo "🔧 Starting backend on port 8000..."
nohup python main.py > ../backend.log 2>&1 &
BACKEND_PID=$!
echo "Backend PID: $BACKEND_PID"

# Wait for backend
sleep 3

# Start frontend
cd ../frontend
echo "🌐 Starting frontend on port 8080..."
nohup python3 -m http.server 8080 > ../frontend.log 2>&1 &
FRONTEND_PID=$!
echo "Frontend PID: $FRONTEND_PID"

sleep 2

echo ""
echo "✅ MVP IS RUNNING!"
echo ""
echo "🌐 Frontend: http://localhost:8080"
echo "🔧 Backend:  http://localhost:8000"
echo "📚 API Docs: http://localhost:8000/docs"
echo ""
echo "📝 Logs:"
echo "   Backend:  tail -f backend.log"
echo "   Frontend: tail -f frontend.log"
echo ""
echo "🛑 To stop: pkill -f 'uvicorn\|http.server 8080'"
echo ""
