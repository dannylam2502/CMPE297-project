#!/bin/bash

# Kill any existing backend on port 5005
lsof -ti:5005 | xargs kill -9 2>/dev/null

# Activate Python venv if exists
if [ -f "../.venv/Scripts/activate" ]; then
    source ../.venv/Scripts/activate
fi

# Start backend
cd src
python server.py &
BACKEND_PID=$!

# Start frontend
cd ../modules/frontend
npx serve -s build -l 3000 &
FRONTEND_PID=$!

echo "===================================="
echo " Backend:  http://localhost:5005"
echo " Frontend: http://localhost:3000"
echo "===================================="

trap "kill $BACKEND_PID $FRONTEND_PID" EXIT
wait
