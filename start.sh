#!/bin/bash
lsof -ti:5005 | xargs kill -9 2>/dev/null

# Start backend
cd src
python server.py &
BACKEND_PID=$!

# Start frontend (go back to root first)
cd ../src/modules/frontend
npx serve -s build -l 3000 &
FRONTEND_PID=$!

trap "kill $BACKEND_PID $FRONTEND_PID" EXIT
wait