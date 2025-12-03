#!/bin/bash
set -e

# Determine Python command
PYRUN="python"
case "$(uname -s)" in
    Darwin|Linux)
        PYRUN="python3"
        ;;
esac

# Kill existing backend on port 5005
if command -v lsof >/dev/null 2>&1; then
    lsof -ti:5005 2>/dev/null | xargs kill -9 2>/dev/null || true
fi

# Activate venv
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f ".venv/Scripts/activate" ]; then
    source .venv/Scripts/activate
fi

# Start backend
cd src
"$PYRUN" server.py &
BACKEND_PID=$!

# Start frontend
cd ./modules/frontend
npx serve -s build -l 3000 &
FRONTEND_PID=$!

echo "===================================="
echo " Backend:  http://localhost:5005"
echo " Frontend: http://localhost:3000"
echo "===================================="

cleanup() {
    kill "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null || true
    SESSION_FILE=".session_state/auth_session.json"
    if [ -f "$SESSION_FILE" ]; then
        rm -f "$SESSION_FILE" && echo "Session state cleared."
    fi
}
trap cleanup EXIT
wait
