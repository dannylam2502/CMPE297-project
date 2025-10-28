#!/bin/sh
set -e

echo "=== Fact-Checking System Setup ==="

command -v python3 >/dev/null 2>&1 || { echo "Python 3 required"; exit 1; }
command -v node >/dev/null 2>&1 || { echo "Node.js required"; exit 1; }
command -v npm >/dev/null 2>&1 || { echo "npm required"; exit 1; }

echo "Installing Python dependencies..."
pip install -r requirements.txt --break-system-packages

echo "Installing frontend dependencies..."
npm install --prefix src/modules/frontend
echo "Building frontend..."
npm run build --prefix src/modules/frontend

if [ ! -f .env ]; then
    echo "Creating .env file..."
    echo "OPENAI_API_KEY=your_key_here" > .env
    echo "Edit .env with your OpenAI API key"
fi

printf "Download FEVER dataset? (y/n): "
read -r reply
case "$reply" in
    [Yy]*) python3 data/load_fever.py ;;
    *) echo "Skipping dataset download" ;;
esac

echo "Setup complete. Run ./start.sh to start"