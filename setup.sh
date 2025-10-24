#!/bin/sh
set -e

echo "=== Fact-Checking System Setup ==="

# Check prerequisites
command -v python3 >/dev/null 2>&1 || { echo "Python 3 required"; exit 1; }
command -v node >/dev/null 2>&1 || { echo "Node.js required"; exit 1; }
command -v npm >/dev/null 2>&1 || { echo "npm required"; exit 1; }

# Install Python deps
echo "Installing Python dependencies..."
pip install -r requirements.txt --break-system-packages

# Install Node deps & build frontend
echo "Installing frontend dependencies..."
cd src/modules/frontend
npm install
echo "Building frontend..."
npm run build
cd ../../..

# Create .env if missing
if [ ! -f .env ]; then
    echo "Creating .env file..."
    echo "OPENAI_API_KEY=your_key_here" > .env
    echo "Edit .env with your OpenAI API key"
fi

# Download dataset (optional)
printf "Download FEVER dataset? (y/n): "
read -r reply
case "$reply" in
    [Yy]*) python3 load_fever.py ;;
    *) echo "Skipping dataset download" ;;
esac

echo "Setup complete! Run ./start.sh to start"