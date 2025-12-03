#!/bin/sh
set -e

# Auto-detect Python
if [ -z "$PYTHON" ]; then
    case "$(uname -s)" in
        Darwin|Linux)
            PYTHON="$(command -v python3 || command -v python || true)"
            ;;
        *)
            # Windows/MINGW fallback
            PYTHON="${PYTHON:-python}"
            ;;
    esac
fi

echo "=== Fact-Checking System Setup (Cloud Edition) ==="

# -------------------------------------------------------------
# REQUIREMENTS
# -------------------------------------------------------------
command -v "$PYTHON" >/dev/null 2>&1 || { echo "Python 3 required"; exit 1; }
command -v node >/dev/null 2>&1 || { echo "Node.js required"; exit 1; }
command -v npm >/dev/null 2>&1 || { echo "npm required"; exit 1; }

# -------------------------------------------------------------
# PYTHON VENV
# -------------------------------------------------------------
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    "$PYTHON" -m venv .venv
fi

echo "Activating virtual environment..."
if [ -f ".venv/bin/activate" ]; then
    . .venv/bin/activate
elif [ -f ".venv/Scripts/activate" ]; then
    . .venv/Scripts/activate
else
    echo "ERROR: Activation script not found in .venv/bin/ or .venv/Scripts/"
    exit 1
fi

echo "Installing Python dependencies..."
pip install -r requirements.txt 2>&1 | grep -v "Requirement already satisfied" || true

# -------------------------------------------------------------
# FRONTEND BUILD
# -------------------------------------------------------------
if [ -d "src/modules/frontend/build" ] && [ "$(ls -A src/modules/frontend/build 2>/dev/null)" ]; then
    printf "Frontend already built. Rebuild? (y/n): "
    read -r rebuild
    case "$rebuild" in
        [Yy]*)
            (cd src/modules/frontend && npm install && npm run build)
            ;;
        *)
            echo "Skipping rebuild."
            ;;
    esac
else
    (cd src/modules/frontend && npm install && npm run build)
fi

# -------------------------------------------------------------
# ENV FILE SETUP
# -------------------------------------------------------------
if [ ! -f .env ]; then
    echo "Creating .env..."
    cat > .env << 'ENVEOF'
OPENAI_API_KEY=
LLM_PROVIDER=openai
EMBEDDING_MODEL=intfloat/e5-small-v2
HF_HOME=./data/models
QDRANT_URL=
QDRANT_API_KEY=
ENVEOF
fi

echo "Using .env from project root."
echo "Make sure QDRANT_URL and QDRANT_API_KEY are set!"

# -------------------------------------------------------------
# VERIFY CLOUD QDRANT CONFIG
# -------------------------------------------------------------
QDRANT_URL=$(grep "^QDRANT_URL=" .env | cut -d'=' -f2)
QDRANT_API_KEY=$(grep "^QDRANT_API_KEY=" .env | cut -d'=' -f2)

if [ -z "$QDRANT_URL" ] || [ -z "$QDRANT_API_KEY" ]; then
    echo ""
    echo "ERROR: QDRANT_URL or QDRANT_API_KEY missing in .env"
    echo "Please open .env and set your Qdrant Cloud credentials."
    exit 1
fi

echo "Qdrant Cloud URL: $QDRANT_URL"
echo "Qdrant Cloud API Key found"

# -------------------------------------------------------------
# EMBEDDING MODEL
# -------------------------------------------------------------
echo ""
echo "Checking embeddings model cache..."
EMBEDDING_MODEL=$(grep "^EMBEDDING_MODEL=" .env | cut -d'=' -f2)
export HF_HOME=$(grep "^HF_HOME=" .env | cut -d'=' -f2)

"$PYTHON" - <<EOF
from sentence_transformers import SentenceTransformer
print("Loading embedding model ($EMBEDDING_MODEL)...")
SentenceTransformer("$EMBEDDING_MODEL")
print("Embedding model available")
EOF

# -------------------------------------------------------------
# CLOUD INGESTION (OPTIONAL)
# -------------------------------------------------------------
echo ""
echo "Your ingestion script pushes data directly to Qdrant Cloud."
echo "To run it now: python src/ingest_news_to_qdrant.py"

printf "Run ingestion now? (y/n): "
read -r ingest_now
case "$ingest_now" in
    [Yy]*)
        "$PYTHON" src/ingest_news_to_qdrant.py
        ;;
esac

# -------------------------------------------------------------
# DONE
# -------------------------------------------------------------
echo ""
echo "Setup Complete (Cloud Mode)"
echo "Run the backend with:   ./start.sh"
echo "Run the frontend with:  open build folder or npm serve"