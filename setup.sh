#!/bin/sh
set -e

echo "=== Fact-Checking System Setup ==="

# -------------------------------------------------------------
# REQUIREMENTS
# -------------------------------------------------------------
command -v python3 >/dev/null 2>&1 || { echo "Python 3 required"; exit 1; }
command -v node >/dev/null 2>&1 || { echo "Node.js required"; exit 1; }
command -v npm >/dev/null 2>&1 || { echo "npm required"; exit 1; }

echo "Checking Python dependencies..."
pip install -r requirements.txt --break-system-packages 2>&1 | grep -v "Requirement already satisfied" || true

# -------------------------------------------------------------
# FRONTEND BUILD
# -------------------------------------------------------------
if [ -d "src/modules/frontend/build" ] && [ "$(ls -A src/modules/frontend/build)" ]; then
    printf "Frontend build exists. Rebuild? (y/n): "
    read -r rebuild
    case "$rebuild" in
        [Yy]*)
            echo "Installing frontend dependencies..."
            (cd src/modules/frontend && npm install)
            echo "Building frontend..."
            (cd src/modules/frontend && npm run build)
            ;;
        *)
            echo "Skipping frontend build"
            ;;
    esac
else
    echo "Installing frontend dependencies..."
    (cd src/modules/frontend && npm install)
    echo "Building frontend..."
    (cd src/modules/frontend && npm run build)
fi

# -------------------------------------------------------------
# ENV FILE SETUP
# -------------------------------------------------------------
if [ ! -f .env ]; then
    echo "Creating .env file..."
    echo "OPENAI_API_KEY=your_key_here" > .env
    echo "EMBEDDING_MODEL=intfloat/e5-small-v2" >> .env
    echo "HF_HOME=./data/models" >> .env
fi

# -------------------------------------------------------------
# DIRECTORIES
# -------------------------------------------------------------
mkdir -p data/models data/qdrant data/cache data/checkpoints

# -------------------------------------------------------------
# LLM PROVIDER SELECTION
# -------------------------------------------------------------
if grep -q "^LLM_PROVIDER=" .env 2>/dev/null; then
    current_provider=$(grep "^LLM_PROVIDER=" .env | cut -d'=' -f2)
    echo ""
    echo "Current LLM provider: $current_provider"
    printf "Change provider? (y/n): "
    read -r change_llm
    case "$change_llm" in
        [Yy]*) ask_llm=true ;;
        *) ask_llm=false ;;
    esac
else
    ask_llm=true
fi

if [ "$ask_llm" = true ]; then
    echo ""
    echo "Select LLM provider:"
    echo "  1. OpenAI (gpt-4o-mini)"
    echo "  2. Ollama (mistral, local)"
    printf "Choice [1-2]: "
    read -r llm_choice

    # Remove existing line
    sed -i.bak '/^LLM_PROVIDER=/d' .env 2>/dev/null || true

    case "$llm_choice" in
        1)
            echo "LLM_PROVIDER=openai" >> .env
            echo "Selected: OpenAI"
            echo "Edit .env with your OpenAI API key"
            ;;
        2)
            echo "LLM_PROVIDER=ollama" >> .env
            echo "Selected: Ollama (Mistral)"
            
            if ! command -v ollama >/dev/null 2>&1; then
                echo "Error: ollama not found. Install from https://ollama.com"
                echo "Defaulting to OpenAI instead"
                sed -i.bak 's/LLM_PROVIDER=ollama/LLM_PROVIDER=openai/' .env
                exit 1
            fi
            
            echo "Checking Ollama service..."
            if ! ollama list >/dev/null 2>&1; then
                echo "Starting Ollama service..."
                ollama serve >/dev/null 2>&1 &
                sleep 3
            fi

            # Pull Mistral model
            if ollama list | grep -q "mistral"; then
                echo "Model mistral already installed"
            else
                echo "Pulling mistral model (this may take several minutes)..."
                ollama pull mistral
                echo "Model download complete"
            fi

            if ! ollama list | grep -q "mistral"; then
                echo "Error: Failed to install mistral model"
                exit 1
            fi

            echo "Ollama setup complete"
            ;;
        *)
            echo "Invalid choice. Defaulting to OpenAI"
            echo "LLM_PROVIDER=openai" >> .env
            ;;
    esac
fi

# -------------------------------------------------------------
# VERIFY PROVIDER SETUP
# -------------------------------------------------------------
if grep -q "^LLM_PROVIDER=" .env 2>/dev/null; then
    CURRENT_PROVIDER=$(grep "^LLM_PROVIDER=" .env | cut -d'=' -f2)
    
    if [ "$CURRENT_PROVIDER" = "ollama" ]; then
        echo ""
        echo "Verifying Ollama setup..."
        if ! command -v ollama >/dev/null 2>&1; then
            echo "Error: ollama not found but LLM_PROVIDER=ollama"
            echo "Install from https://ollama.com or rerun setup to switch providers"
            exit 1
        fi

        if ! ollama list >/dev/null 2>&1; then
            echo "Starting Ollama service..."
            ollama serve >/dev/null 2>&1 &
            sleep 3
        fi

        if ollama list | grep -q "mistral"; then
            echo "✓ Ollama service running"
            echo "✓ Model mistral installed"
        else
            echo "Model mistral not found. Pulling now..."
            ollama pull mistral
            if ollama list | grep -q "mistral"; then
                echo "✓ Model mistral installed"
            else
                echo "Error: Failed to install mistral model"
                exit 1
            fi
        fi
    elif [ "$CURRENT_PROVIDER" = "openai" ]; then
        echo ""
        echo "Verifying OpenAI setup..."
        if grep -q "^OPENAI_API_KEY=sk-" .env 2>/dev/null; then
            echo "✓ OpenAI API key configured"
        else
            echo "Warning: OPENAI_API_KEY not set or invalid"
            echo "Edit .env and set: OPENAI_API_KEY=sk-your-key-here"
        fi
    fi
fi

# -------------------------------------------------------------
# EMBEDDINGS MODEL CHECK
# -------------------------------------------------------------
echo ""
echo "Checking embeddings model cache..."
EMBEDDING_MODEL=$(grep "^EMBEDDING_MODEL=" .env | cut -d'=' -f2)
HF_HOME_PATH=$(grep "^HF_HOME=" .env | cut -d'=' -f2)
export HF_HOME="$HF_HOME_PATH"
MODEL_CACHE_DIR=$(echo "$EMBEDDING_MODEL" | sed 's/\/\/*/--/g')
if [ -d "$HF_HOME/hub/models--$MODEL_CACHE_DIR" ]; then
    echo "Embeddings model ($EMBEDDING_MODEL) already cached"
else
    echo "Downloading embeddings model ($EMBEDDING_MODEL) - this will take a few minutes..."
    python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('$EMBEDDING_MODEL')"
    echo "Download complete"
fi

# -------------------------------------------------------------
# OPTIONAL FEVER DATASET
# -------------------------------------------------------------
printf "Download FEVER dataset? (y/n): "
read -r reply
case "$reply" in
    [Yy]*) python3 data/load_fever.py ;;
    *) echo "Skipping dataset download" ;;
esac

echo "Setup complete. Run ./start.sh to start"
