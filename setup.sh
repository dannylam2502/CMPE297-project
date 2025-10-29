#!/bin/sh
set -e

echo "=== Fact-Checking System Setup ==="

command -v python3 >/dev/null 2>&1 || { echo "Python 3 required"; exit 1; }
command -v node >/dev/null 2>&1 || { echo "Node.js required"; exit 1; }
command -v npm >/dev/null 2>&1 || { echo "npm required"; exit 1; }

echo "Checking Python dependencies..."
pip install -q -r requirements.txt --break-system-packages

# Check if frontend build already exists
if [ -d "src/modules/frontend/build" ] && [ "$(ls -A src/modules/frontend/build)" ]; then
    printf "Frontend build exists. Rebuild? (y/n): "
    read -r rebuild
    case "$rebuild" in
        [Yy]*)
            cd src/modules/frontEnd
            echo "Installing frontend dependencies..."
            npm install
            echo "Building frontend..."
            npm run build
            cd ../../../.. 
            ;;
        *)
            echo "Skipping frontend build"
            ;;
    esac
else
	cd src/modules/frontEnd
    echo "Installing frontend dependencies..."
    npm install
    echo "Building frontend..."
    npm run build
    cd ../../../.. 
fi

if [ ! -f .env ]; then
    echo "Creating .env file..."
    echo "OPENAI_API_KEY=your_key_here" > .env
    echo "EMBEDDING_MODEL=intfloat/e5-small-v2" >> .env
    echo "HF_HOME=./data/models" >> .env
fi

# Check if embedding model is set, add if missing
if ! grep -q "^EMBEDDING_MODEL=" .env 2>/dev/null; then
    echo "EMBEDDING_MODEL=intfloat/e5-small-v2" >> .env
fi

# Check if HF_HOME is set, add if missing
if ! grep -q "^HF_HOME=" .env 2>/dev/null; then
    # Use absolute path to ensure consistency regardless of where script is run from
    SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
    echo "HF_HOME=${SCRIPT_DIR}/data/models" >> .env
fi

# Create data subdirectories
mkdir -p data/models
mkdir -p data/qdrant
mkdir -p data/cache
mkdir -p data/checkpoints

# Check if LLM provider already set
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
    echo "  2. Ollama (llama3.1, local)"
    printf "Choice [1-2]: "
    read -r llm_choice

    # Remove existing LLM_PROVIDER line if present
    sed -i.bak '/^LLM_PROVIDER=/d' .env 2>/dev/null || true

    case "$llm_choice" in
        1)
            echo "LLM_PROVIDER=openai" >> .env
            echo "Selected: OpenAI"
            echo "Edit .env with your OpenAI API key"
            ;;
        2)
            echo "LLM_PROVIDER=ollama" >> .env
            echo "Selected: Ollama"
            
            # Check if ollama is installed
            if ! command -v ollama >/dev/null 2>&1; then
                echo "Error: ollama not found. Install from https://ollama.com"
                echo "Defaulting to OpenAI instead"
                sed -i.bak 's/LLM_PROVIDER=ollama/LLM_PROVIDER=openai/' .env
                exit 1
            fi
            
            # Start ollama service if not running
            echo "Checking ollama service..."
            if ! ollama list >/dev/null 2>&1; then
                echo "Starting ollama service..."
                ollama serve >/dev/null 2>&1 &
                sleep 3
            fi
            
            # Check if llama3.1 model exists
            if ollama list | grep -q "llama3.1"; then
                echo "Model llama3.1 already installed"
            else
                echo "Pulling llama3.1 model (this may take several minutes)..."
                ollama pull llama3.1
                echo "Model download complete"
            fi
            
            # Verify model is available
            if ! ollama list | grep -q "llama3.1"; then
                echo "Error: Failed to install llama3.1 model"
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

# Verify existing provider setup (runs whether provider changed or not)
if grep -q "^LLM_PROVIDER=" .env 2>/dev/null; then
    CURRENT_PROVIDER=$(grep "^LLM_PROVIDER=" .env | cut -d'=' -f2)
    
    if [ "$CURRENT_PROVIDER" = "ollama" ]; then
        echo ""
        echo "Verifying Ollama setup..."
        
        # Check if ollama is installed
        if ! command -v ollama >/dev/null 2>&1; then
            echo "Error: ollama not found but LLM_PROVIDER=ollama"
            echo "Install from https://ollama.com or run setup again to switch to OpenAI"
            exit 1
        fi
        
        # Start ollama service if not running
        if ! ollama list >/dev/null 2>&1; then
            echo "Starting ollama service..."
            ollama serve >/dev/null 2>&1 &
            sleep 3
        fi
        
        # Check if llama3.1 model exists
        if ollama list | grep -q "llama3.1"; then
            echo "âœ“ Ollama service running"
            echo "âœ“ Model llama3.1 installed"
        else
            echo "Model llama3.1 not found. Pulling now..."
            ollama pull llama3.1
            if ollama list | grep -q "llama3.1"; then
                echo "âœ“ Model llama3.1 installed"
            else
                echo "Error: Failed to install llama3.1 model"
                exit 1
            fi
        fi
        
    elif [ "$CURRENT_PROVIDER" = "openai" ]; then
        echo ""
        echo "Verifying OpenAI setup..."
        if grep -q "^OPENAI_API_KEY=sk-" .env 2>/dev/null; then
            echo "âœ“ OpenAI API key configured"
        else
            echo "Warning: OPENAI_API_KEY not set or invalid"
            echo "Edit .env and set: OPENAI_API_KEY=sk-your-key-here"
        fi
    fi
fi

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

printf "Download FEVER dataset? (y/n): "
read -r reply
case "$reply" in
    [Yy]*) python3 data/load_fever.py ;;
    *) echo "Skipping dataset download" ;;
esac

echo "Setup complete. Run ./start.sh to start"