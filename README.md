# AI Fact-Checking System

Multi-module fact-checking system with LLM reasoning, vector search, and interactive UI.

## Overview
End-to-end pipeline for automated fact-checking:
- **Input Extraction** → Extracts verifiable claims from text
- **Vector Database** → Semantic search over evidence using Qdrant + E5 embeddings
- **Fact Validation** → NLI-based verification with scoring
- **LLM Reasoning** → Multi-step reasoning for explanations
- **Frontend UI** → React interface with verdict badges and citations

## Architecture
```
User Input → Claim Extraction → Vector DB → Fact Validator → Reasoning → UI Display
              (Danny)           (Adam)      (Sam)            (Akshay)    (Yuxiao)
```

## Quick Start

### Prerequisites
- Python 3.10+
- Node.js 16+
- OpenAI API key (if using OpenAI) OR Ollama installed (if using Ollama)

### Setup & Run
```bash
# 1. One-time setup (selects LLM provider, downloads models)
chmod +x setup.sh
./setup.sh

# Select LLM provider during setup:
#   1. OpenAI (gpt-4o-mini) - requires API key
#   2. Ollama (llama3.1) - runs locally, auto-downloads model

# 2. Configure API key (OpenAI only)
# Edit .env and set: OPENAI_API_KEY=sk-your-key-here

# 3. Start system
chmod +x start.sh
./start.sh
```

**First startup:** 5-15 minutes (embedding 145K claims)  
**Subsequent startups:** <2 seconds (uses cached vectors)

Open http://localhost:3000

## Setup Script Features

The setup script now handles:
- ✅ Automatic Ollama model pulling (`llama3.1`)
- ✅ Embeddings model caching (no re-downloads)
- ✅ FEVER dataset download with `trust_remote_code=True`
- ✅ Vector database persistence (no re-embedding)
- ✅ Service health checks

## Configuration

### Environment Variables (.env)
```bash
# LLM Provider (set by setup.sh)
LLM_PROVIDER=openai              # or: ollama

# OpenAI API Key (required if using OpenAI)
OPENAI_API_KEY=sk-your-key-here

# Embedding Model
EMBEDDING_MODEL=intfloat/e5-small-v2

# Data Directories
HF_HOME=./data/models            # Embedding model cache
QDRANT_LOCATION=./data/qdrant    # Vector database storage
```

### Toggle Reasoning
```bash
# Disable (faster, ~2s response)
curl -X POST http://localhost:5005/toggle-reasoning \
  -H "Content-Type: application/json" \
  -d '{"enable":false}'

# Enable (slower, ~6s, better explanations)
curl -X POST http://localhost:5005/toggle-reasoning \
  -H "Content-Type: application/json" \
  -d '{"enable":true}'
```

## Project Structure
```
.
├── src/
│   ├── modules/
│   │   ├── input_extraction/      # Claim extraction (Danny)
│   │   ├── misinformation_module/ # Vector DB + embeddings (Adam)
│   │   ├── claim_extraction/      # Fact validator (Sam)
│   │   ├── llm/                   # Reasoning engine (Akshay)
│   │   └── frontend/              # React UI (Yuxiao)
│   ├── pipeline.py                # Main integration
│   └── server.py                  # Flask API
├── data/
│   ├── models/                    # Embedding models (cached)
│   ├── qdrant/                    # Vector database storage
│   ├── mock.json                  # Small test dataset (2 entries)
│   └── fever.json                 # FEVER dataset (~145K claims)
├── requirements.txt
├── setup.sh
└── start.sh
```

## Datasets

### Mock (Default fallback)
- 2 entries for quick testing
- Used when fever.json not present

### FEVER (Primary)
```bash
# Download during setup.sh or manually:
cd data
python load_fever.py
```
- ~145K Wikipedia-based claims
- Downloads to `data/fever.json`
- Requires `trust_remote_code=True` for HuggingFace dataset

## API Endpoints

### Health Check
```bash
GET http://localhost:5005/health
```

### Fact Check
```bash
POST http://localhost:5005/chat
Content-Type: application/json

{
  "question": "The Moon landing happened in 1969"
}
```

Response:
```json
{
  "claim": "The Moon landing occurred in 1969",
  "verdict": "Supported",
  "score": 85,
  "explanation": "...",
  "citations": [...],
  "features": {...}
}
```

### Toggle Reasoning
```bash
POST http://localhost:5005/toggle-reasoning
Content-Type: application/json

{
  "enable": true
}
```

## Module Details

### Vector Database (Adam)
- **Embeddings**: `intfloat/e5-small-v2` (384-dim)
- **Storage**: Qdrant at `./data/qdrant`
- **Similarity**: Cosine distance
- **Top-K**: 20 passages retrieved per query
- **Persistence**: Vectors cached on disk, instant subsequent loads

### Fact Validator (Sam)
- **Method**: NLI (Natural Language Inference)
- **Features**: Entailment, contradiction, domain agreement, reliability
- **Scoring**: 0-100 confidence score
- **Verdicts**: Supported, Refuted, Contested, Not enough evidence
- **JSON Parsing**: Robust extraction from LLM responses with preambles

### Reasoning Engine (Akshay)
- **Models**: GPT-4o-mini (OpenAI) or Llama3.1 (Ollama)
- **Steps**: Understand → Decompose → Solve → Combine → Verify
- **Latency**: ~5-7 seconds for full reasoning
- **Fallback**: Simple explanation if reasoning fails

### Frontend (Yuxiao)
- **Framework**: React + Ant Design
- **Features**: Verdict badges, collapsible evidence, clickable citations
- **Build**: Static files served via `npx serve`

## Troubleshooting

### Setup Issues

#### FEVER Dataset Error: `trust_remote_code=True`
**Error:** `ValueError: trust_remote_code=True required`

**Fix:**
```bash
# Use provided fixed load_fever.py
cp fixes/load_fever.py data/
cd data
python load_fever.py
```

#### Ollama Model Not Found
**Error:** `model 'llama3.1' not found (status code: 404)`

**Fix:**
```bash
# Pull model manually
ollama pull llama3.1

# Or re-run setup (fixed version auto-pulls)
./setup.sh
```

#### Embeddings Model Re-downloads Every Time
**Symptom:** Setup always downloads embedding model despite cache existing

**Fix:** Replace `setup.sh` with fixed version (uses double-dash format for cache check)

### Runtime Issues

#### Backend Hangs on Startup
**Symptom:** Server shows "Pipeline initialized" but never prints "Server URL"

**Cause:** Embedding 145K claims takes 5-15 minutes on first run

**Fix:**
```bash
# Option 1: Use small test dataset
cp data/mock.json data/fever.json
./start.sh

# Option 2: Wait for full load (one-time only)
# Watch for progress messages:
# "Generating embeddings..."
# "Processed 10000/145449 points..."
```

#### Knowledge Base Re-embeds Every Startup
**Symptom:** Every restart takes 5-15 minutes

**Cause:** Pipeline calls `reset_collection()` instead of `ensure_collection()`

**Fix:** Replace `src/pipeline.py` and `src/modules/misinformation_module/src/qdrant_db.py` with fixed versions

#### Evidence Analysis Shows 0.00 for All Values
**Symptom:** Features show:
```
Max Entailment: 0.00
Max Contradiction: 0.00
Agreeing Domains: 0
```

**Cause:** LLM returns JSON wrapped in text preamble/explanation, JSON parser fails

**Fix:** Replace `src/modules/claim_extraction/fact_validator.py` with fixed version (robust JSON extraction)

#### Double Loading/Execution
**Symptom:** Two loading indicators appear or backend logs show duplicate processing

**Potential Causes:**
1. React StrictMode (intentional double-render in development)
2. Health check + actual request appearing as duplicates
3. Component re-render triggering duplicate API calls

**Check:**
```bash
# Watch server logs - should only see one POST per submission
# Look for duplicate timestamp entries:
# 127.0.0.1 - - [timestamp] "POST /chat HTTP/1.1" 200
```

**Fix if needed:**
- Disable React StrictMode in frontend (development only)
- Add request deduplication in backend
- Check for useEffect dependencies causing re-renders

### Performance

#### Slow Response Times (>30s)
**Cause:** Reasoning enabled (5 LLM calls)

**Fix:**
```bash
# Disable reasoning for faster responses
curl -X POST http://localhost:5005/toggle-reasoning \
  -d '{"enable":false}'
```

#### Qdrant Warning: Large Collection
**Warning:** `Local mode not recommended for 116K+ points`

**Impact:** Minor performance degradation, still functional

**Fix (optional):**
```bash
# Use Qdrant Docker for better performance
docker run -p 6333:6333 qdrant/qdrant

# Update .env:
QDRANT_LOCATION=http://localhost:6333
```

## Manual Setup

### Backend
```bash
# Install Python dependencies
pip install -r requirements.txt --break-system-packages

# Download dataset
cd data
python load_fever.py
cd ..
```

### Frontend
```bash
cd src/modules/frontend
npm install
npm run build
cd ../../..
```

### LLM Setup

**OpenAI:**
- Add API key to `.env`
- No additional setup needed

**Ollama:**
```bash
# Install Ollama from https://ollama.com
# Pull model
ollama pull llama3.1

# Verify
ollama list | grep llama3.1
```

## Development Mode

```bash
# Backend (hot reload)
cd src
python server.py &

# Frontend (hot reload)
cd src/modules/frontend
npm start
```

Open http://localhost:3000 (frontend dev server) or http://localhost:5005 (API only)

## Testing

### Quick Test
```bash
# Health check
curl http://localhost:5005/health

# Fact check
curl -X POST http://localhost:5005/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"The Moon landing happened in 1969"}'
```

### Verify Setup
- [ ] Backend starts without errors
- [ ] Knowledge base loads (instantly on 2nd+ run)
- [ ] Health endpoint responds
- [ ] Frontend loads at http://localhost:3000
- [ ] Test query returns valid result
- [ ] Evidence analysis shows non-zero values
- [ ] Citations appear with links

## Known Issues & Fixes

All major setup issues have been identified and fixed. Updated files provided:

1. **setup.sh** - Ollama model pulling, embeddings cache check
2. **load_fever.py** - `trust_remote_code=True` for FEVER dataset
3. **pipeline.py** - Knowledge base persistence, progress logging
4. **qdrant_db.py** - Collection existence checking
5. **fact_validator.py** - Robust JSON extraction from LLM responses

See `COMPLETE_FIX_SUMMARY.md` for detailed breakdown of all fixes.

## Team
- **Input Extraction**: Danny
- **Vector Database**: Adam
- **Fact Validation**: Sam
- **LLM Reasoning**: Akshay
- **Frontend UI**: Yuxiao
- **Integration**: Stephen

## License
MIT