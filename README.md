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
- OpenAI API key

### Setup & Run
```bash
# 1. One-time setup
chmod +x setup.sh
./setup.sh

# 2. Configure API key
echo "OPENAI_API_KEY=sk-your-key-here" > .env

# 3. Start system
chmod +x start.sh
./start.sh
```

Open http://localhost:3000

## Manual Setup

### Backend
```bash
# Install Python dependencies
pip install -r requirements.txt --break-system-packages

# Load dataset (optional) ? where is this
python load_fever.py
```

### Frontend
```bash
cd src/modules/frontend
npm install
npm run build
cd ../../..
```

## Running the System

### Development Mode (hot reload)
```bash
cd src
python server.py &

cd modules/frontend
npm start
```

### Production Mode (pre-built)
```bash
./start.sh
```

## Configuration

### Environment Variables
- `OPENAI_API_KEY` - Required for reasoning engine

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

### Dataset Selection
Edit `server.py` line 24:
```python
KNOWLEDGE_BASE_PATH = "data/fever.json"  # or data/mock.json
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
│   ├── mock.json                  # Small test dataset
│   └── fever.json                 # FEVER dataset (10K claims)
├── requirements.txt
├── setup.sh
└── start.sh
```

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

## Module Details

### Vector Database (Adam)
- **Embeddings**: `intfloat/e5-small-v2` (384-dim)
- **Storage**: Qdrant (local disk at `./qdrant_data`)
- **Similarity**: Cosine distance
- **Top-K**: 20 passages retrieved per query

### Fact Validator (Sam)
- **Method**: NLI (Natural Language Inference)
- **Features**: Entailment, contradiction, domain agreement, reliability
- **Scoring**: 0-100 confidence score
- **Verdicts**: Supported, Refuted, Contested, Not enough evidence

### Reasoning Engine (Akshay)
- **Model**: GPT-4o-mini
- **Steps**: Understand → Decompose → Solve → Combine → Verify
- **Latency**: ~5-7 seconds for full reasoning
- **Fallback**: Simple explanation if reasoning fails

### Frontend (Yuxiao)
- **Framework**: React + Ant Design
- **Features**: Verdict badges, collapsible evidence, clickable citations
- **Build**: Static files served via `npx serve`

## Datasets

### Mock (Default)
- 2 entries for quick testing
- Pre-loaded on startup

### FEVER (Recommended)
```bash
python load_fever.py  # Downloads 10K claims
```
- 185K total claims available
- Wikipedia-based evidence
- Standard benchmark

## Troubleshooting

### Backend won't start
```bash
# Check dependencies
pip list | grep -E "flask|qdrant|openai"

# Check API key
grep OPENAI_API_KEY .env
```

### Frontend shows 404
```bash
# Rebuild
cd src/modules/frontend
npm run build
```

### "Not enough evidence" for all queries
- Knowledge base too small (only 2 entries)
- Run `python load_fever.py` to load more data

### Slow responses (>30s)
- Reasoning enabled (5 API calls)
- Disable via `/toggle-reasoning` endpoint

## Development

### Adding a Dataset
```python
# my_loader.py
import json

data = [
    {"id": i, "claim": "...", "source": "...", "confidence": 1.0}
    for i, claim in enumerate(my_claims)
]

with open("data/my_dataset.json", "w") as f:
    json.dump(data, f)
```

### Testing Backend
```bash
python outputs/test_integration.py
```

## Team
- **Input Extraction**: Danny
- **Vector Database**: Adam
- **Fact Validation**: Sam
- **LLM Reasoning**: Akshay
- **Frontend UI**: Yuxiao
- **Integration**: Stephen

## License
MIT