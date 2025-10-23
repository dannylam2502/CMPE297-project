# Misinformation LLM — Vector Database + Embeddings Module

## Overview
This module builds the retrieval backbone for a misinformation detection system.  
It embeds factual statements, stores them in Qdrant, and retrieves semantically similar entries for reasoning or verification.

## What it does
Loads mock data from data/mock.json

Embeds text using intfloat/e5-small-v2

Stores vectors in Qdrant

Runs semantic search with cosine similarity

Prints top matches and the JSON payload for LLM reasoning

## Responsibilities
- Embed data using a Hugging Face model (`intfloat/e5-small-v2`)
- Store embeddings in Qdrant (in-memory or Docker persistent)
- Query Qdrant for semantic similarity
- Return top matches for the reasoning (LLM) layer

## Setup
```bash
cd Misinformation_LLM

# Create and activate virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt


#Run the Demo

python -m src.pipeline_demo
