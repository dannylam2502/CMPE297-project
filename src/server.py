"""
Flask Backend Server for Fact-Checking Pipeline
Provides REST API endpoint for the React frontend
"""
from dotenv import load_dotenv
load_dotenv()
from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os
from pathlib import Path
from threading import Lock           # Used to prevent race conditions when switching LLMs
from dotenv import load_dotenv

# -------------------------------------------------------------------------
# Setup environment and paths
# -------------------------------------------------------------------------
SERVER_DIR = Path(__file__).parent.resolve()
if SERVER_DIR.name == 'src':
    PROJECT_ROOT = SERVER_DIR.parent
else:
    # server.py is at project root
    PROJECT_ROOT = SERVER_DIR
os.chdir(PROJECT_ROOT)  # Change working directory to project root

# Load .env from project root
load_dotenv(PROJECT_ROOT / '.env')

from pipeline import FactCheckingPipeline

# -------------------------------------------------------------------------
# Flask app setup
# -------------------------------------------------------------------------
app = Flask(__name__)
CORS(app)
pipeline_lock = Lock()

# -------------------------------------------------------------------------
# Initialize pipeline once at startup
# -------------------------------------------------------------------------


LLM_PROVIDER = os.environ.get('LLM_PROVIDER')
if not LLM_PROVIDER:
    raise ValueError("LLM_PROVIDER not set. Run setup.sh first.")

print("\nInitializing Fact-Checking Pipeline...")
print(f"  LLM Provider: {LLM_PROVIDER}")
print(f"  Project Root: {PROJECT_ROOT}")

# Use absolute paths from project root
QDRANT_URL = os.environ["QDRANT_URL"]
QDRANT_API_KEY = os.environ["QDRANT_API_KEY"]


# Initialize pipeline
pipeline = FactCheckingPipeline(
    use_reasoning=True,
    llm_provider=LLM_PROVIDER,
    qdrant_url=QDRANT_URL,
    qdrant_api_key=QDRANT_API_KEY
)

# -------------------------------------------------------------------------
# Verify Qdrant database and warn if ingestion is missing
# -------------------------------------------------------------------------
collection_name = "nba_claims"
try:
    size = pipeline.vector_db.get_collection_size()
    if size == 0:
        print(f" Qdrant collection '{collection_name}' is empty.")
        print(f"    → Run ingestion manually: python src/modules/misinformation_module/src/ingest_nba.py")
    else:
        print(f" Qdrant collection '{collection_name}' loaded successfully with {size} entries.")
except Exception as e:
    print(f" Could not check collection size: {e}")
    print("    → Run ingestion manually if you haven't already.")

# -------------------------------------------------------------------------
# Flask API endpoints
# -------------------------------------------------------------------------


# ============================================
# Function: rebuild_pipeline
# ============================================
def rebuild_pipeline(new_provider: str) -> bool:
    """
    Reinitialize the fact-checking pipeline with a different LLM provider (OpenAI or Ollama).
    Returns True if switched successfully, False if no change was needed.
    """
    global pipeline, LLM_PROVIDER

    # Normalize the incoming LLM name
    normalized_provider = (new_provider or '').lower()
    if normalized_provider not in ('openai', 'ollama'):
        raise ValueError(f"Invalid LLM provider: {new_provider}")
    
    # Lock ensures no other process changes pipeline during rebuild
    with pipeline_lock:
        try:
            if hasattr(pipeline, "vector_db") and hasattr(pipeline.vector_db, "client"):
                pipeline.vector_db.client.close()  # Free older qdrant
                print("Closed previous Qdrant client.")
        except Exception as e:
            print(f"Warning: could not close Qdrant client: {e}")          

        current_provider = (LLM_PROVIDER or '').lower()
        if normalized_provider == current_provider:
            print(f"LLM provider already set to '{LLM_PROVIDER}'. No changes made.")
            return False
        
        # Log the provider switch
        print(f"Switching LLM provider: {LLM_PROVIDER} → {normalized_provider}")
        print("Reinitializing fact-checking pipeline with requested provider...")
        current_reasoning = getattr(pipeline, 'use_reasoning', True)
        new_pipeline = FactCheckingPipeline(
            use_reasoning=current_reasoning,
            llm_provider=normalized_provider,
            qdrant_url=QDRANT_URL,
            qdrant_api_key=QDRANT_API_KEY
        )


        # Replace old pipeline with new one
        pipeline = new_pipeline
        LLM_PROVIDER = normalized_provider
        print(f"LLM provider switched successfully to '{LLM_PROVIDER}'.")
        return True

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    """
    Main chat endpoint that processes user queries.
    
    GET params: ?question=<user_query>
    POST body: {"question": "<user_query>"}
    
    Returns: JSON response with verdict, score, explanation, citations
    """
    try:
        # Support both GET and POST
        if request.method == 'GET':
            question = request.args.get('question', '')
        else:
            data = request.get_json(force=True)
            question = data.get("question", "")

        if not question.strip():
            return jsonify({
                "error": "No question provided",
                "claim": "",
                "verdict": "Not enough evidence",
                "score": 0,
                "explanation": "Please provide a factual claim or question.",
                "citations": [],
                "features": {}
            }), 400

        # Only use NBA data
        pipeline.available_collections = ["nba_claims"]

        # Run query through pipeline
        result = pipeline.process_query(question)

        # Prepare JSON for frontend
        response = {
            "claim": result.get("claim", question),
            "verdict": result.get("verdict", "Error"),
            "score": result.get("score", 0),
            "explanation": result.get("explanation", "No explanation available."),
            "citations": result.get("citations", []),
            "features": result.get("features", {}),
            "formatted_text": pipeline.format_for_ui(result)
        }

        return jsonify(response)

    except Exception as e:
        print(f"Error processing query: {e}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            'error': str(e),
            'claim': question if 'question' in locals() else '',
            'verdict': 'Error',
            'score': 0,
            'explanation': f'An error occurred while processing your request: {str(e)}',
            'citations': [],
            'features': {}
        }), 500



@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "ok",
        "service": "fact-checking-api",
        "reasoning_enabled": getattr(pipeline, "use_reasoning", True)
    })


@app.route("/toggle-reasoning", methods=["POST"])
def toggle_reasoning():
    """Toggle reasoning engine on/off"""
    try:
        data = request.get_json()
        enable = data.get('enable', True)
        pipeline.use_reasoning = enable
        return jsonify({
            'status': 'ok',
            'reasoning_enabled': pipeline.use_reasoning
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --- Runtime LLM Provider Switching ---
# This route and helper method allow developers to change the backend model
# (OpenAI or Ollama) without restarting the Flask server.
# Safe extension: does not modify any teammate’s code.
@app.route('/set-llm', methods=['POST'])
def set_llm():
    """
    Switch the active LLM provider at runtime.
    
    Expected JSON payload: {"llm_provider": "openai" | "ollama"}
    Returns the normalized provider so the frontend can confirm which model is live.
    """
    try:
        data = request.get_json(silent=True) or {}
    except Exception:
        data = {}
    
    requested_provider = (data.get('llm_provider') or "").strip()
    if not requested_provider:
        return jsonify({
            'error': "Missing 'llm_provider' in request body.",
            'allowed_providers': ['openai', 'ollama']
        }), 400
    
    try:
        normalized = pipeline.set_llm_provider(requested_provider)
    except ValueError as ve:
        return jsonify({
            'error': str(ve),
            'allowed_providers': ['openai', 'ollama']
        }), 400
    except Exception as exc:
        print(f"Error switching LLM provider: {exc}")
        return jsonify({
            'error': 'Failed to update LLM provider.',
            'details': str(exc)
        }), 500
    
    return jsonify({
        'status': 'ok',
        'llm_provider': normalized
    })

if __name__ == '__main__':
    PORT = int(os.environ.get('PORT', 5005))
    print(f"\n{'='*60}")
    print(f"Fact-Checking API Server")
    print(f"{'='*60}")
    print(f"Server URL: http://localhost:{PORT}")
    print(f"API endpoint: http://localhost:{PORT}/chat")
    print(f"Health check: http://localhost:{PORT}/health")
    print(f"Reasoning: {'Enabled' if pipeline.use_reasoning else 'Disabled'}")
    print(f"{'='*60}\n")
    app.run(host='0.0.0.0', port=PORT, debug=True, use_reloader=False)
