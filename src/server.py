"""
Flask Backend Server for Fact-Checking Pipeline
Provides REST API endpoint for the React frontend
"""
from dotenv import load_dotenv
load_dotenv()

from flask import Flask, request, jsonify, session
from flask_cors import CORS
import sys
import os
from pathlib import Path
from threading import Lock

# -------------------------------------------------------------------------
# Setup environment and paths
# -------------------------------------------------------------------------
SERVER_DIR = Path(__file__).parent.resolve()
if SERVER_DIR.name == 'src':
    PROJECT_ROOT = SERVER_DIR.parent
else:
    PROJECT_ROOT = SERVER_DIR
os.chdir(PROJECT_ROOT)

load_dotenv(PROJECT_ROOT / '.env')

from pipeline import FactCheckingPipeline
from modules.backEnd.auth import AuthDB

# -------------------------------------------------------------------------
# Flask app setup
# -------------------------------------------------------------------------
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key")
CORS(
    app,
    resources={r"/*": {"origins": ["http://localhost:3000", "http://127.0.0.1:3000"]}},
    supports_credentials=True,
)
pipeline_lock = Lock()

# Initialize auth database
auth_db = AuthDB(PROJECT_ROOT / "src" / "modules" / "backEnd" / "users.db")

# -------------------------------------------------------------------------
# Initialize pipeline
# -------------------------------------------------------------------------
LLM_PROVIDER = os.environ.get('LLM_PROVIDER')
if not LLM_PROVIDER:
    raise ValueError("LLM_PROVIDER not set. Run setup.sh first.")

print("\nInitializing Fact-Checking Pipeline...")
print(f"  LLM Provider: {LLM_PROVIDER}")
print(f"  Project Root: {PROJECT_ROOT}")

QDRANT_URL = os.environ.get("QDRANT_URL")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")

if not QDRANT_URL or not QDRANT_API_KEY:
    print("ERROR: QDRANT_URL or QDRANT_API_KEY is missing.")
    print("Check your .env file and ensure these values are set.")
    sys.exit(1)

pipeline = FactCheckingPipeline(
    use_reasoning=True,
    llm_provider=LLM_PROVIDER,
    qdrant_url=QDRANT_URL,
    qdrant_api_key=QDRANT_API_KEY
)

# Verify Qdrant
collection_name = "nba_claims"
try:
    size = pipeline.vector_db.get_collection_size()
    if size == 0:
        print(f"Qdrant collection '{collection_name}' is empty.")
        print("  Run: python src/modules/misinformation_module/src/ingest_nba.py")
    else:
        print(f"Qdrant collection '{collection_name}' loaded: {size} entries")
except Exception as e:
    print(f"Could not check collection size: {e}")


# -------------------------------------------------------------------------
# Pipeline management
# -------------------------------------------------------------------------
def rebuild_pipeline(new_provider: str) -> bool:
    """Reinitialize pipeline with different LLM provider."""
    global pipeline, LLM_PROVIDER

    normalized_provider = (new_provider or '').lower()
    if normalized_provider not in ('openai', 'ollama'):
        raise ValueError(f"Invalid LLM provider: {new_provider}")

    with pipeline_lock:
        try:
            if hasattr(pipeline, "vector_db") and hasattr(pipeline.vector_db, "client"):
                pipeline.vector_db.client.close()
        except Exception as e:
            print(f"Warning: could not close Qdrant client: {e}")

        current_provider = (LLM_PROVIDER or '').lower()
        if normalized_provider == current_provider:
            return False

        print(f"Switching LLM provider: {LLM_PROVIDER} -> {normalized_provider}")
        new_pipeline = FactCheckingPipeline(
            use_reasoning=getattr(pipeline, 'use_reasoning', True),
            llm_provider=normalized_provider,
            qdrant_url=QDRANT_URL,
            qdrant_api_key=QDRANT_API_KEY
        )

        pipeline = new_pipeline
        LLM_PROVIDER = normalized_provider
        return True


# -------------------------------------------------------------------------
# API Routes: Fact-checking
# -------------------------------------------------------------------------
@app.route('/chat', methods=['GET', 'POST'])
def chat():
    """Process user queries through fact-checking pipeline."""
    try:
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

        pipeline.available_collections = ["nba_claims"]
        result = pipeline.process_query(question)

        return jsonify({
            "claim": result.get("claim", question),
            "verdict": result.get("verdict", "Error"),
            "score": result.get("score", 0),
            "explanation": result.get("explanation", "No explanation available."),
            "citations": result.get("citations", []),
            "features": result.get("features", {}),
            "formatted_text": pipeline.format_for_ui(result)
        })

    except Exception as e:
        print(f"Error processing query: {e}")
        import traceback
        traceback.print_exc()

        return jsonify({
            'error': str(e),
            'claim': question if 'question' in locals() else '',
            'verdict': 'Error',
            'score': 0,
            'explanation': f'An error occurred: {str(e)}',
            'citations': [],
            'features': {}
        }), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "ok",
        "service": "fact-checking-api",
        "reasoning_enabled": getattr(pipeline, "use_reasoning", True)
    })


@app.route("/toggle-reasoning", methods=["POST"])
def toggle_reasoning():
    """Toggle reasoning engine on/off."""
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


# -------------------------------------------------------------------------
# API Routes: Authentication
# -------------------------------------------------------------------------
@app.route('/login', methods=['POST'])
def login_user():
    """Authenticate user and establish session."""
    payload = request.get_json(silent=True) or {}
    email = (payload.get("email") or "").strip()
    password = payload.get("password") or ""

    if not email or not password:
        return jsonify({"success": False, "message": "Email and password are required"}), 400

    user = auth_db.authenticate(email, password)
    if not user:
        return jsonify({"success": False, "message": "Invalid email or password"}), 401

    session["user_id"] = user["id"]
    session["user_email"] = user["email"]
    session["user_name"] = user["full_name"]
    return jsonify({"success": True, "message": "Login successful"})


@app.route('/logout', methods=['POST'])
def logout_user():
    """Clear session."""
    session.clear()
    return jsonify({"success": True, "message": "Logged out successfully"})


@app.route('/register', methods=['GET', 'POST'])
def register_user():
    """Register new user."""
    if request.method == 'GET':
        return jsonify({"message": "Registration endpoint is active. Use POST to submit user data."})

    payload = request.get_json(silent=True) or {}
    full_name = (payload.get("full_name") or "").strip()
    email = (payload.get("email") or "").strip()
    password = payload.get("password") or ""

    if not full_name or not email or not password:
        return jsonify({"success": False, "message": "Full name, email, and password are required"}), 400

    success, message = auth_db.register(full_name, email, password)
    if not success:
        return jsonify({"success": False, "message": message}), 400

    return jsonify({"success": True, "message": message}), 201


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
if __name__ == '__main__':
    PORT = int(os.environ.get('PORT', 5005))
    print(f"\n{'='*60}")
    print("Fact-Checking API Server")
    print(f"{'='*60}")
    print(f"Server URL: http://localhost:{PORT}")
    print(f"API endpoint: http://localhost:{PORT}/chat")
    print(f"Health check: http://localhost:{PORT}/health")
    print(f"Reasoning: {'Enabled' if pipeline.use_reasoning else 'Disabled'}")
    print(f"{'='*60}\n")
    app.run(host='0.0.0.0', port=PORT, debug=True, use_reloader=False)