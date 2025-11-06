"""
Flask Backend Server for Fact-Checking Pipeline
Provides REST API endpoint for the React frontend
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from pathlib import Path
from dotenv import load_dotenv

# -------------------------------------------------------------------------
# Setup environment and paths
# -------------------------------------------------------------------------
SERVER_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SERVER_DIR.parent if SERVER_DIR.name == "src" else SERVER_DIR
os.chdir(PROJECT_ROOT)
load_dotenv(PROJECT_ROOT / ".env")

from pipeline import FactCheckingPipeline

# -------------------------------------------------------------------------
# Flask app setup
# -------------------------------------------------------------------------
app = Flask(__name__)
CORS(app)

# -------------------------------------------------------------------------
# Initialize pipeline once at startup
# -------------------------------------------------------------------------
LLM_PROVIDER = os.environ.get("LLM_PROVIDER")
if not LLM_PROVIDER:
    raise ValueError("LLM_PROVIDER not set. Run setup.sh first.")

print("\nInitializing Fact-Checking Pipeline...")
print(f"  LLM Provider: {LLM_PROVIDER}")
print(f"  Project Root: {PROJECT_ROOT}")

QDRANT_LOCATION = str(PROJECT_ROOT / "data" / "qdrant")
NBA_DATA_PATH = PROJECT_ROOT / "data" / "nba.json"

# Initialize pipeline
pipeline = FactCheckingPipeline(
    llm_provider=LLM_PROVIDER,
    qdrant_location=QDRANT_LOCATION
)

# -------------------------------------------------------------------------
# Verify Qdrant database and warn if ingestion is missing
# -------------------------------------------------------------------------
collection_name = "nba_claims"
try:
    size = pipeline.vector_db.get_collection_size(collection=collection_name)
    if size == 0:
        print(f"[⚠] Qdrant collection '{collection_name}' is empty.")
        print(f"    → Run ingestion manually: python src/modules/misinformation_module/src/ingest_nba.py")
    else:
        print(f"[✓] Qdrant collection '{collection_name}' loaded successfully with {size} entries.")
except Exception as e:
    print(f"[⚠] Could not check collection size: {e}")
    print("    → Run ingestion manually if you haven't already.")

# -------------------------------------------------------------------------
# Flask API endpoints
# -------------------------------------------------------------------------

@app.route("/chat", methods=["GET", "POST"])
def chat():
    """Main chat endpoint that processes user queries."""
    try:
        if request.method == "GET":
            question = request.args.get("question", "")
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
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": str(e),
            "claim": question if "question" in locals() else "",
            "verdict": "Error",
            "score": 0,
            "explanation": f"An error occurred: {e}",
            "citations": [],
            "features": {}
        }), 500


@app.route("/health", methods=["GET"])
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
        data = request.get_json(force=True)
        enable = data.get("enable", True)
        pipeline.use_reasoning = enable
        return jsonify({
            "status": "ok",
            "reasoning_enabled": pipeline.use_reasoning
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -------------------------------------------------------------------------
# Run Flask server
# -------------------------------------------------------------------------
if __name__ == "__main__":
    PORT = int(os.environ.get("PORT", 5005))
    print("\n" + "=" * 60)
    print(" Fact-Checking API Server ")
    print("=" * 60)
    print(f"Server URL:     http://localhost:{PORT}")
    print(f"API endpoint:   http://localhost:{PORT}/chat")
    print(f"Health check:   http://localhost:{PORT}/health")
    print(f"Reasoning:      {'Enabled' if getattr(pipeline, 'use_reasoning', True) else 'Disabled'}")
    print("=" * 60 + "\n")
    app.run(host="0.0.0.0", port=PORT, debug=True, use_reloader=False)
