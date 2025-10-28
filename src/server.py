"""
Flask Backend Server for Fact-Checking Pipeline
Provides REST API endpoint for the React frontend
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Calculate project root - handle both src/server.py and ./server.py locations
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

app = Flask(__name__)
CORS(app)

# Initialize pipeline once at startup with reasoning enabled
LLM_PROVIDER = os.environ.get('LLM_PROVIDER')
if not LLM_PROVIDER:
    raise ValueError("LLM_PROVIDER not set. Run setup.sh first.")

print(f"Initializing fact-checking pipeline...")
print(f"  LLM Provider: {LLM_PROVIDER}")
print(f"  Project Root: {PROJECT_ROOT}")

# Use absolute paths from project root
QDRANT_LOCATION = str(PROJECT_ROOT / "data" / "qdrant")
pipeline = FactCheckingPipeline(
    use_reasoning=True, 
    llm_provider=LLM_PROVIDER,
    qdrant_location=QDRANT_LOCATION
)

# Load knowledge base with hash-based rebuild
KNOWLEDGE_BASE_PATH = PROJECT_ROOT / "data" / "fever.json"
if not KNOWLEDGE_BASE_PATH.exists():
    print(f"Warning: Knowledge base not found at {KNOWLEDGE_BASE_PATH}")
    print("Falling back to mock.json...")
    KNOWLEDGE_BASE_PATH = PROJECT_ROOT / "data" / "mock.json"

if KNOWLEDGE_BASE_PATH.exists():
    collection_size = pipeline.vector_db.get_collection_size()
    current_hash = pipeline.compute_source_hash(str(KNOWLEDGE_BASE_PATH))
    metadata = pipeline.load_metadata()
    stored_hash = metadata.get('source_hash')
    stored_model = metadata.get('embedding_model')
    current_model = os.environ.get('EMBEDDING_MODEL', 'intfloat/e5-small-v2')
    
    needs_rebuild = False
    rebuild_reason = None
    
    if collection_size == 0:
        needs_rebuild = True
        rebuild_reason = "Collection is empty"
    elif stored_hash != current_hash:
        needs_rebuild = True
        rebuild_reason = f"Source file changed (hash mismatch)"
    elif stored_model != current_model:
        needs_rebuild = True
        rebuild_reason = f"Embedding model changed ({stored_model} → {current_model})"
    
    if needs_rebuild:
        print(f"Rebuilding knowledge base: {rebuild_reason}")
        if KNOWLEDGE_BASE_PATH.name == 'fever.json':
            print(f"⚠ This will take 15-20 minutes for FEVER dataset (145K claims)")
            print(f"  Install tqdm for progress bars: pip install tqdm")
        pipeline.vector_db.reset_collection()
        pipeline.load_knowledge_base(str(KNOWLEDGE_BASE_PATH))
        print(f"Knowledge base loaded from {KNOWLEDGE_BASE_PATH.name}")
    else:
        print(f"Using existing knowledge base ({collection_size} entries)")
        print(f"  Source: {metadata.get('source_file')}")
        print(f"  Model: {metadata.get('embedding_model')}")
        print(f"  Loaded: {metadata.get('loaded_at')}")
else:
    print(f"Error: No knowledge base file found")

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
            data = request.get_json()
            question = data.get('question', '')
        
        if not question:
            return jsonify({
                'error': 'No question provided',
                'claim': '',
                'verdict': 'Not enough evidence',
                'score': 0,
                'explanation': 'No question was provided.',
                'citations': [],
                'features': {}
            }), 400
        
        # Process through pipeline
        result = pipeline.process_query(question)
        
        # Format for frontend - ensure all fields are present
        response = {
            'claim': result.get('claim', question),
            'verdict': result.get('verdict', 'Error'),
            'score': result.get('score', 0),
            'explanation': result.get('explanation', 'No explanation available.'),
            'citations': result.get('citations', []),
            'features': result.get('features', {}),
            'formatted_text': pipeline.format_for_ui(result)
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
        'status': 'ok',
        'service': 'fact-checking-api',
        'reasoning_enabled': pipeline.use_reasoning
    })

@app.route('/toggle-reasoning', methods=['POST'])
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