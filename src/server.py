"""
Flask Backend Server for Fact-Checking Pipeline
Provides REST API endpoint for the React frontend
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pipeline import FactCheckingPipeline

app = Flask(__name__)
CORS(app)

# Initialize pipeline once at startup
print("Initializing fact-checking pipeline...")
pipeline = FactCheckingPipeline()

# Load knowledge base
KNOWLEDGE_BASE_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'mock.json')
if os.path.exists(KNOWLEDGE_BASE_PATH):
    pipeline.load_knowledge_base(KNOWLEDGE_BASE_PATH)
    print(f"Knowledge base loaded from {KNOWLEDGE_BASE_PATH}")
else:
    print(f"Warning: Knowledge base not found at {KNOWLEDGE_BASE_PATH}")

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    """
    Main chat endpoint that processes user queries.
    
    GET params: ?question=<user_query>
    POST body: {"question": "<user_query>"}
    
    Returns: JSON response with verdict, score, citations
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
                'score': 0
            }), 400
        
        # Process through pipeline
        result = pipeline.process_query(question)
        
        # Format for frontend
        response = {
            'claim': result['claim'],
            'verdict': result['verdict'],
            'score': result['score'],
            'citations': result['citations'],
            'features': result['features'],
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
            'score': 0
        }), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'ok', 'service': 'fact-checking-api'})

if __name__ == '__main__':
    PORT = int(os.environ.get('PORT', 5005))
    print(f"\nStarting server on port {PORT}...")
    print(f"API endpoint: http://localhost:{PORT}/chat")
    print(f"Health check: http://localhost:{PORT}/health\n")
    app.run(host='0.0.0.0', port=PORT, debug=True)
