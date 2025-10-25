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

# Initialize pipeline once at startup with reasoning enabled
print("Initializing fact-checking pipeline with reasoning engine...")
pipeline = FactCheckingPipeline(use_reasoning=True)

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