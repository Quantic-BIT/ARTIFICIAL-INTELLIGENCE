"""
Flask Routes and Main Application Logic
"""
from flask import Blueprint, render_template, request, jsonify
import time

main_bp = Blueprint('main', __name__)


# Lazy load RAG chain to avoid loading during imports
_rag_chain = None


def get_chain():
    """Get the RAG chain with lazy loading."""
    global _rag_chain
    if _rag_chain is None:
        from app.rag.chain import get_rag_chain
        _rag_chain = get_rag_chain()
    return _rag_chain


@main_bp.route('/')
def index():
    """Render the main chat interface."""
    return render_template('index.html')


@main_bp.route('/chat', methods=['POST'])
def chat():
    """
    Process a chat message and return the response.
    
    Request JSON:
        {"message": "user question"}
    
    Response JSON:
        {
            "answer": "response text",
            "sources": [...],
            "latency_ms": 123.4
        }
    """
    try:
        data = request.get_json()
        
        if not data or 'message' not in data:
            return jsonify({
                "error": "Missing 'message' field in request"
            }), 400
        
        question = data['message'].strip()
        
        if not question:
            return jsonify({
                "error": "Message cannot be empty"
            }), 400
        
        # Get response from RAG chain
        chain = get_chain()
        response = chain.query(question)
        
        return jsonify({
            "answer": response.answer,
            "sources": response.sources,
            "latency_ms": round(response.latency_ms, 2)
        })
        
    except Exception as e:
        return jsonify({
            "error": f"An error occurred: {str(e)}"
        }), 500


@main_bp.route('/health')
def health():
    """Health check endpoint."""
    try:
        # Basic health check
        chain = get_chain()
        doc_count = chain.vectorstore.count
        
        return jsonify({
            "status": "healthy",
            "documents_indexed": doc_count,
            "model": chain.model_name,
            "timestamp": time.time()
        })
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500


@main_bp.route('/api/reindex', methods=['POST'])
def reindex():
    """Force reindex of all documents."""
    try:
        from app.rag.vectorstore import initialize_vectorstore
        store = initialize_vectorstore(force_reindex=True)
        
        return jsonify({
            "status": "success",
            "documents_indexed": store.count
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500
