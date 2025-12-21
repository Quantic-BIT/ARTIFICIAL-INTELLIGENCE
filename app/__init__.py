"""
Acme Policy Assistant - RAG Application
"""
from flask import Flask
import os


def create_app():
    """Create and configure the Flask application."""
    app = Flask(__name__, 
                template_folder='templates',
                static_folder='static')
    
    # Configuration
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key')
    
    # Register routes
    from app.main import main_bp
    app.register_blueprint(main_bp)
    
    return app
