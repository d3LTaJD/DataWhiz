"""
DataWhiz Flask Application Factory
"""
from flask import Flask, jsonify
from flask_cors import CORS
import os
import json
import numpy as np

class NaNEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, float) and np.isnan(obj):
            return None
        return super().default(obj)

def create_app():
    """Create and configure the Flask application"""
    app = Flask(__name__)
    
    # Configure CORS - Allow requests from Electron app and localhost
    CORS(app, origins=[
        'http://localhost:3000', 
        'http://127.0.0.1:3000',
        'http://localhost:5000',
        'http://127.0.0.1:5000',
        'file://'
    ], supports_credentials=True)
    
    # Configure upload settings
    app.config['UPLOAD_FOLDER'] = 'uploads'
    app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB (increased for large datasets)
    app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False  # Smaller JSON responses
    
    # Set custom JSON encoder
    app.json_encoder = NaNEncoder
    
    # Ensure upload directory exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Register blueprints
    from app.api.upload_routes import upload_bp
    from app.api.business_routes import business_bp
    from app.api.healthcare_routes import healthcare_bp
    from app.api.analytics_routes import analytics_bp
    
    app.register_blueprint(upload_bp)
    app.register_blueprint(business_bp)
    app.register_blueprint(healthcare_bp)
    app.register_blueprint(analytics_bp)
    
    # Basic routes
    @app.route('/', methods=['GET'])
    def root():
        """Root endpoint - API information"""
        return jsonify({
            'message': 'DataWhiz Backend API',
            'version': '1.0.0',
            'endpoints': {
                'upload': '/api/upload',
                'preview': '/api/preview/<filename>',
                'data': '/api/data/<filename>',
                'business_quality': '/api/business/quality-check',
                'business_clean': '/api/business/clean-data',
                'business_transform': '/api/business/transform-data',
                'business_analyze': '/api/business/analyze',
                'healthcare_quality': '/api/healthcare/quality-check',
                'healthcare_clean': '/api/healthcare/clean-data',
                'healthcare_transform': '/api/healthcare/transform-data',
                'healthcare_analyze': '/api/healthcare/analyze',
                'analytics_revenue': '/api/analytics/revenue/total',
                'analytics_trends': '/api/analytics/revenue/trends',
                'analytics_rfm': '/api/analytics/customer/rfm',
                'analytics_products': '/api/analytics/products/top',
                'analytics_geographic': '/api/analytics/geographic/analysis',
                'analytics_ml': '/api/analytics/ml/*'
            }
        })
    
    @app.route('/favicon.ico', methods=['GET'])
    def favicon():
        """Handle favicon requests"""
        from flask import abort
        abort(404)
    
    @app.route('/api/health', methods=['GET'])
    def health_check():
        """Health check endpoint"""
        return jsonify({
            'status': 'healthy',
            'message': 'DataWhiz Backend is running'
        })
    
    return app
