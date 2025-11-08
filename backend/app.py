"""
DataWhiz Backend API
Flask-based backend for data processing and analytics
"""

from app import create_app

# Create the Flask application using the factory pattern
app = create_app()


if __name__ == '__main__':
    print("Starting DataWhiz Backend API...")
    print("API will be available at: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
