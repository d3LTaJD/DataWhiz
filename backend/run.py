"""
DataWhiz Backend Server Entry Point
"""
import os
import sys
from app import create_app

if __name__ == '__main__':
    app = create_app()
    
    # Debug mode detection:
    # 1. Check FLASK_DEBUG environment variable
    # 2. Check command line argument --debug
    # 3. Default to True in development (when run directly, not packaged)
    flask_debug_env = os.environ.get('FLASK_DEBUG', '').lower()
    has_debug_arg = '--debug' in sys.argv
    
    # Enable debug by default if running directly (not packaged)
    # Or if explicitly set via environment variable or command line
    debug_mode = (
        flask_debug_env == 'true' or 
        has_debug_arg or
        (flask_debug_env != 'false' and not app.config.get('PRODUCTION', False))
    )
    
    use_reloader = debug_mode  # Only use reloader in debug mode
    
    # Print startup messages in debug mode
    if debug_mode:
        print("=" * 50)
        print("Starting DataWhiz Backend API (DEBUG MODE)")
        print("=" * 50)
        print("API will be available at: http://localhost:5000")
        print("Debug mode: ENABLED")
        print("Auto-reloader: ENABLED" if use_reloader else "Auto-reloader: DISABLED")
        print("=" * 50)
    else:
        print("Starting DataWhiz Backend API (PRODUCTION MODE)")
        print("API will be available at: http://localhost:5000")
    
    # Enable detailed error pages and auto-reload in debug mode
    app.run(
        debug=debug_mode, 
        use_reloader=use_reloader, 
        host='0.0.0.0', 
        port=5000,
        threaded=True
    )
