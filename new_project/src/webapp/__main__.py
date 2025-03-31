"""
RAVDESS Emotion Recognition - Web Application Entry Point
This module allows running the web application as a module: python -m src.webapp
"""

import os
from .app import app

if __name__ == "__main__":
    # Get port from environment variable or use default
    port = int(os.environ.get("PORT", 8080))
    
    # Run the app
    app.run(host="0.0.0.0", port=port, debug=True)
