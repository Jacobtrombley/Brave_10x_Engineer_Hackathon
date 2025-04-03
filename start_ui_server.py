#!/usr/bin/env python
"""
Simple HTTP Server for Job Matching Recommendation System UI
"""

import http.server
import socketserver
import webbrowser
import os
import time
from pathlib import Path

# Configuration
PORT = 8080
DIRECTORY = "ui"

# Set the current directory to the UI directory
os.chdir(DIRECTORY)

Handler = http.server.SimpleHTTPRequestHandler

# Create a web server
with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print(f"Server started at http://localhost:{PORT}")
    print(f"Job Seeker UI: http://localhost:{PORT}/job_seeker/")
    print(f"Employer UI: http://localhost:{PORT}/employer/")
    
    # Open the browser
    webbrowser.open(f"http://localhost:{PORT}/job_seeker/")
    time.sleep(1)  # Wait a second before opening the second tab
    webbrowser.open(f"http://localhost:{PORT}/employer/")
    
    # Keep the server running
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("Server stopped.") 