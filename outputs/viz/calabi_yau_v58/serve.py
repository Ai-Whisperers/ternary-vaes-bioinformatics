"""Simple HTTP server for Calabi-Yau v5.8 fibration viewer."""
import http.server
import socketserver
import webbrowser
import os

PORT = 8010

os.chdir(os.path.dirname(os.path.abspath(__file__)))

Handler = http.server.SimpleHTTPRequestHandler
Handler.extensions_map.update({
    '.js': 'application/javascript',
    '.json': 'application/json',
})

print(f"Starting server at http://localhost:{PORT}")
print("Press Ctrl+C to stop")

webbrowser.open(f'http://localhost:{PORT}/fibration_viewer.html')

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped")
