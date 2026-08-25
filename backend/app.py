import eventlet
eventlet.monkey_patch()

from pathlib import Path

from flask import Flask, send_from_directory
from flask_socketio import SocketIO, emit


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"


# --------------------------------------------------
# Flask application
# --------------------------------------------------

app = Flask(__name__)
app.config["SECRET_KEY"] = "vitalvision-development-key"

socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode="eventlet",
)


# --------------------------------------------------
# Routes
# --------------------------------------------------

@app.route("/")
def index():
    """Serve the VitalVision frontend."""
    return send_from_directory(FRONTEND_DIR, "index.html")

@app.route("/css/<path:filename>")
def css_files(filename):
    """Serve frontend CSS files."""
    return send_from_directory(FRONTEND_DIR / "css", filename)


@app.route("/js/<path:filename>")
def js_files(filename):
    """Serve frontend JavaScript files."""
    return send_from_directory(FRONTEND_DIR / "js", filename)


@app.route("/health")
def health():
    """Simple backend health check."""
    return {
        "status": "ok",
        "service": "VitalVision",
    }


# --------------------------------------------------
# Socket.IO events
# --------------------------------------------------

@socketio.on("connect")
def handle_connect():
    """Called whenever a browser connects to the backend."""
    print("Client connected.")
    emit(
        "backend_status",
        {
            "status": "connected",
            "message": "VitalVision backend connected",
        },
    )


@socketio.on("disconnect")
def handle_disconnect():
    """Called whenever a browser disconnects."""
    print("Client disconnected.")


# --------------------------------------------------
# Application entry point
# --------------------------------------------------

if __name__ == "__main__":
    print("Starting VitalVision backend...")
    print("Open http://localhost:5001")

    socketio.run(
        app,
        host="127.0.0.1",
        port=5001,
        debug=True,
    )