#!/usr/bin/env python3
"""
Lightweight Flask Web Server with Server-Sent Events (SSE).
Provides real-time visual feedback for the voice control system:
- Detected intent, slot, confidence, latency
- Virtual RGB LED indicators with PWM dimming
- Music player status
- Live context updates (weather, time, timers)
"""

import os
import sys
import json
import time
import queue
from flask import Flask, render_template, request, Response, jsonify

app = Flask(__name__, template_folder="templates")

# Client queues for SSE broadcasting
subscribers = []

def broadcast_event(event_type, payload):
    data = json.dumps({"type": event_type, **payload})
    msg = f"data: {data}\n\n"
    for q in subscribers[:]:
        try:
            q.put_nowait(msg)
        except Exception:
            subscribers.remove(q)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/stream")
def sse_stream():
    def event_generator():
        client_queue = queue.Queue(maxsize=100)
        subscribers.append(client_queue)
        
        # Send initial status
        init_json = json.dumps({'type': 'status', 'state': 'LISTENING', 'details': 'LISTENING FOR "BETELGEUSE"'})
        yield f"data: {init_json}\n\n"
        
        try:
            while True:
                msg = client_queue.get()
                yield msg
        except GeneratorExit:
            if client_queue in subscribers:
                subscribers.remove(client_queue)

    return Response(event_generator(), mimetype="text/event-stream")

@app.route("/api/command", methods=["POST"])
def post_command():
    data = request.get_json(silent=True) or {}
    broadcast_event("command", {
        "intent": data.get("intent", "UNKNOWN"),
        "slot": data.get("slot", ""),
        "confidence": float(data.get("confidence", 0.0)),
        "latency_ms": float(data.get("latency_ms", 0.0))
    })
    return jsonify({"status": "ok"})

@app.route("/api/leds", methods=["POST"])
def post_leds():
    data = request.get_json(silent=True) or {}
    broadcast_event("leds", {
        "red": float(data.get("red", 0.0)),
        "green": float(data.get("green", 0.0)),
        "blue": float(data.get("blue", 0.0))
    })
    return jsonify({"status": "ok"})

@app.route("/api/audio", methods=["POST"])
def post_audio():
    data = request.get_json(silent=True) or {}
    broadcast_event("audio", {
        "level": float(data.get("level", 0.0)),
        "state": data.get("state", "LISTENING")
    })
    return jsonify({"status": "ok"})

@app.route("/api/status", methods=["POST"])
def post_status():
    data = request.get_json(silent=True) or {}
    broadcast_event("status", {
        "state": data.get("state", "LISTENING"),
        "details": data.get("details", "")
    })
    return jsonify({"status": "ok"})

@app.route("/api/music", methods=["POST"])
def post_music():
    data = request.get_json(silent=True) or {}
    broadcast_event("music", {
        "status": data.get("status", "stopped"),
        "track": data.get("track", ""),
        "volume": float(data.get("volume", 0.8))
    })
    return jsonify({"status": "ok"})

@app.route("/api/context", methods=["POST"])
def post_context():
    data = request.get_json(silent=True) or {}
    broadcast_event("context", {
        "text": data.get("text", "")
    })
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"[*] Starting Voice Control Dashboard at http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)

