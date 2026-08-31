"""
IBVAP - Universal Live Camera Hub for Frigate NVR
Receives browser-selected camera streams (with permission prompts & hardware names)
and serves them directly into Frigate's YOLOv8 detection pipeline.
"""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import time

latest_frame = None
frame_lock = threading.Lock()
running = True

# Fallback test pattern image if browser hasn't sent frames yet
import cv2
import numpy as np

def make_standby_frame(text="Waiting for Browser Camera..."):
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    img[:] = (30, 41, 59) # Slate background
    cv2.putText(img, "IBVAP - Live Camera Stream", (400, 320), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(img, text, (360, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (59, 130, 246), 2)
    cv2.putText(img, "Open: http://localhost:8080 to select camera & broadcast", (320, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (148, 163, 184), 1)
    _, jpeg = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return jpeg.tobytes()

standby_jpeg = make_standby_frame()

class StreamHubHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass # Silence verbose streaming logs

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_POST(self):
        global latest_frame
        if self.path == '/push_frame':
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length > 0:
                frame_data = self.rfile.read(content_length)
                with frame_lock:
                    latest_frame = frame_data
                self.send_response(200)
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(b'OK')
            else:
                self.send_response(400)
                self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    def do_GET(self):
        global latest_frame
        if self.path.startswith('/live'):
            # MJPEG stream for Frigate NVR
            self.send_response(200)
            self.send_header('Age', '0')
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
            self.end_headers()

            while running:
                with frame_lock:
                    current_jpeg = latest_frame if latest_frame is not None else standby_jpeg

                try:
                    self.wfile.write(b'--FRAME\r\n')
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', str(len(current_jpeg)))
                    self.end_headers()
                    self.wfile.write(current_jpeg)
                    self.wfile.write(b'\r\n')
                    time.sleep(0.066) # ~15 FPS
                except (
                    BrokenPipeError,
                    ConnectionAbortedError,
                    ConnectionResetError,
                ):
                    break

        elif self.path == '/' or self.path.startswith('/index'):
            # Interactive Camera Selector Page with Browser Permissions
            html = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>IBVAP - Live Camera Selector & Streamer</title>
  <style>
    :root {
      --bg: #0b0f19;
      --card: #151e2e;
      --accent: #2563eb;
      --accent-hover: #1d4ed8;
      --text: #f8fafc;
      --muted: #94a3b8;
      --success: #10b981;
      --danger: #ef4444;
    }
    body {
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background-color: var(--bg);
      color: var(--text);
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      min-height: 100vh;
      padding: 20px;
      box-sizing: border-box;
    }
    .card {
      background: var(--card);
      border-radius: 16px;
      box-shadow: 0 20px 35px -10px rgba(0,0,0,0.6);
      width: 100%;
      max-width: 800px;
      padding: 28px;
      border: 1px solid rgba(255,255,255,0.08);
    }
    h1 {
      font-size: 1.6rem;
      margin-top: 0;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }
    .badge {
      font-size: 0.8rem;
      background: rgba(37, 99, 235, 0.2);
      color: #60a5fa;
      padding: 4px 10px;
      border-radius: 6px;
      border: 1px solid rgba(96, 165, 250, 0.3);
    }
    .section-title {
      font-size: 0.9rem;
      color: var(--muted);
      margin-bottom: 8px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }
    .controls {
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      margin-bottom: 20px;
    }
    select {
      flex: 1;
      min-width: 260px;
      padding: 12px 16px;
      border-radius: 10px;
      font-size: 1rem;
      border: 1px solid rgba(255,255,255,0.15);
      background: #1e293b;
      color: var(--text);
      outline: none;
      cursor: pointer;
    }
    button {
      padding: 12px 24px;
      border-radius: 10px;
      font-size: 1rem;
      font-weight: 600;
      border: none;
      cursor: pointer;
      background: var(--accent);
      color: white;
      transition: all 0.2s;
    }
    button:hover {
      background: var(--accent-hover);
    }
    button.stop {
      background: var(--danger);
    }
    .video-container {
      position: relative;
      width: 100%;
      aspect-ratio: 16 / 9;
      background: #020617;
      border-radius: 12px;
      overflow: hidden;
      display: flex;
      align-items: center;
      justify-content: center;
      border: 1px solid rgba(255,255,255,0.1);
    }
    video {
      width: 100%;
      height: 100%;
      object-fit: cover;
    }
    canvas {
      display: none;
    }
    .status-bar {
      margin-top: 16px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      font-size: 0.9rem;
      color: var(--muted);
    }
    .live-indicator {
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .dot {
      width: 10px;
      height: 10px;
      border-radius: 50%;
      background: var(--muted);
    }
    .dot.active {
      background: var(--success);
      box-shadow: 0 0 12px var(--success);
    }
    .frigate-link {
      color: #60a5fa;
      text-decoration: none;
      font-weight: 500;
    }
    .frigate-link:hover {
      text-decoration: underline;
    }
  </style>
</head>
<body>

  <div class="card">
    <h1>
      <span>📹 Live Camera Selector</span>
      <span class="badge">Frigate YOLOv8 Broadcaster</span>
    </h1>

    <div class="section-title">Select Physical Camera Device:</div>
    <div class="controls">
      <select id="cameraSelect">
        <option value="">Requesting camera permissions...</option>
      </select>
      <button id="broadcastBtn" onclick="toggleBroadcast()">▶ Start Live Stream</button>
    </div>

    <div class="video-container">
      <video id="previewVideo" autoplay playsinline muted></video>
      <canvas id="captureCanvas" width="1280" height="720"></canvas>
    </div>

    <div class="status-bar">
      <div class="live-indicator">
        <div class="dot" id="statusDot"></div>
        <span id="statusText">Select your camera and click Start Live Stream.</span>
      </div>
      <a href="http://localhost:5000" target="_blank" class="frigate-link">Open Frigate UI ↗</a>
    </div>
  </div>

  <script>
    let currentStream = null;
    let isStreaming = false;
    let streamInterval = null;

    const select = document.getElementById("cameraSelect");
    const video = document.getElementById("previewVideo");
    const canvas = document.getElementById("captureCanvas");
    const ctx = canvas.getContext("2d");
    const broadcastBtn = document.getElementById("broadcastBtn");
    const statusDot = document.getElementById("statusDot");
    const statusText = document.getElementById("statusText");

    async function initCameraDevices() {
      try {
        // Trigger browser permission modal
        const initialStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        initialStream.getTracks().forEach(t => t.stop());

        // Enumerate with full hardware names
        const devices = await navigator.mediaDevices.enumerateDevices();
        const videoDevices = devices.filter(d => d.kind === "videoinput");

        select.innerHTML = "";
        videoDevices.forEach((device, index) => {
          const option = document.createElement("option");
          option.value = device.deviceId;
          option.text = device.label || `Camera ${index + 1} (${device.deviceId.slice(0, 8)}...)`;
          select.appendChild(option);
        });

        startCameraPreview();
      } catch (err) {
        statusText.innerText = "Camera permission denied or camera not found: " + err.message;
      }
    }

    async function startCameraPreview() {
      if (currentStream) {
        currentStream.getTracks().forEach(t => t.stop());
      }
      const deviceId = select.value;
      currentStream = await navigator.mediaDevices.getUserMedia({
        video: { deviceId: deviceId ? { exact: deviceId } : undefined, width: 1280, height: 720 },
        audio: false
      });
      video.srcObject = currentStream;
    }

    select.addEventListener("change", () => {
      startCameraPreview();
    });

    function toggleBroadcast() {
      if (isStreaming) {
        stopBroadcast();
      } else {
        startBroadcast();
      }
    }

    function startBroadcast() {
      if (!currentStream) return;

      isStreaming = true;
      broadcastBtn.innerText = "⏹ Stop Streaming";
      broadcastBtn.className = "stop";
      statusDot.className = "dot active";
      statusText.innerText = "LIVE: Broadcasting frames to Frigate YOLOv8 Detection Pipeline";

      // Stream frames at ~15 FPS to Python hub
      streamInterval = setInterval(() => {
        if (!isStreaming || video.videoWidth === 0) return;
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        canvas.toBlob((blob) => {
          if (!blob) return;
          fetch('/push_frame', {
            method: 'POST',
            body: blob
          }).catch(() => {});
        }, 'image/jpeg', 0.8);
      }, 66);
    }

    function stopBroadcast() {
      isStreaming = false;
      if (streamInterval) clearInterval(streamInterval);
      broadcastBtn.innerText = "▶ Start Live Stream";
      broadcastBtn.className = "";
      statusDot.className = "dot";
      statusText.innerText = "Stream stopped. Video preview remains active.";
    }

    window.onload = initCameraDevices;
  </script>
</body>
</html>
"""
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            self.wfile.write(html.encode())
        else:
            self.send_response(404)
            self.end_headers()

def run_hub(port=8080):
    server = ThreadingHTTPServer(('0.0.0.0', port), StreamHubHandler)
    print(f"==================================================")
    print(f" Camera Stream Hub running on http://localhost:{port}")
    print(f" Frigate endpoint: http://host.docker.internal:{port}/live")
    print(f"==================================================")
    server.serve_forever()

if __name__ == '__main__':
    run_hub(8080)
