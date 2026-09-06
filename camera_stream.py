"""
Resilient multithreaded camera streamer for Frigate NVR.
Captures local webcam video and streams MJPEG over HTTP.

Usage:
  python camera_stream.py [cam_index] [port]
Example:
  python camera_stream.py 0 8080
"""

import sys
import time
import threading
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import cv2
import numpy as np

# Configuration
cam_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0
port = int(sys.argv[2]) if len(sys.argv) > 2 else 8080
width = 1280
height = 720
target_fps = 20

latest_jpeg = None
frame_lock = threading.Lock()
frame_condition = threading.Condition(frame_lock)
running = True

def make_standby_frame(text="Connecting to Camera..."):
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = (15, 23, 42)  # Slate dark background
    cv2.putText(img, "Frigate Webcam Streamer", (width // 2 - 250, height // 2 - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(img, text, (width // 2 - 180, height // 2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (59, 130, 246), 2)
    _, jpeg = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return jpeg.tobytes()

standby_frame = make_standby_frame()
latest_jpeg = standby_frame

def camera_capture_loop(device_id):
    global latest_jpeg, running
    print(f"[*] Initializing camera index {device_id} (DirectShow)...")
    
    cap = cv2.VideoCapture(device_id, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, target_fps)

    if not cap.isOpened():
        print(f"[!] DirectShow failed on camera {device_id}. Retrying standard backend...")
        cap = cv2.VideoCapture(device_id)

    if not cap.isOpened():
        print(f"[!] ERROR: Could not open camera {device_id}. Please check device connection/index.")
        with frame_condition:
            latest_jpeg = make_standby_frame(f"Camera {device_id} Not Available")
            frame_condition.notify_all()
        return

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[+] Camera {device_id} successfully opened: {actual_w}x{actual_h}")

    sleep_duration = 1.0 / target_fps

    while running:
        start_time = time.time()
        ret, frame = cap.read()
        
        if not ret or frame is None:
            time.sleep(0.1)
            continue

        ret, encoded_jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ret:
            with frame_condition:
                latest_jpeg = encoded_jpeg.tobytes()
                frame_condition.notify_all()

        elapsed = time.time() - start_time
        remaining = sleep_duration - elapsed
        if remaining > 0:
            time.sleep(remaining)

    cap.release()
    print("[*] Camera capture thread stopped.")

PREVIEW_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Live Webcam Streamer - Frigate</title>
  <style>
    body {
      margin: 0;
      background: #0f172a;
      color: #f8fafc;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      min-height: 100vh;
      padding: 20px;
      box-sizing: border-box;
    }
    .card {
      background: #1e293b;
      border-radius: 16px;
      padding: 24px;
      width: 100%;
      max-width: 900px;
      box-shadow: 0 20px 25px -5px rgba(0,0,0,0.5);
      border: 1px solid #334155;
    }
    .header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 16px;
    }
    .title {
      font-size: 1.4rem;
      font-weight: 700;
    }
    .badge {
      background: #22c55e20;
      color: #4ade80;
      border: 1px solid #22c55e50;
      padding: 4px 12px;
      border-radius: 9999px;
      font-size: 0.85rem;
      font-weight: 600;
    }
    .video-box {
      border-radius: 12px;
      overflow: hidden;
      background: #020617;
      aspect-ratio: 16 / 9;
      display: flex;
      align-items: center;
      justify-content: center;
      border: 1px solid #334155;
    }
    .video-box img {
      width: 100%;
      height: 100%;
      object-fit: cover;
    }
    .footer {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-top: 16px;
      font-size: 0.9rem;
      color: #94a3b8;
    }
    .btn {
      background: #3b82f6;
      color: #fff;
      padding: 8px 18px;
      border-radius: 8px;
      text-decoration: none;
      font-weight: 600;
      transition: background 0.2s;
    }
    .btn:hover {
      background: #2563eb;
    }
    code {
      background: #0f172a;
      padding: 2px 6px;
      border-radius: 4px;
      color: #38bdf8;
      font-size: 0.85rem;
    }
  </style>
</head>
<body>
  <div class="card">
    <div class="header">
      <div class="title">📹 Live Webcam Streamer</div>
      <div class="badge">● STREAM ACTIVE</div>
    </div>
    <div class="video-box">
      <img src="/live" alt="Live Camera Feed">
    </div>
    <div class="footer">
      <div>Feeding Frigate at: <code>http://host.docker.internal:8080/live</code></div>
      <a href="http://localhost:5000" target="_blank" class="btn">Open Frigate UI ↗</a>
    </div>
  </div>
</body>
</html>
"""

class StreamHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # Silence routine frame log requests

    def do_GET(self):
        global latest_jpeg, running
        if self.path == '/live':
            self.send_response(200)
            self.send_header('Age', '0')
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
            self.end_headers()

            while running:
                with frame_condition:
                    frame_condition.wait(timeout=0.1)
                    frame_bytes = latest_jpeg

                if frame_bytes is None:
                    continue

                try:
                    self.wfile.write(b'--FRAME\r\n')
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', str(len(frame_bytes)))
                    self.end_headers()
                    self.wfile.write(frame_bytes)
                    self.wfile.write(b'\r\n')
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    break
        elif self.path == '/' or self.path.startswith('/index'):
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.send_header('Content-Length', str(len(PREVIEW_HTML)))
            self.end_headers()
            self.wfile.write(PREVIEW_HTML.encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

if __name__ == '__main__':
    # Start dedicated camera capture thread
    cam_thread = threading.Thread(target=camera_capture_loop, args=(cam_index,), daemon=True)
    cam_thread.start()

    print("=" * 60)
    print(f"[*] Webcam Streamer started on port {port}")
    print(f"[*] Web Preview:     http://localhost:{port}")
    print(f"[*] MJPEG Endpoint:  http://localhost:{port}/live")
    print(f"[*] Docker Endpoint: http://host.docker.internal:{port}/live")
    print(f"[*] Frigate UI:      http://localhost:5000")
    print("=" * 60)

    server = ThreadingHTTPServer(('0.0.0.0', port), StreamHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[*] Stopping camera server...")
    finally:
        running = False
        server.server_close()
