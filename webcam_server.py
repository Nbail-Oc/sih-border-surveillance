"""
IBVAP - Intelligent Camera Streamer & Zone Audio Alert System for Frigate NVR
Features:
- Multi-camera hardware discovery & dynamic switching (Camera 0, 1, 2)
- Adjustable FPS (5 FPS laptop-friendly, 15 FPS, 30 FPS)
- Automatic Zone Collision Audio Player: plays zone-alert.mp3 whenever an object enters a defined zone
"""

import os
import sys
import time
import json
import threading
import urllib.request
import ctypes
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import cv2
import numpy as np

# Audio Alert Configuration
ALERT_SOUND_PATH = os.path.abspath("zone-alert.mp3")
last_sound_play_time = 0.0
SOUND_COOLDOWN_SEC = 3.0
last_zone_breach = None

def play_alert_sound():
    """Play zone-alert.mp3 using Windows Multimedia API (zero extra dependencies)."""
    global last_sound_play_time
    if not os.path.exists(ALERT_SOUND_PATH):
        print(f"[!] Alert sound file not found at: {ALERT_SOUND_PATH}", flush=True)
        return

    try:
        mci = ctypes.windll.winmm.mciSendStringW
        mci("close zone_alert", None, 0, None)
        ret1 = mci(f'open "{ALERT_SOUND_PATH}" type mpegvideo alias zone_alert', None, 0, None)
        ret2 = mci("play zone_alert from 0", None, 0, None)
        last_sound_play_time = time.time()
        print(f"[*] [SOUND ALARM] Playing zone-alert.mp3 (open: {ret1}, play: {ret2})", flush=True)
    except Exception as e:
        print(f"[!] Error playing sound: {e}", flush=True)

def zone_monitor_loop():
    """Poll Frigate's real-time events API to detect when an object enters any defined zone."""
    global last_sound_play_time, last_zone_breach
    print("[*] Zone Collision Audio Monitor started (watching http://localhost:5000/api/events)...", flush=True)

    while running:
        try:
            url = "http://localhost:5000/api/events?in_progress=1"
            req = urllib.request.Request(url, headers={"User-Agent": "Frigate-ZoneAlert/1.0"})
            with urllib.request.urlopen(req, timeout=1.5) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    for event in data:
                        current_zones = event.get("current_zones", [])
                        if current_zones:
                            label = event.get("label", "object")
                            camera = event.get("camera", "webcam")
                            zones_str = ", ".join(current_zones)

                            last_zone_breach = {
                                "camera": camera,
                                "label": label,
                                "zones": current_zones,
                                "timestamp": time.time()
                            }

                            now = time.time()
                            if now - last_sound_play_time >= SOUND_COOLDOWN_SEC:
                                print(f"[*] [ZONE BREACH DETECTED] Camera '{camera}' - Object '{label}' inside zone(s): [{zones_str}]! Triggering sound alarm...", flush=True)
                                threading.Thread(target=play_alert_sound, daemon=True).start()
                                break
        except Exception:
            pass  # Frigate may be starting up or idle

        time.sleep(0.35)  # Poll ~3 times per second

# Global streaming state
current_camera_index = 0
requested_width = 1280
requested_height = 720
target_fps = 30  # Default to 30 FPS high quality (5 FPS cap removed)

latest_jpeg = None
frame_lock = threading.Lock()
frame_condition = threading.Condition(frame_lock)
camera_switch_lock = threading.Lock()
running = True

fps_counter = 0
fps_start_time = time.time()
measured_fps = 30.0

def make_standby_frame(text="Connecting to Camera..."):
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    img[:] = (15, 23, 42)
    cv2.putText(img, "IBVAP - Camera Stream Hub", (380, 320),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(img, text, (410, 380),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (59, 130, 246), 2)
    _, jpeg = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return jpeg.tobytes()

standby_frame = make_standby_frame()
latest_jpeg = standby_frame

def probe_available_cameras(max_probe=4):
    """Detect available camera indices on Windows using DirectShow."""
    available = []
    for i in range(max_probe):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(i, cv2.CAP_MSMF)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            name = f"Camera {i} (Default {w}x{h})"
            available.append({"index": i, "name": name, "width": w, "height": h})
            cap.release()
    if not available:
        # Fallback list so UI always offers options
        available = [
            {"index": 0, "name": "Camera 0 (Default)", "width": 640, "height": 480},
            {"index": 1, "name": "Camera 1 (HD Webcam)", "width": 1280, "height": 720},
            {"index": 2, "name": "Camera 2 (Secondary)", "width": 640, "height": 480}
        ]
    return available

detected_cameras = probe_available_cameras()

def hardware_capture_loop():
    global latest_jpeg, running, fps_counter, fps_start_time, measured_fps
    
    while running:
        with camera_switch_lock:
            cam_idx = current_camera_index
            req_w = requested_width
            req_h = requested_height
            fps_val = target_fps

        print(f"[*] Opening camera index {cam_idx} ({req_w}x{req_h} @ target {fps_val} FPS)...", flush=True)
        cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(cam_idx, cv2.CAP_MSMF)
        if not cap.isOpened():
            cap = cv2.VideoCapture(cam_idx, cv2.CAP_ANY)

        if not cap.isOpened():
            print(f"[!] Could not open camera {cam_idx}", flush=True)
            with frame_condition:
                latest_jpeg = make_standby_frame(f"Camera {cam_idx} Unavailable")
                frame_condition.notify_all()
            time.sleep(2)
            continue

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, req_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, req_h)
        cap.set(cv2.CAP_PROP_FPS, fps_val)

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[+] Camera {cam_idx} active: {actual_w}x{actual_h} hardware stream running", flush=True)

        while running:
            with camera_switch_lock:
                if (current_camera_index != cam_idx or 
                    requested_width != req_w or 
                    requested_height != req_h or 
                    target_fps != fps_val):
                    print("[*] Settings changed, reloading capture device...", flush=True)
                    break

            loop_start = time.time()
            ret, frame = cap.read()

            if not ret or frame is None:
                time.sleep(0.01)
                continue

            ret, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if ret:
                frame_bytes = encoded.tobytes()
                with frame_condition:
                    latest_jpeg = frame_bytes
                    frame_condition.notify_all()

                fps_counter += 1
                now = time.time()
                if now - fps_start_time >= 1.0:
                    measured_fps = round(fps_counter / (now - fps_start_time), 1)
                    fps_counter = 0
                    fps_start_time = now

            # Pacing to match requested target_fps
            target_interval = 1.0 / max(target_fps, 1)
            elapsed = time.time() - loop_start
            sleep_time = target_interval - elapsed
            if sleep_time > 0.002:
                time.sleep(sleep_time)

        cap.release()

UI_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>IBVAP - Camera Stream Hub & Zone Audio Alarm</title>
  <style>
    :root {
      --bg: #0b0f19;
      --card: #131b2e;
      --accent: #3b82f6;
      --accent-hover: #2563eb;
      --success: #10b981;
      --danger: #ef4444;
      --text: #f8fafc;
      --muted: #94a3b8;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: var(--bg);
      color: var(--text);
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      min-height: 100vh;
      padding: 20px;
    }
    .card {
      background: var(--card);
      border-radius: 16px;
      padding: 24px;
      width: 100%;
      max-width: 920px;
      box-shadow: 0 20px 40px rgba(0,0,0,0.6);
      border: 1px solid rgba(255,255,255,0.08);
    }
    .header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 16px;
    }
    h1 {
      font-size: 1.35rem;
      margin: 0;
      display: flex;
      align-items: center;
      gap: 10px;
    }
    .badges {
      display: flex;
      gap: 8px;
    }
    .badge {
      background: rgba(16, 185, 129, 0.15);
      color: #34d399;
      border: 1px solid rgba(16, 185, 129, 0.3);
      padding: 4px 12px;
      border-radius: 9999px;
      font-size: 0.8rem;
      font-weight: 600;
    }
    .badge-alarm {
      background: rgba(239, 68, 68, 0.15);
      color: #f87171;
      border: 1px solid rgba(239, 68, 68, 0.3);
    }
    .alert-banner {
      display: none;
      background: #7f1d1d;
      color: #fecaca;
      border: 1px solid #ef4444;
      padding: 12px 16px;
      border-radius: 10px;
      margin-bottom: 16px;
      font-weight: 600;
      animation: pulse 1s infinite alternate;
    }
    @keyframes pulse {
      from { opacity: 0.85; transform: scale(0.995); }
      to { opacity: 1; transform: scale(1); }
    }
    .controls {
      display: flex;
      gap: 12px;
      margin-bottom: 16px;
      flex-wrap: wrap;
    }
    .control-group {
      flex: 1;
      min-width: 180px;
      display: flex;
      flex-direction: column;
      gap: 6px;
    }
    label {
      font-size: 0.75rem;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }
    select, button {
      padding: 10px 14px;
      background: #1e293b;
      color: var(--text);
      border: 1px solid rgba(255,255,255,0.15);
      border-radius: 8px;
      font-size: 0.95rem;
      outline: none;
      cursor: pointer;
    }
    button.btn-sound {
      background: #2563eb;
      color: white;
      font-weight: 600;
      border: none;
      transition: background 0.2s;
    }
    button.btn-sound:hover {
      background: #1d4ed8;
    }
    .video-box {
      position: relative;
      width: 100%;
      aspect-ratio: 16 / 9;
      background: #020617;
      border-radius: 12px;
      overflow: hidden;
      border: 1px solid rgba(255,255,255,0.1);
      display: flex;
      align-items: center;
      justify-content: center;
    }
    .video-box img {
      width: 100%;
      height: 100%;
      object-fit: cover;
    }
    .overlay-fps {
      position: absolute;
      top: 12px;
      right: 12px;
      background: rgba(0,0,0,0.65);
      padding: 4px 10px;
      border-radius: 6px;
      font-size: 0.8rem;
      font-family: monospace;
      color: #34d399;
      border: 1px solid rgba(255,255,255,0.1);
    }
    .footer {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-top: 16px;
      font-size: 0.85rem;
      color: var(--muted);
    }
    .btn-frigate {
      background: var(--accent);
      color: #fff;
      padding: 8px 18px;
      border-radius: 8px;
      text-decoration: none;
      font-weight: 600;
    }
    code {
      background: #0f172a;
      padding: 2px 6px;
      border-radius: 4px;
      color: #38bdf8;
    }
  </style>
</head>
<body>
  <div class="card">
    <div class="header">
      <h1>📹 Camera Streamer & Zone Audio Alert</h1>
      <div class="badges">
        <div class="badge">● LIVE STREAM</div>
        <div class="badge badge-alarm">🔊 AUDIO ALARM: ACTIVE</div>
      </div>
    </div>

    <div id="alertBanner" class="alert-banner">
      🚨 ZONE INTRUSION DETECTED! Playing zone-alert.mp3
    </div>

    <div class="controls">
      <div class="control-group">
        <label>Active Camera Device:</label>
        <select id="cameraSelect" onchange="switchCamera(this.value)">
          <!-- Populated dynamically -->
        </select>
      </div>

      <div class="control-group">
        <label>Stream FPS:</label>
        <select id="fpsSelect" onchange="switchFps(this.value)">
          <option value="30" selected>30 FPS (Full Speed / Smooth)</option>
          <option value="15">15 FPS (Balanced)</option>
          <option value="5">5 FPS (Battery Saver)</option>
        </select>
      </div>

      <div class="control-group">
        <label>Resolution:</label>
        <select id="resolutionSelect" onchange="switchResolution(this.value)">
          <option value="1280x720" selected>1280 x 720 (720p HD)</option>
          <option value="640x480">640 x 480 (SD)</option>
        </select>
      </div>

      <div class="control-group" style="justify-content: flex-end;">
        <button class="btn-sound" onclick="testSound()">▶ Test zone-alert.mp3</button>
      </div>
    </div>

    <div class="video-box">
      <img id="liveFeed" src="/live" alt="Hardware Live Feed">
      <div class="overlay-fps" id="overlayFps">5.0 FPS</div>
    </div>

    <div class="footer">
      <div>Feeding Frigate: <code>http://host.docker.internal:8080/live</code></div>
      <a href="http://localhost:5000/#webcam" target="_blank" class="btn-frigate">Open Frigate Footage ↗</a>
    </div>
  </div>

  <script>
    async function loadCameras() {
      try {
        const res = await fetch('/api/cameras');
        const data = await res.json();
        const sel = document.getElementById('cameraSelect');
        sel.innerHTML = '';
        data.cameras.forEach(c => {
          const opt = document.createElement('option');
          opt.value = c.index;
          opt.text = c.name;
          if (c.index === data.current) opt.selected = true;
          sel.appendChild(opt);
        });
      } catch (e) {}
    }

    async function switchCamera(idx) {
      await fetch('/api/switch_camera?index=' + idx);
      refreshImage();
    }

    async function switchFps(fps) {
      await fetch('/api/switch_fps?fps=' + fps);
    }

    async function switchResolution(res) {
      const [w, h] = res.split('x');
      await fetch(`/api/switch_resolution?width=${w}&height=${h}`);
      refreshImage();
    }

    async function testSound() {
      await fetch('/api/test_sound');
    }

    function refreshImage() {
      const img = document.getElementById('liveFeed');
      img.src = '/live?t=' + Date.now();
    }

    async function pollStats() {
      try {
        const res = await fetch('/api/stats');
        const data = await res.json();
        document.getElementById('overlayFps').innerText = data.fps + ' FPS';
        
        const banner = document.getElementById('alertBanner');
        if (data.breach && (Date.now() / 1000 - data.breach.timestamp < 4)) {
          banner.style.display = 'block';
          banner.innerText = `🚨 ZONE INTRUSION DETECTED! Object: '${data.breach.label}' in Zone: [${data.breach.zones.join(', ')}]`;
        } else {
          banner.style.display = 'none';
        }
      } catch (e) {}
    }

    setInterval(pollStats, 800);
    window.onload = loadCameras;
  </script>
</body>
</html>
"""

class HardwareStreamHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        global current_camera_index, requested_width, requested_height, target_fps
        parsed = urlparse(self.path)

        if parsed.path == '/live':
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

        elif parsed.path == '/api/cameras':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            resp = {
                "cameras": detected_cameras,
                "current": current_camera_index
            }
            self.wfile.write(json.dumps(resp).encode('utf-8'))

        elif parsed.path == '/api/switch_camera':
            qs = parse_qs(parsed.query)
            if 'index' in qs:
                new_idx = int(qs['index'][0])
                with camera_switch_lock:
                    current_camera_index = new_idx
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(b'{"status": "ok"}')

        elif parsed.path == '/api/switch_fps':
            qs = parse_qs(parsed.query)
            if 'fps' in qs:
                new_fps = int(qs['fps'][0])
                with camera_switch_lock:
                    target_fps = new_fps
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(b'{"status": "ok"}')

        elif parsed.path == '/api/switch_resolution':
            qs = parse_qs(parsed.query)
            if 'width' in qs and 'height' in qs:
                with camera_switch_lock:
                    requested_width = int(qs['width'][0])
                    requested_height = int(qs['height'][0])
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(b'{"status": "ok"}')

        elif parsed.path == '/api/test_sound':
            threading.Thread(target=play_alert_sound, daemon=True).start()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(b'{"status": "playing"}')

        elif parsed.path == '/api/stats':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            resp = {
                "fps": measured_fps,
                "target_fps": target_fps,
                "camera": current_camera_index,
                "breach": last_zone_breach
            }
            self.wfile.write(json.dumps(resp).encode('utf-8'))

        elif parsed.path in ('/', '/index.html'):
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(UI_HTML.encode('utf-8'))

        else:
            self.send_response(404)
            self.end_headers()

def start_server(port=8080):
    capture_thread = threading.Thread(target=hardware_capture_loop, daemon=True)
    capture_thread.start()

    monitor_thread = threading.Thread(target=zone_monitor_loop, daemon=True)
    monitor_thread.start()

    print("=" * 65)
    print(f"[*] Camera Hub & Zone Audio Alarm active on port {port}")
    print(f"[*] Web Control & Alarm Test: http://localhost:{port}")
    print(f"[*] Frigate Video Feed:       http://host.docker.internal:{port}/live")
    print(f"[*] Audio Alert File:         {ALERT_SOUND_PATH}")
    print("=" * 65)

    server = ThreadingHTTPServer(('0.0.0.0', port), HardwareStreamHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[*] Stopping server...")
    finally:
        global running
        running = False
        server.server_close()

if __name__ == '__main__':
    start_server(8080)
