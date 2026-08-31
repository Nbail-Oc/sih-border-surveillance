"""
Resilient multithreaded camera streamer for Frigate NVR.
Usage: python camera_stream.py [cam_index]
"""
import sys
import cv2
import time
import threading
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

cam_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0
print(f"[*] Opening camera index {cam_index} (use 'python camera_stream.py 1' for secondary/USB camera)...")
cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 15)

if not cap.isOpened():
    print(f"[!] Warning: Camera {cam_index} DirectShow failed. Retrying default backend...")
    cap = cv2.VideoCapture(cam_index)

cap_lock = threading.Lock()

class StreamHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        try:
            self.send_response(200)
            self.send_header('Age', '0')
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
            self.end_headers()

            while True:
                with cap_lock:
                    ret, frame = cap.read() if cap and cap.isOpened() else (False, None)

                if not ret or frame is None:
                    time.sleep(0.04)
                    continue

                _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                self.wfile.write(b'--FRAME\r\n')
                self.send_header('Content-Type', 'image/jpeg')
                self.send_header('Content-Length', str(len(jpeg)))
                self.end_headers()
                self.wfile.write(jpeg.tobytes())
                self.wfile.write(b'\r\n')
                time.sleep(0.04) # ~20 FPS limit
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

if __name__ == '__main__':
    port = 8080
    print(f"[*] Live stream active at: http://localhost:{port}/live")
    print(f"[*] Feeding Frigate at:    http://host.docker.internal:{port}/live")
    server = ThreadingHTTPServer(('0.0.0.0', port), StreamHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping camera server...")
    finally:
        cap.release()
