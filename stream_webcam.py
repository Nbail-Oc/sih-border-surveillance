"""Experimental RTSP publisher for local webcam testing.

This is not the Windows Docker demo path. Frigate's published RTSP port is a
restream/read endpoint, so publishing to it requires a separately configured
RTSP ingest server. Use ``webcam_server.py`` with the MJPEG input in
``config/config.yml`` for the supported browser-based demo.
"""

import sys
import subprocess
import cv2

def list_cameras():
    """Find available camera indexes."""
    available = []
    for i in range(5):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            available.append(i)
            cap.release()
    return available

def stream(camera_index=0, width=1280, height=720, fps=15):
    print(f"Opening camera index {camera_index} at {width}x{height} @ {fps}fps...")
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_index}")
        return

    # FFmpeg pipe pushing RTSP / raw stream to Frigate go2rtc
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-tune", "zerolatency",
        "-pix_fmt", "yuv420p",
        "-f", "rtsp",
        "-rtsp_transport", "tcp",
        "rtsp://127.0.0.1:8554/live_webcam"
    ]

    print("Launching RTSP stream to Frigate (rtsp://127.0.0.1:8554/live_webcam)...")
    print("Press Ctrl+C to stop streaming.")
    
    try:
        proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            proc.stdin.write(frame.tobytes())
    except KeyboardInterrupt:
        print("\nStopping camera stream...")
    finally:
        cap.release()
        if 'proc' in locals():
            proc.stdin.close()
            proc.wait()

if __name__ == "__main__":
    cams = list_cameras()
    print(f"Detected cameras: {cams}")
    cam_id = cams[0] if cams else 0
    if len(sys.argv) > 1:
        cam_id = int(sys.argv[1])
    stream(camera_index=cam_id)
