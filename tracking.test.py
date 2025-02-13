from ultralytics import YOLO
import cv2
from datetime import datetime
import time
from dotenv import load_dotenv
import os
import yaml
import numpy as np

load_dotenv()
RTSP_URL = os.getenv("RTSP_URL")
if not RTSP_URL:
    raise ValueError("RTSP_URL not found in environment variables")
CONFIG_FILE = "./new_bytetrack.yml"
MODEL_PATH = "./yolov11n-face.pt"
PROCESS_DURATION = 1800  # seconds
try:
    with open(CONFIG_FILE, "r") as f:
        config = yaml.safe_load(f)
        track_buffer = config.get("track_buffer")
        if not track_buffer:
            raise ValueError("track_buffer not found in config")
        print(f"Track buffer: {track_buffer}")
except Exception as e:
    print(f"Error loading config: {e}")
    exit(1)
# Initialize camera parameters
try:
    cap = cv2.VideoCapture(RTSP_URL)
    if not cap.isOpened():
        raise RuntimeError("Failed to open video capture")
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
except Exception as e:
    print(f"Error initializing camera: {e}")
    exit(1)
# Calculate tracking parameters
TRACK_BUFFER_TIMEOUT = int(frame_rate/30.0 * track_buffer)  # frames
print(f"Frame rate: {frame_rate} FPS")
print(f"Track buffer timeout: {TRACK_BUFFER_TIMEOUT} frames")

def create_capture():
    """Create and verify camera capture with retry logic"""
    MAX_RETRIES = 3
    RECONNECT_DELAY = 2  # seconds
    
    def configure_capture(cap):
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 1000)
        return cap

    for attempt in range(MAX_RETRIES):
        try:
            print(f"Attempting to connect to camera ({attempt + 1}/{MAX_RETRIES})...")
            cap = cv2.VideoCapture(RTSP_URL)
            if not cap.isOpened():
                raise RuntimeError("Failed to open video capture")
            
            cap = configure_capture(cap)
            
            # Test read
            ret, frame = cap.read()
            if not ret or frame is None:
                raise RuntimeError("Failed to read test frame")
                
            return cap
        except Exception as e:
            print(f"Connection attempt {attempt + 1} failed: {e}")
            if cap:
                cap.release()
            if attempt < MAX_RETRIES - 1:
                time.sleep(RECONNECT_DELAY)
    
    raise RuntimeError(f"Failed to connect to camera after {MAX_RETRIES} attempts")

def main():
    print("Starting tracking...")
    model = YOLO(MODEL_PATH)
    try:
        track_last_seen = {}
        saved_tracks = set()
        start_time = datetime.now()
        frame_count = 0
        last_frame_time = time.time()
        consecutive_failures = 0
        MAX_CONSECUTIVE_FAILURES = 10
        last_reconnect_time = time.time()
        RECONNECT_INTERVAL = 30  # Force reconnect every 30 seconds if having issues

        cap = create_capture()
        while True:
            if PROCESS_DURATION > 0 and (datetime.now() - start_time).total_seconds() > PROCESS_DURATION:
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Processing duration reached. Stopping...")
                break

            try:
                success, frame = cap.read()
                current_time = time.time()

                if not success or frame is None:
                    consecutive_failures += 1
                    print(f"Failed to read frame: attempt {consecutive_failures}/{MAX_CONSECUTIVE_FAILURES}")
                    
                    # Check if we need to force reconnect
                    if consecutive_failures >= MAX_CONSECUTIVE_FAILURES or \
                       (current_time - last_reconnect_time) > RECONNECT_INTERVAL:
                        print("Attempting to reconnect to camera...")
                        cap.release()
                        cap = create_capture()
                        consecutive_failures = 0
                        last_reconnect_time = current_time
                        time.sleep(1)
                        continue
                    
                    time.sleep(0.1)  # Short delay between retries
                    continue

                # Reset failure counter on successful frame
                consecutive_failures = 0

                # Calculate FPS
                frame_time = current_time - last_frame_time
                fps = 1 / frame_time if frame_time > 0 else 0
                if fps < 5:
                    print(f"Warning: Low FPS detected: {fps:.2f}")

                last_frame_time = current_time
                
                # Process frame
                frame = np.rot90(frame, 3)
                results = next(model.track(
                    source=frame,
                    stream=True,
                    persist=True,
                    tracker=CONFIG_FILE,
                    iou=.35,
                    conf=.5,
                    show=False
                ))
                
                if results.boxes is not None:
                    for box in results.boxes:
                        if box.id is None:
                            continue
                        track_id = int(box.id.item())
                        confidence = float(box.conf.item())
                        track_last_seen[track_id] = frame_count
                        if track_id not in saved_tracks:
                            saved_tracks.add(track_id)
                            print(f"New track detected: {track_id}")
                
                frame_count += 1

            except Exception as e:
                print(f"Error processing frame: {e}")
                consecutive_failures += 1
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    print("Too many failures, attempting to reconnect...")
                    cap.release()
                    cap = create_capture()
                    consecutive_failures = 0
                    last_reconnect_time = time.time()
                time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nStopping tracking...")
    except Exception as e:
        print(f"Fatal error: {e}")
    finally:
        if cap:
            cap.release()

if __name__ == "__main__":
    main()