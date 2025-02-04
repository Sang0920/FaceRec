from ultralytics import YOLO
from dotenv import load_dotenv
import os
import cv2
from datetime import datetime
import numpy as np
from collections import deque
import threading
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
import yaml
# import asyncio
from img_recg import load_gallery_faces, recognize_image, upscale_image
from PIL import Image
import json

load_dotenv()
# Environment and configuration
RTSP_URL = os.getenv('RTSP_URL')
if not RTSP_URL:
    raise ValueError("RTSP_URL not found in environment variables")
# File paths and directories 
PROFILES_DIR = "profiles"
CONFIG_FILE = "./new_bytetrack.yml"
MODEL_PATH = "./yolov11n-face.pt"
# Processing parameters
PROCESS_DURATION = 30  # seconds
MIN_FRAMES_PER_TRACK = 3
MAX_BUFFER_SIZE = 15
NUM_WORKERS = 5
# Load tracker configuration
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
print(f"Frame rate: {frame_rate}")
print(f"Track buffer timeout: {TRACK_BUFFER_TIMEOUT}")

class ProfileManager:
    def __init__(self):
        self.save_queue = Queue()
        self.worker = threading.Thread(target=self._process_queue, daemon=True)
        self.current_date = datetime.now().strftime("%Y-%m-%d")
        self.profile_dir = os.path.join(PROFILES_DIR, self.current_date)
        os.makedirs(self.profile_dir, exist_ok=True)
        self.worker.start()
        self.executor = ThreadPoolExecutor(max_workers=NUM_WORKERS)

    def _process_queue(self):
        while True:
            item = self.save_queue.get()
            if item is None:
                break
            self._save_profile(*item)
            self.save_queue.task_done()

    def _save_profile(self, face_img, track_id, confidence, timestamp):
        try:
            track_dir = os.path.join(self.profile_dir, f"track_{track_id}")
            os.makedirs(track_dir, exist_ok=True)
            
            filename = f"profile_{timestamp}_{confidence:.3f}.png"
            filepath = os.path.join(track_dir, filename)
            cv2.imwrite(filepath, face_img)
            print(f"Saved profile: {filepath}")
            return filepath
        except Exception as e:
            print(f"Error saving profile: {e}")
            return None

    def add_profile(self, face_img, track_id, confidence):
        timestamp = datetime.now().strftime("%H-%M-%S-%f")
        self.save_queue.put((face_img, track_id, confidence, timestamp))

    def shutdown(self):
        self.save_queue.put(None)
        self.worker.join()
        self.executor.shutdown()

def extract_face(frame, box, padding=0.2):
    try:
        coords = box.xyxy[0].cpu().numpy().astype(np.int32)
        x1, y1, x2, y2 = coords
        face_size = max(x2 - x1, y2 - y1)
        pad = int(face_size * padding)
        h, w = frame.shape[:2]
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)
        return frame[y1:y2, x1:x2]
    except Exception as e:
        print(f"Error extracting face: {e}")
        return None

def process_track_profiles(frames_buffers, track_id, profile_manager, gallery_features, gallery_names):
    best_recognition = {
        'confidence': 0,
        'name': "Unknown",
        'timestamp': None,
        'profiles': []
    }
    
    if frames_buffers[track_id]:
        best_frames = sorted(frames_buffers[track_id], 
                           key=lambda x: x[1], 
                           reverse=True)[:MIN_FRAMES_PER_TRACK]
        
        for face_img, conf in best_frames:
            try:
                timestamp = datetime.now().strftime("%H-%M-%S-%f")
                profile_path = profile_manager._save_profile(face_img, track_id, conf, timestamp)
                if profile_path:
                    best_recognition['profiles'].append(profile_path)                
                pil_img = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
                names, confidences, _, _ = recognize_image(
                    pil_img,
                    gallery_features,
                    gallery_names,
                    threshold=0.19
                )
                print(f"Track {track_id}: Recognized as {names} ({confidences})")
                if names and confidences and names[0] != "Unknown":
                    if confidences[0] > best_recognition['confidence']:
                        best_recognition.update({
                            'confidence': float(confidences[0]),
                            'name': names[0],
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'best_profile': profile_path
                        })
                        print(f"Track {track_id}: Recognized as {names[0]} ({confidences[0]:.3f})")
            except Exception as e:
                print(f"Error processing recognition for track {track_id}: {e}")
        # Save recognition results in track directory
        if best_recognition['name'] != "Unknown":
            track_dir = os.path.join(profile_manager.profile_dir, f"track_{track_id}")
            json_path = os.path.join(track_dir, "recognition.json")
            with open(json_path, 'w') as f:
                json.dump(best_recognition, f, indent=2)
    return best_recognition

def process_tracks(frames_buffers, track_last_seen, saved_tracks, profile_manager, frame_count):
    expired_tracks = [
        track_id for track_id, last_seen in track_last_seen.items()
        if frame_count - last_seen >= TRACK_BUFFER_TIMEOUT
    ]
    # Load gallery once
    gallery_features, gallery_names = load_gallery_faces("faces")
    for track_id in expired_tracks:
        if track_id not in saved_tracks and frames_buffers[track_id]:
            recognition = process_track_profiles(
                frames_buffers,
                track_id,
                profile_manager,
                gallery_features,
                gallery_names
            )
            saved_tracks.add(track_id)
        del frames_buffers[track_id]
        del track_last_seen[track_id]

def main():
    print("Starting face tracking...")
    model = YOLO(MODEL_PATH)
    profile_manager = ProfileManager()
    frames_buffers = {}
    track_last_seen = {}
    saved_tracks = set()
    start_time = datetime.now()
    frame_count = 0
    try:
        cap = cv2.VideoCapture(RTSP_URL)
        while (datetime.now() - start_time).total_seconds() <= PROCESS_DURATION:
            success, frame = cap.read()
            if not success:
                continue
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
                current_tracks = set()
                for box in results.boxes:
                    if box.id is None:
                        continue
                    track_id = int(box.id.item())
                    current_tracks.add(track_id)
                    confidence = float(box.conf.item())
                    track_last_seen[track_id] = frame_count
                    if track_id not in frames_buffers:
                        frames_buffers[track_id] = deque(maxlen=MAX_BUFFER_SIZE)
                    face_crop = extract_face(frame, box)
                    if face_crop is not None:
                        frames_buffers[track_id].append((face_crop, confidence))
                process_tracks(frames_buffers, track_last_seen, saved_tracks, 
                             profile_manager, frame_count)
            frame_count += 1
    except KeyboardInterrupt:
        print("Stopping tracking...")
    finally:
        for track_id, buffer in frames_buffers.items():
            if track_id not in saved_tracks and buffer:
                best_frames = sorted(buffer, key=lambda x: x[1], 
                                  reverse=True)[:MIN_FRAMES_PER_TRACK]
                for face_img, conf in best_frames:
                    profile_manager.add_profile(face_img, track_id, conf)
        profile_manager.shutdown()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        main()
        # del TRACK_BUFFER_TIMEOUT
    except Exception as e:
        print(f"Unexpected error: {e}")