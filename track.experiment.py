from ultralytics import YOLO
from dotenv import load_dotenv
import os
import cv2
from datetime import datetime
import numpy as np
from collections import deque
import threading
from queue import Queue
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Constants
load_dotenv()
RTSP_URL = os.getenv('RTSP_URL')
PROFILES_DIR = "profiles"
PROCESS_DURATION = 2 * 60
MIN_FRAMES_PER_TRACK = 3
TRACK_BUFFER_TIMEOUT = 120
MAX_BUFFER_SIZE = 10
COMPRESSION_PARAMS = [cv2.IMWRITE_JPEG_QUALITY, 90]

class ProfileManager:
    def __init__(self):
        self.save_queue = Queue()
        self.worker = threading.Thread(target=self._process_queue, daemon=True)
        self.current_date = datetime.now().strftime("%Y-%m-%d")
        self.profile_dir = os.path.join(PROFILES_DIR, self.current_date)
        os.makedirs(self.profile_dir, exist_ok=True)
        self.worker.start()
        self.executor = ThreadPoolExecutor(max_workers=4)

    def _process_queue(self):
        while True:
            item = self.save_queue.get()
            if item is None:
                break
            self._save_profile(*item)
            self.save_queue.task_done()

    def _save_profile(self, face_img, track_id, confidence, timestamp):
        try:
            filename = f"track_{track_id}_{timestamp}_{confidence:.3f}.jpg"
            filepath = os.path.join(self.profile_dir, filename)
            cv2.imwrite(filepath, face_img, COMPRESSION_PARAMS)
            print(f"Saved profile: {filepath}")
        except Exception as e:
            print(f"Error saving profile: {e}")

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

def process_tracks(frames_buffers, track_last_seen, saved_tracks, profile_manager, frame_count):
    expired_tracks = [
        track_id for track_id, last_seen in track_last_seen.items()
        if frame_count - last_seen >= TRACK_BUFFER_TIMEOUT
    ]
    
    for track_id in expired_tracks:
        if track_id not in saved_tracks and frames_buffers[track_id]:
            best_frames = sorted(frames_buffers[track_id], 
                               key=lambda x: x[1], 
                               reverse=True)[:MIN_FRAMES_PER_TRACK]
            
            for face_img, conf in best_frames:
                profile_manager.add_profile(face_img, track_id, conf)
            
            saved_tracks.add(track_id)
        
        del frames_buffers[track_id]
        del track_last_seen[track_id]

def main():
    print("Starting face tracking...")
    model = YOLO("yolov11n-face.pt")
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
                tracker="./new_bytetrack.yml",
                iou=.3,
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
    except Exception as e:
        print(f"Unexpected error: {e}")