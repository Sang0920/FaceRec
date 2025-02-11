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
from multiprocessing import Process, Queue, Event
import signal
from api_client import DracoAPIClient
import base64
from queue import Empty, Full
import argparse
import psutil

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
PROCESS_DURATION = 60  # seconds
MIN_PROFILES_PER_TRACK = 3
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
print(f"Frame rate: {frame_rate} FPS")
print(f"Track buffer timeout: {TRACK_BUFFER_TIMEOUT} frames")
Client = DracoAPIClient()
CHECKIN_TYPES = ["IN", "OUT"]
CHECKIN_TYPE = CHECKIN_TYPES[0]
SHIFT_NAME = os.getenv('SHIFT_NAME')

# get the process_duration and checkin_type from the command line (argparse)
def parse_args():
    parser = argparse.ArgumentParser(description='Face Recognition Check-out System')
    parser.add_argument('--process_duration', type=int, default=60,
                      help='Duration for face recognition processing (seconds)')
    parser.add_argument('--checkin_type', type=str, default="IN",
                        help='Check-in type (IN or OUT)')
    args = parser.parse_args()
    return args
args = parse_args()
PROCESS_DURATION = args.process_duration
CHECKIN_TYPE = args.checkin_type
BASE_PROFILE_DIR = os.path.join(PROFILES_DIR, datetime.now().strftime("%Y-%m-%d"), CHECKIN_TYPE)
os.makedirs(BASE_PROFILE_DIR, exist_ok=True)

def get_memory_usage():
    """Get current memory usage of the process"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    return {
        'rss': memory_info.rss / 1024 / 1024,  # RSS in MB
        'vms': memory_info.vms / 1024 / 1024,  # VMS in MB
    }

def signal_handler(signum, frame):
    """Handle shutdown signals and display memory usage"""
    mem_stats = get_memory_usage()
    print("\n=== Memory Usage at Shutdown ===")
    print(f"RSS Memory: {mem_stats['rss']:.2f} MB")
    print(f"Virtual Memory: {mem_stats['vms']:.2f} MB")
    print("=== Shutting down gracefully ===")
    # sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

class ProfileManager:
    def __init__(self):
        self.save_queue = Queue()
        self.worker = threading.Thread(target=self._process_queue, daemon=True)
        self.current_date = datetime.now().strftime("%Y-%m-%d")
        # self.profile_dir = os.path.join(PROFILES_DIR, self.current_date)
        self.profile_dir = BASE_PROFILE_DIR
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

class RecognitionProcess:
    def __init__(self):
        self.input_queue = Queue()
        self.stop_event = Event()
        self.process = Process(target=self._recognition_worker, daemon=True)
        self.process.start()

    def _recognition_worker(self):
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            gallery_features, gallery_names = load_gallery_faces("faces")
            profile_manager = ProfileManager()
            while not self.stop_event.is_set():
                try:
                    track_id, frames = self.input_queue.get(timeout=1)
                    process_track_profiles(
                        {track_id: frames},
                        track_id,
                        profile_manager,
                        gallery_features,
                        gallery_names
                    )
                except Empty:
                    continue
                except Exception as e:
                    print(f"Recognition worker error: {e}")
                    continue
                    
        except Exception as e:
            print(f"Fatal worker error: {e}")
        finally:
            profile_manager.shutdown()

    def add_track(self, track_id, frames):
        if not self.stop_event.is_set() and self.process.is_alive():
            try:
                self.input_queue.put((track_id, frames), timeout=1)
            except Full:
                print(f"Queue full - skipping track {track_id}")
            except Exception as e:
                print(f"Error adding track: {e}")

    def shutdown(self):
        print("Shutting down recognition process...")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Initiating recognition process shutdown...")
        # Show the RAM usage before shutdown
        mem_stats = get_memory_usage()
        print(f"RSS Memory: {mem_stats['rss']:.2f} MB")
        print(f"Virtual Memory: {mem_stats['vms']:.2f} MB")

        self.stop_event.set()
        # Wait for queue to empty
        while not self.input_queue.empty():
            try:
                self.input_queue.get_nowait()
            except Empty:
                break
                
        self.process.join(timeout=5) # secs
        if self.process.is_alive():
            print("Force terminating recognition process...")
            self.process.terminate()
            self.process.join()

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

checkins = set() # set of checkins (email)

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
                           reverse=True)[:MIN_PROFILES_PER_TRACK]
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
        if best_recognition['name'] != "Unknown":
            if best_recognition['name'] in checkins:
                print(f"Already checked in: {best_recognition['name']}")
                return None
            track_dir = os.path.join(profile_manager.profile_dir, f"track_{track_id}")
            json_path = os.path.join(track_dir, "recognition.json")
            with open(json_path, 'w') as f:
                json.dump(best_recognition, f, indent=2)
            try:
                timestamp = best_recognition['timestamp']
                email = best_recognition['name']
                if best_recognition['best_profile']:
                    with open(best_recognition['best_profile'], 'rb') as img_file:
                        base64_image = base64.b64encode(img_file.read()).decode('utf-8')
                else:
                    base64_image = None
                Client.create_checkin(email=email, 
                                      timestamp=timestamp,
                                      log_type=CHECKIN_TYPE,
                                      image_base64=base64_image
                                      )
                checkins.add(best_recognition['name'])
            except Exception as e:
                print(f"Error creating checkin: {e}")
    return best_recognition

def process_tracks(frames_buffers, track_last_seen, saved_tracks, recognition_manager, frame_count):
    expired_tracks = [
        track_id for track_id, last_seen in track_last_seen.items()
        if frame_count - last_seen >= TRACK_BUFFER_TIMEOUT
    ]
    
    for track_id in expired_tracks:
        if track_id not in saved_tracks and frames_buffers[track_id]:
            recognition_manager.add_track(track_id, frames_buffers[track_id])
            saved_tracks.add(track_id)
        del frames_buffers[track_id]
        del track_last_seen[track_id]

def main():
    print("Starting face tracking...")
    model = YOLO(MODEL_PATH)
    recognition_process = RecognitionProcess()
    try:
        Client.sync_employee_photos()
        frames_buffers = {}
        track_last_seen = {}
        saved_tracks = set()
        start_time = datetime.now()
        frame_count = 0
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
                for box in results.boxes:
                    if box.id is None:
                        continue
                    track_id = int(box.id.item())
                    confidence = float(box.conf.item())
                    track_last_seen[track_id] = frame_count
                    if track_id not in frames_buffers:
                        frames_buffers[track_id] = deque(maxlen=MAX_BUFFER_SIZE)
                    face_crop = extract_face(frame, box)
                    if face_crop is not None:
                        frames_buffers[track_id].append((face_crop, confidence))
                expired_tracks = [
                    track_id for track_id, last_seen in track_last_seen.items()
                    if frame_count - last_seen >= TRACK_BUFFER_TIMEOUT
                ]
                for track_id in expired_tracks:
                    if track_id not in saved_tracks and frames_buffers[track_id]:
                        recognition_process.add_track(
                            track_id, 
                            frames_buffers[track_id]
                        )
                        saved_tracks.add(track_id)
                    del frames_buffers[track_id]
                    del track_last_seen[track_id]
            frame_count += 1
            
    except KeyboardInterrupt:
        print("\nStopping tracking...")
    except Exception as e:
        print(f"Error in main: {e}")
    finally:
        # Process any remaining unsaved tracks before shutdown
        for track_id, frames in frames_buffers.items():
            if track_id not in saved_tracks and frames:
                recognition_process.add_track(track_id, frames)
                saved_tracks.add(track_id)

        cap.release()
        recognition_process.shutdown()
        cv2.destroyAllWindows()

if __name__ == "__main__": # python main.py --process_duration 60 --checkin_type IN
    try:
        # shift_details = Client.get_shift_details()
        shift_details = Client.get_shift_details(shift_name=SHIFT_NAME)
        if not shift_details.get('success'):
            raise ValueError(f"Failed to get shift details: {shift_details.get('message')}")
        current_date = datetime.now().strftime("%Y-%m-%d")
        print(f"Current date: {current_date}")  
        holidays = shift_details.get('holiday_list', {}).get('holidays', [])
        print(f'First holiday: {holidays[0]["date"]}')
        is_holiday = any(holiday['date'] == current_date for holiday in holidays)
        if is_holiday:
            print(f"Today ({current_date}) is a holiday. Skipping processing.")
            exit()

        main()
    except Exception as e:
        print(f"Unexpected error: {e}")