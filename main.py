from ultralytics import YOLO
from dotenv import load_dotenv
import os
import sys
import cv2
from datetime import datetime
import numpy as np
from collections import deque
import threading
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
import yaml
from img_recg import load_gallery_faces, recognize_image
from PIL import Image
import json
from multiprocessing import Process, Queue, Event
import signal
from api_client import DracoAPIClient
import base64
from queue import Empty, Full
import argparse
import psutil
from threading import Timer
import time

load_dotenv()
RTSP_URL = os.getenv('RTSP_URL')
if not RTSP_URL:
    raise ValueError("RTSP_URL not found in environment variables")
K_ROTATION = int(os.getenv('K_ROTATION', 3))
PROFILES_DIR = "profiles"
CONFIG_FILE = "./new_bytetrack.yml"
MODEL_PATH = "./yolov11n-face.pt"
MIN_PROFILES_PER_TRACK = 3
MAX_BUFFER_SIZE = 15
NUM_WORKERS = 5
TRACK_CONFIDENCE = float(os.getenv('TRACK_CONFIDENCE', 0.5))
TRACK_IOU = float(os.getenv('TRACK_IOU', 0.45))
RECOGNITION_THRESHOLD = float(os.getenv('RECOGNITION_THRESHOLD', 0.22))
TRACK_MAX_DET = int(os.getenv('TRACK_MAX_DET', 15))
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
try:
    cap = cv2.VideoCapture(RTSP_URL)
    if not cap.isOpened():
        raise RuntimeError("Failed to open video capture")
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
except Exception as e:
    print(f"Error initializing camera: {e}")
    exit(1)
TRACK_BUFFER_TIMEOUT = int(frame_rate/30.0 * track_buffer)  # frames
print(f"Frame rate: {frame_rate} FPS")
print(f"Track buffer timeout: {TRACK_BUFFER_TIMEOUT} frames")
Client = DracoAPIClient()
CHECKIN_TYPES = ["IN", "OUT"]
CHECKIN_TYPE = CHECKIN_TYPES[0]
SHIFT_NAME = os.getenv('SHIFT_NAME')

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
    print("\n=== Memory Usage at Shutdown (signal_handler) ===")
    print(f"RSS Memory: {mem_stats['rss']:.2f} MB")
    print(f"Virtual Memory: {mem_stats['vms']:.2f} MB")
    print("=== Shutting down gracefully ===")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

class ProfileManager:
    def __init__(self):
        self.save_queue = Queue()
        self.worker = threading.Thread(target=self._process_queue, daemon=True)
        self.current_date = datetime.now().strftime("%Y-%m-%d")
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
            gallery_features, gallery_names = load_gallery_faces("./faces")
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
        mem_stats = get_memory_usage()
        print(f"RSS Memory: {mem_stats['rss']:.2f} MB")
        print(f"Virtual Memory: {mem_stats['vms']:.2f} MB")
        self.stop_event.set()
        while not self.input_queue.empty():
            try:
                self.input_queue.get_nowait()
            except Empty:
                break        
        self.process.join(timeout=3) # secs
        if self.process.is_alive():
            print("Force terminating recognition process...")
            self.process.terminate()
            self.process.join()

def extract_face(frame, box, padding: float=0.2):
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
                    threshold=RECOGNITION_THRESHOLD
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
                # Client.create_checkin(email=email, 
                #                       timestamp=timestamp,
                #                       log_type=CHECKIN_TYPE,
                #                       image_base64=base64_image
                #                       )
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

class StreamReader(Process):
    def __init__(self, rtsp_url, frame_queue, stop_event, queue_size=30):
        super().__init__()
        self.rtsp_url = rtsp_url
        self.frame_queue = frame_queue
        self.stop_event = stop_event
        self.queue_size = queue_size
        
    def run(self):
        cap = cv2.VideoCapture(self.rtsp_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000) # Configure FFMPEG buffer size
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 1000)
        while not self.stop_event.is_set():
            try:
                ret, frame = cap.read()
                if not ret or frame is None:
                    time.sleep(0.1)
                    continue
                    
                if self.frame_queue.qsize() < self.queue_size: # Only add frame if queue isn't full
                    self.frame_queue.put(frame)
                else:   # Clear oldest frame if queue is full
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put(frame)
                    except:
                        pass
                        
            except Exception as e:
                print(f"Stream reader error: {e}")
                time.sleep(0.1)
                
        cap.release()

def main():
    print("Starting face tracking...")
    model = YOLO(MODEL_PATH)
    recognition_process = RecognitionProcess()
    frame_queue = Queue(maxsize=30)
    stop_event = Event()
    
    try:
        Client.sync_employee_photos()
        frames_buffers = {}
        track_last_seen = {}
        saved_tracks = set()
        start_time = datetime.now()
        frame_count = 0
        
        stream_reader = StreamReader(RTSP_URL, frame_queue, stop_event)
        stream_reader.start()

        while True:
            print(datetime.now())
            if (datetime.now() - start_time).total_seconds() > PROCESS_DURATION:
                break

            try:
                frame = frame_queue.get(timeout=1.0)  # 1 second timeout
            except Empty:
                print("No frame available, continuing...")
                continue

            frame = np.rot90(frame, K_ROTATION) if K_ROTATION > 0 else frame
            results = next(model.track(
                source=frame,
                stream=True,
                persist=True,
                tracker=CONFIG_FILE,
                iou=TRACK_IOU,
                conf=TRACK_CONFIDENCE,
                max_det=TRACK_MAX_DET,
                classes=[0], # Only detect faces class
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

                expired_tracks = [  # Process any expired tracks
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

        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Processing duration reached. Stopping...")

    except KeyboardInterrupt:
        print("\nKeyboard interrupt. Stopping tracking...")
    except Exception as e:
        print(f"Error in main loop: {e}")
    finally:
        # Cleanup
        stop_event.set()
        stream_reader.join(timeout=3)
        if stream_reader.is_alive():
            stream_reader.terminate()
        
        try:
            for track_id, frames in frames_buffers.items(): # Process any remaining tracks
                if track_id not in saved_tracks and frames:
                    recognition_process.add_track(track_id, frames)
                    saved_tracks.add(track_id)
        except Exception as e:
            print(f"Error processing remaining tracks: {e}")
            
        try:
            recognition_process.shutdown()
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Error during cleanup: {e}")

if __name__ == "__main__": # python main.py --process_duration 60 --checkin_type IN
    try:
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
        os.makedirs(BASE_PROFILE_DIR, exist_ok=True)    
        main()
    except Exception as e:
        print(f"Unexpected error: {e}")