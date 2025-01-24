from ultralytics import YOLO
from dotenv import load_dotenv
import os
import cv2
from datetime import datetime, timedelta
import numpy as np
# import logging
from PIL import Image

load_dotenv()
RTSP_URL = os.getenv('RTSP_URL')
PROFILES_DIR = "profiles"
# PROCESS_DURATION = 1 * 60  # 1 minutes in seconds
PROCESS_DURATION = 2 * 60 
MIN_FRAMES_PER_TRACK = 3
TRACK_BUFFER_TIMEOUT = 120  # frames

# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler('track.log'),
#         logging.StreamHandler()
#     ]
# )
# logger = logging.getLogger(__name__)

last_seen = {}
best_profiles = {}  # track_id -> (confidence, frame, box)

def save_profile(frame, track_id, confidence, box):
    try:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])        
        face_size = max(x2 - x1, y2 - y1)
        pad = int(face_size * 0.2)  # 20% padding
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)
        face_img = frame[y1:y2, x1:x2] # Extract face region
        current_date = datetime.now().strftime("%Y-%m-%d")
        profile_dir = os.path.join(PROFILES_DIR, current_date)
        os.makedirs(profile_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%H-%M-%S")
        filename = f"track_{track_id}_{timestamp}_{confidence:.3f}.jpg"
        filepath = os.path.join(profile_dir, filename)
        if face_img is not None and face_img.size > 0:
            cv2.imwrite(filepath, face_img)
            # logger.info(f"Saved profile: {filepath}")
            print(f"Saved profile: {filepath}")
            return filepath
        return None
    except Exception as e:
        # logger.error(f"Error saving profile: {e}")
        print(f"Error saving profile: {e}")
        return None

def save_best_profiles(track_id, frames_buffer, saved_tracks):
    """Save top 3 frames by confidence for a track"""
    if not frames_buffer or track_id in saved_tracks:
        return
    sorted_frames = sorted(frames_buffer, key=lambda x: x[1], reverse=True)
    best_frames = sorted_frames[:MIN_FRAMES_PER_TRACK]
    try:
        for idx, (frame, conf, box) in enumerate(best_frames):
            timestamp = datetime.now().strftime("%H-%M-%S-%f")
            profile_path = save_profile(frame, track_id, conf, box)
            if profile_path:
                # logger.info(f"Saved profile {idx} for track {track_id} (conf: {conf:.3f})")
                print(f"Saved profile {idx} for track {track_id} (conf: {conf:.3f})")
        # Mark track as saved and clear its buffer
        saved_tracks.add(track_id)
        frames_buffer.clear()
    except Exception as e:
        # logger.error(f"Error saving best profiles for track {track_id}: {e}")
        print(f"Error saving best profiles for track {track_id}: {e}")

def main():
    # logger.info("Starting face tracking...")
    print("Starting face tracking...")
    os.makedirs(PROFILES_DIR, exist_ok=True)
    model = YOLO("yolov11n-face.pt")
    frames_buffers = {}
    track_last_seen = {}
    saved_tracks = set()
    start_time = datetime.now()
    frame_count = 0
    try:
        cap = cv2.VideoCapture(RTSP_URL)
        while cap.isOpened():
            current_time = datetime.now() # Check duration first
            if (current_time - start_time).total_seconds() > PROCESS_DURATION:
                for track_id in frames_buffers: # Save remaining tracks before exit
                    save_best_profiles(track_id, frames_buffers[track_id], saved_tracks)
                # logger.info(f"Process duration {PROCESS_DURATION}s reached, stopping...")
                print(f"Process duration {PROCESS_DURATION}s reached, stopping...")
                break
            success, frame = cap.read()
            if not success:
                # logger.warning("Can't receive frame from stream. Retrying...")
                print("Can't receive frame from stream. Retrying...")
                continue
            frame = np.rot90(frame, 3)
            for results in model.track(source=frame,
                                    stream=True,
                                    persist=True,
                                    tracker="./new_bytetrack.yml",
                                    iou=.3,
                                    conf=.5,
                                    show=False,
                                    ):
                current_time = datetime.now()
                if (current_time - start_time).total_seconds() > PROCESS_DURATION:
                    for track_id in frames_buffers: # Save all remaining tracks and exit
                        save_best_profiles(track_id, frames_buffers[track_id], saved_tracks)
                    break
                if results.boxes is None:
                    continue

                current_tracks = set()
                for box in results.boxes:
                    if box.id is None:
                        continue
                    track_id = int(box.id.item())
                    current_tracks.add(track_id)
                    confidence = float(box.conf.item())
                    track_last_seen[track_id] = frame_count
                    if track_id not in frames_buffers:
                        frames_buffers[track_id] = []
                    frames_buffers[track_id].append((results.orig_img.copy(), confidence, box))

                for track_id in list(track_last_seen.keys()):
                    if track_id not in current_tracks and \
                       frame_count - track_last_seen[track_id] >= TRACK_BUFFER_TIMEOUT:
                        save_best_profiles(track_id, frames_buffers[track_id], saved_tracks)
                        del frames_buffers[track_id]
                        del track_last_seen[track_id]

                frame_count += 1

    except KeyboardInterrupt:
        # logger.info("Stopping tracking...")
        print("Stopping tracking...")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # logger.info("Shutting down gracefully...")
        print("Shutting down gracefully...")
    except Exception as e:
        # logger.error(f"Unexpected error: {e}")
        print(f"Unexpected error: {e}")