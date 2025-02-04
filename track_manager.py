from multiprocessing import Process, Queue, Manager
import numpy as np
from datetime import datetime

class ProfileWorker(Process):
    def __init__(self, task_queue, result_queue):
        super().__init__()
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.gallery_features = None
        self.gallery_names = None
        
    def run(self):
        self.gallery_features, self.gallery_names = load_gallery_faces("faces")
        while True:
            task = self.task_queue.get()
            if task is None:
                break
            track_id, frames = task
            self.process_track(track_id, frames)
            
    def process_track(self, track_id, frames):
        try:
            profiles = process_track_profiles(
                {track_id: frames},
                track_id,
                self.profile_manager,
                self.gallery_features,
                self.gallery_names
            )
            self.result_queue.put((track_id, profiles))
        except Exception as e:
            print(f"Worker error processing track {track_id}: {e}")