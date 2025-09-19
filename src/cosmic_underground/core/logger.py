import os
import threading
from datetime import datetime
from .resources import runtime_dir

class AudioGenLogger:
    def __init__(self, filename="audio_generation.log"):
        self._log_path = os.path.join(runtime_dir(), filename)
        self._lock = threading.Lock()
        # Clear log on startup
        with open(self._log_path, "w", encoding="utf-8") as f:
            f.write(f"[{datetime.now().isoformat()}] Log initialized.\n")

    def log(self, message: str):
        with self._lock:
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(f"[{datetime.now().isoformat()}] {message}\n")

# Global instance for easy access
logger = AudioGenLogger()