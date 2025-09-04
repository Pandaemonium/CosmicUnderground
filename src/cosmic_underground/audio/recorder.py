import pygame, os, threading, re
from cosmic_underground.core import config as C
from typing import Optional, Tuple, Dict, List, Any
import time
import wave
import uuid

import threading, traceback as _tb
def _thread_excepthook(args):
    print(f"\n[THREAD-ERROR] {args.thread.name}: {args.exc_type.__name__}: {args.exc_value}")
    _tb.print_tb(args.exc_traceback)
threading.excepthook = _thread_excepthook

class Recorder:
    def __init__(self):
        self.dir = os.path.abspath("./inventory")
        os.makedirs(self.dir, exist_ok=True)
        self.thread = None
        self.stop_flag = threading.Event()
        self.active_path = None

    @staticmethod
    def _safe_label(text: str, maxlen: int = 80) -> str:
        """
        Make a Windows-safe filename fragment:
        - Replace \ / : * ? " < > | with _
        - Collapse whitespace to single spaces, then convert spaces to _
        - Trim to a reasonable length
        """
        s = re.sub(r'[\\/:*?"<>|]', "_", str(text))
        s = re.sub(r"\s+", " ", s).strip()
        s = s.replace(" ", "_")
        return s[:maxlen]


    def is_recording(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def start(self, loop_path: str, max_seconds: float, meta: Dict):
        if self.is_recording():
            return
        self.stop_flag.clear()

        # NEW: prefer 'label' if provided, else fall back to 'zone'
        raw_label = meta.get("label") or meta.get("zone") or "loop"
        label = self._safe_label(raw_label)

        fname = f"rec_{int(time.time())}_{label}_{uuid.uuid4().hex[:6]}.wav"
        out_path = os.path.join(self.dir, fname)
        self.active_path = out_path

        def _run():
            try:
                with wave.open(loop_path, "rb") as src, wave.open(out_path, "wb") as dst:
                    dst.setnchannels(src.getnchannels())
                    dst.setsampwidth(src.getsampwidth())
                    dst.setframerate(src.getframerate())
                    frames_total = int(max_seconds * src.getframerate())
                    chunk = 2048; written = 0
                    while written < frames_total and not self.stop_flag.is_set():
                        to_read = min(chunk, frames_total - written)
                        data = src.readframes(to_read)
                        if not data: break
                        dst.writeframes(data)
                        written += to_read
                print(f"[REC] saved {out_path}")
            except Exception as e:
                error = f"{e.__class__.__name__}: {e}"
                import traceback
                print(f"[FATAL][GEN] src={src} in {__file__}::AudioService._worker_loop")
                traceback.print_exc()  # <-- full file/line stack

        self.thread = threading.Thread(target=_run, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_flag.set()
        if self.thread: self.thread.join(timeout=2.0)
        self.thread = None
        return self.active_path
