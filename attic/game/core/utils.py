from typing import Optional
import math

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def fmt_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    m = int(seconds // 60)
    s = seconds - m * 60
    if m > 0:
        return f"{m}:{s:05.2f}"
    return f"{s:.2f}"

def parse_time(s: str) -> Optional[float]:
    s = (s or "").strip()
    try:
        if ":" in s:
            mm, ss = s.split(":")
            return max(0.0, int(mm) * 60 + float(ss))
        return max(0.0, float(s))
    except Exception:
        return None

def seconds_to_samples(t: float, sr: int) -> int:
    return int(max(0.0, t) * sr)

def samples_to_seconds(n: int, sr: int) -> float:
    return float(n) / float(sr)

def nice_grid_step(px_per_sec: float) -> float:
    # pick a readable major grid based on zoom (similar to the old version)
    if px_per_sec < 40: return 4.0
    if px_per_sec < 80: return 2.0
    if px_per_sec > 200: return 0.5
    return 1.0
