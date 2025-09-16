def quantize_beat(beat: float, div: int) -> float:
    # div: 1,2,4,8,16 etc
    step = 1.0 / div
    return round(beat / step) * step

def grid_label(div: int) -> str:
    if div == 1: return "1/1"
    return f"1/{div}"
