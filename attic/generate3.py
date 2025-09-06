import os, time, secrets, math
import numpy as np
import torch
from einops import rearrange
from huggingface_hub import login
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond
import soundfile as sf  # <- more reliable WAV writer on Windows
import re
from datetime import datetime

os.environ["OMP_NUM_THREADS"] = "2"
torch.set_num_threads(2)

# -------- setup --------
os.makedirs("outputs", exist_ok=True)
token = os.getenv("HUGGINGFACE_HUB_TOKEN")  # optional
if token: login(token=token)

torch.set_num_threads(max(1, os.cpu_count() // 2))  # be polite to CPU
device = "cuda"  # force CPU; avoids autocast/cuda warnings

print("[1/4] Loading model…")
t0 = time.time()
model, cfg = get_pretrained_model("stabilityai/stable-audio-open-small")
model = model.to(device)
sr, n = cfg["sample_rate"], cfg["sample_size"]
print(f"    loaded in {time.time()-t0:.2f}s, sr={sr}, frames={n}")

# -------- generation --------
BPM, BARS = 120, 4
seconds_total = BARS * (60.0 / BPM) * 4
#prompt = "Alien FM bass groove, no drums, mono low end, 120 BPM, D minor, 4-bar seamless loop"
prompt = "extra funky NES ToeJam and Earl synth solo, 120 BPM, D minor, 4-bar seamless loop"

seed = secrets.randbits(31)  # safe < 2**31
print(f"[2/4] Sampling… seed={seed}")

t1 = time.time()
with torch.inference_mode():
    audio = generate_diffusion_cond(
        model,
        steps=8, cfg_scale=1.0, sampler_type="pingpong",
        conditioning=[{"prompt": prompt, "seconds_total": seconds_total}],
        sample_size=n, device=device, seed=seed
    )
print(f"    sampling done in {time.time()-t1:.2f}s")

# -------- post: normalize & micro-fades --------
print("[3/4] Post-processing…")
t2 = time.time()
audio = rearrange(audio, "b d t -> d (b t)").to(torch.float32).cpu()
audio = audio / (audio.abs().max() + 1e-9)

# 3 ms micro-fades
fade = int(sr * 0.003)
ramp = torch.linspace(0, 1, fade).numpy()
audio[:, :fade] *= ramp
audio[:, -fade:] *= ramp[::-1]

# to numpy (channels last for soundfile)
wav = audio.T.numpy()  # shape (T, C)
print(f"    post done in {time.time()-t2:.2f}s")

# -------- save (soundfile) --------
print("[4/4] Generating filename and writing output…")
t3 = time.time()

# 1. Sanitize the prompt for the filename
# Make it lowercase, replace spaces/commas with a hyphen
sanitized_prompt = prompt.lower().replace(",", " ").replace(" ", "-") 
# Remove any characters that aren't letters, numbers, or hyphens
sanitized_prompt = re.sub(r'[^a-z0-9-]', '', sanitized_prompt)
# Truncate to a reasonable length
sanitized_prompt = sanitized_prompt[:60]

# 2. Get the current timestamp
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

# 3. Combine into a final filename and path
filename = f"{sanitized_prompt}_{timestamp}.wav"
output_path = os.path.join("outputs", filename)

# 4. Write the audio file
sf.write(output_path, wav, sr, subtype="PCM_16")
print(f"   Saved file: {output_path}")
print(f"   saved in {time.time()-t3:.2f}s")
print("DONE.")
