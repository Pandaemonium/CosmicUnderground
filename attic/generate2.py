import os, secrets, torch, torchaudio
from einops import rearrange
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond

device = "cuda" if torch.cuda.is_available() else "cpu"
model, cfg = get_pretrained_model("stabilityai/stable-audio-open-small")
model = model.to(device)

BPM, BARS = 120, 4
seconds_total = BARS * (60.0 / BPM) * 4

# pick a seed guaranteed to be < 2**31
seed = secrets.randbits(31)         # or just a constant like 123456789

audio = generate_diffusion_cond(
    model,
    steps=8, cfg_scale=1.0, sampler_type="pingpong",
    conditioning=[{"prompt":"Alien FM bass groove, no drums, mono low end, 120 BPM, D minor, 4-bar seamless loop",
                   "seconds_total": seconds_total}],
    sample_size=cfg["sample_size"],
    device=device,
    seed=seed,                      # <- THIS avoids the NumPy bug path
)
os.makedirs("outputs", exist_ok=True)
torchaudio.save("outputs/loop.wav", (audio*32767).to(torch.int16).cpu(), cfg["sample_rate"])