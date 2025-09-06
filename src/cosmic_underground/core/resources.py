import os, re

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

def runtime_dir(*parts: str) -> str:
    p = os.path.join(REPO_ROOT, "runtime", *parts)
    os.makedirs(p, exist_ok=True)
    return p

def assets_dir(*parts: str) -> str:
    return os.path.join(REPO_ROOT, "assets", *parts)

def data_dir(*parts: str) -> str:
    return os.path.join(REPO_ROOT, "data", *parts)

def slug(s: str, maxlen: int = 80) -> str:
    s = re.sub(r'[\\/:*?"<>|]', "_", str(s))
    s = re.sub(r"\s+", " ", s).strip().replace(" ", "_")
    return s[:maxlen]
