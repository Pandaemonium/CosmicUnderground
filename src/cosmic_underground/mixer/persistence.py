from __future__ import annotations
import json
from .state import Project

def save_project(path: str, proj: Project):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(proj, default=lambda o: o.__dict__, indent=2)

# For MVP we can write a lightweight loader later
