from pathlib import Path
import json
import joblib
def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
def save_joblib(obj, path: Path) -> str:
    ensure_dir(path.parent)
    joblib.dump(obj, path)
    return str(path)
def load_joblib(path: Path):
    return joblib.load(path)
def save_json(data, path: Path) -> str:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return str(path)
