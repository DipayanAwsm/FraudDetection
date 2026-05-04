from pathlib import Path
import yaml
import json


def load_config(path="config/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def ensure_parent(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def save_json(obj, path):
    ensure_parent(path)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
