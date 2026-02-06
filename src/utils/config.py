import yaml
from pathlib import Path

def load_config(path: str = "config/config.yaml") -> dict:
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path.resolve()}")
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)
