from dataclasses import dataclass
from pathlib import Path
import json


CONFIG_PATH = Path("rag_interval_cluster/rag_runtime_config.json")


@dataclass
class RagRuntimeConfig:
    enabled: bool = False
    model: str = "tinyswallow-vtuber:latest"
    base_url: str = "http://localhost:11434/v1"
    temperature: float = 0.2
    max_tokens: int = 300


def create_default_config() -> None:
    if CONFIG_PATH.exists():
        return

    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "enabled": False,
        "model": "tinyswallow-vtuber:latest",
        "base_url": "http://localhost:11434/v1",
        "temperature": 0.2,
        "max_tokens": 300,
    }

    CONFIG_PATH.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_runtime_config() -> RagRuntimeConfig:
    create_default_config()

    data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    return RagRuntimeConfig(
        enabled=bool(data.get("enabled", False)),
        model=str(data.get("model", "tinyswallow-vtuber:latest")),
        base_url=str(data.get("base_url", "http://localhost:11434/v1")),
        temperature=float(data.get("temperature", 0.2)),
        max_tokens=int(data.get("max_tokens", 300)),
    )