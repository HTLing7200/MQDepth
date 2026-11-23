from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class MQDepthConfig:
    queries: int = 256
    connector_layers: int = 4
    connector_heads: int = 8
    geometry_tokens: int = 1200
    geometry_pool_tokens: int = 256
    prompt: str = "Estimate metric scene depth from this image."
    width: int = 768
    qwen_long_side: int = 768
    focal_reference: float = 1000.0
    depth_min: float = 0.1
    depth_max: float = 512.0
    size_multiple: int = 32
    geometry_max_aspect_ratio: float = 2.0
    moge2_cls_token: bool = False

    @classmethod
    def from_directory(cls, directory: str | Path) -> "MQDepthConfig":
        path = Path(directory) / "config.json"
        return cls(**json.loads(path.read_text()))

    def to_dict(self) -> dict:
        return asdict(self)
