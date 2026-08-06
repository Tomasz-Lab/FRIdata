from pathlib import Path
from typing import Union, Literal
from pydantic import BaseModel, field_validator
import json

CarbonAtomType = Literal["CA", "CB"]

class Config(BaseModel):
    debug_mode: Literal["debug", "warning", "error"] = "warning"
    data_path: str = "path/to/data/"
    disto_type: CarbonAtomType = "CA"
    disto_thr: Union[int, str] = "inf"
    separator: str = "-"
    batch_size: Union[int, str] = 1000

    @field_validator("batch_size")
    def batch_size_valid(cls, v):
        if v == "infer":
            return v
        if isinstance(v, int) and v >= 1:
            return v
        raise ValueError("batch_size must be >= 1 or 'infer'")

    @field_validator("disto_thr")
    def disto_thr_valid(cls, v):
        if v == "inf":
            return v
        if isinstance(v, int):
            return v
        raise ValueError("disto_thr must be an integer or 'inf'")

def load_config(config_path: Path = None) -> Config:
    default_path = Path(__file__).parent.parent / "config.json"
    path = config_path or default_path
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        data = json.load(f)
    return Config(**data) 