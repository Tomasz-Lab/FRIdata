from pathlib import Path
from typing import Union, Literal
from pydantic import BaseModel, field_validator
import json
import os

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

CONFIG_ENV_VAR = "FRIDATA_CONFIG"


def _resolve_config_path(config_path: Path = None) -> tuple[Path, str]:
    """Resolve the config file location and report where the choice came from.

    Precedence: explicit argument, then the FRIDATA_CONFIG environment
    variable, then config.json in the current working directory. The package
    directory is deliberately not consulted — it is read-only once installed.
    """
    if config_path is not None:
        return Path(config_path), "the config_path argument"

    from_env = os.environ.get(CONFIG_ENV_VAR)
    if from_env:
        return Path(from_env), f"the {CONFIG_ENV_VAR} environment variable"

    return Path.cwd() / "config.json", "the current working directory"


def load_config(config_path: Path = None) -> Config:
    path, source = _resolve_config_path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path} (from {source})")
    with open(path) as f:
        data = json.load(f)
    return Config(**data) 