import yaml
from pathlib import Path
from types import SimpleNamespace
from typing import Union


def _dict_to_namespace(d):
    """Recursively convert dicts to SimpleNamespace for dot access."""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, (list, tuple)):
        return [_dict_to_namespace(i) for i in d]
    else:
        return d


def load_config(path: Union[str, Path]):
    """Load YAML configuration file and return a SimpleNamespace for dot access."""
    path = Path(path)
    with path.open("r") as f:
        cfg_dict = yaml.safe_load(f)
    return _dict_to_namespace(cfg_dict) 