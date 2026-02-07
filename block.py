from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple, List


@dataclass
class ParamInfo:
    name: str
    vtype: type  # float, int, bool
    default: Any
    min: float = 0.0
    max: float = 1.0


def block_param(name: str, vtype: type, default: Any, min_v: float = 0.0, max_v: float = 1.0):
    """Decorator to register metadata for GUI generation."""

    def wrapper(cls):
        if not hasattr(cls, "_gui_params"):
            cls._gui_params = []
        cls._gui_params.append(ParamInfo(name, vtype, default, min_v, max_v))
        return cls

    return wrapper


@dataclass
class BaseBlock:
    _gui_params: List[ParamInfo] = field(default_factory=list, init=False)

    def execute(self, payload: Any, *, params: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        raise NotImplementedError