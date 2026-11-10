"""配置加载：极简 YAML 子集（嵌套 map / 标量 / 内联列表）。

沿用 N5 的教训：先剥离注释再判断空值，否则带行尾注释的嵌套节会被误判为标量。
"""
from __future__ import annotations

from pathlib import Path


def _strip_comment(line: str) -> str:
    return line.split("#", 1)[0].strip()


def _parse_scalar(s: str):
    if s.startswith("[") and s.endswith("]"):
        inner = s[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(p.strip()) for p in inner.split(",")]
    if s in ("true", "True"):
        return True
    if s in ("false", "False"):
        return False
    if s in ("null", "None", "~"):
        return None
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s.strip('"').strip("'")


def load_config(path: Path) -> dict:
    root: dict = {}
    stack: list[tuple[int, dict]] = [(-1, root)]
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = _strip_comment(raw)
        if not line:
            continue
        indent = len(raw) - len(raw.lstrip())
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        key = key.strip()
        val = val.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if val == "":
            child: dict = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_scalar(val)
    return root


def case_from_config(cfg: dict):
    """由 configs/config.yaml 组装 ThermalCase。"""
    from .materials import Layer, ThermalCase

    g = cfg["geometry"]
    b = cfg["boundary"]
    s = cfg["source"]
    n = cfg["numerics"]
    layers = tuple(
        Layer(str(L["name"]), float(L["thickness"]), float(L["k"]), float(L["rc"]))
        for L in cfg["layers"]
    ) if isinstance(cfg.get("layers"), list) else tuple(
        Layer(str(name), float(vals["thickness"]), float(vals["k"]), float(vals["rc"]))
        for name, vals in cfg["layers"].items()
    )
    return ThermalCase(
        width=float(g["width"]),
        layers=layers,
        h_conv=float(b["h_conv"]),
        t_amb=float(b["t_amb"]),
        power_per_length=float(s["power_per_length"]),
        heat_source_width=float(s.get("heat_source_width", g["width"])),
        nx=int(n["nx"]),
        ny_per_layer=int(n["ny_per_layer"]),
        theta=float(n.get("theta", 1.0)),
        dt=float(n.get("dt", 0.01)),
        t_end=float(n.get("t_end", 10.0)),
        pulse_period=float(s.get("pulse_period", 2.0)),
        duty=float(s.get("duty", 0.5)),
    )


def default_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "config.yaml"
