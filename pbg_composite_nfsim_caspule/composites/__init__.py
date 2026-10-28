"""Composite specs for the CASPULE + NFSim workspace.

Declarative ``*.composite.yaml`` files in this directory define the two
demonstration composites (dimer dimerization and polymer fractionation).
The dashboard's Composites tab discovers them automatically once the package
is installed in a workspace.

``build_composite(name)`` loads a spec by short name and instantiates
``process_bigraph.Composite`` with parameter substitution.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

_COMPOSITES_DIR = Path(__file__).parent
_FULL_PH = re.compile(r"^\$\{([a-zA-Z_][a-zA-Z0-9_]*)\}$")
_INLINE_PH = re.compile(r"\$\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


def _cast(value: Any, declared_type: str | None) -> Any:
    if declared_type is None:
        return value
    if declared_type == "float":
        return float(value)
    if declared_type in ("int", "integer"):
        return int(value)
    if declared_type in ("string", "str"):
        return str(value)
    return value


def _substitute(obj: Any, params: dict, overrides: dict) -> Any:
    if isinstance(obj, dict):
        return {k: _substitute(v, params, overrides) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_substitute(v, params, overrides) for v in obj]
    if isinstance(obj, str):
        m = _FULL_PH.match(obj)
        if m:
            name = m.group(1)
            pdef = params.get(name, {})
            raw = overrides.get(name, pdef.get("default"))
            return _cast(raw, pdef.get("type"))
        if _INLINE_PH.search(obj):
            return _INLINE_PH.sub(
                lambda mm: str(
                    overrides.get(mm.group(1),
                                  params.get(mm.group(1), {}).get("default", ""))
                ),
                obj,
            )
    return obj


def list_composite_specs() -> list[str]:
    return sorted(
        p.name[: -len(".composite.yaml")]
        for p in _COMPOSITES_DIR.glob("*.composite.yaml")
    )


def load_composite_spec(name: str) -> dict:
    path = _COMPOSITES_DIR / f"{name}.composite.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"composite spec not found: {path}")
    return yaml.safe_load(path.read_text())


def build_composite(name: str, *, overrides: dict | None = None, core=None):
    """Load a composite spec by name and return a process_bigraph.Composite."""
    from process_bigraph import Composite
    from pbg_composite_nfsim_caspule.core import build_core

    spec = load_composite_spec(name)
    if core is None:
        core = build_core()
    params = spec.get("parameters") or {}
    state = _substitute(spec.get("state") or {}, params, overrides or {})
    return Composite({"state": state}, core=core)
