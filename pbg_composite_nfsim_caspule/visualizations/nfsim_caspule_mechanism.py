"""NFSimCaspuleMechanism — self-contained mechanism time-series visualization.

Plots scalar CASPULE observables (num_bonds, num_clusters, temperature, ...)
alongside NFSim species counts (stored as a nested dict under the 'species'
key in runs.db history rows) on shared time axes.

Supports dot-notation observable names: "species.Dimer" reads
state["species"]["Dimer"] from each history row.

Config keys:
  observables: list[str]   — observable names to plot; dot-notation for nested
                             dicts (e.g. "species.Dimer", "species.Active")
  title: str               — chart title (optional)
  sources: list[str]       — restrict to runs by sim_name (optional)

Private keys injected by the renderer:
  runs_db_path: str        — absolute path to runs.db
  study_yaml_path: str     — absolute path to study.yaml (for units metadata)
"""
from __future__ import annotations

import html as _html
import json
import sqlite3
from pathlib import Path

from process_bigraph.visualization import Visualization

_PALETTE = [
    "#6366f1", "#10b981", "#f43f5e", "#f59e0b",
    "#8b5cf6", "#06b6d4", "#84cc16", "#ec4899",
    "#14b8a6", "#dc2626", "#3b82f6", "#a855f7",
]


def _resolve_nested(state: dict, name: str):
    """Resolve a possibly dot-notation key from a state dict.
    "species.Dimer" -> state["species"]["Dimer"] if it's a dict, else None.
    """
    parts = name.split(".", 1)
    val = state.get(parts[0])
    if len(parts) == 1:
        return val
    if isinstance(val, dict):
        return val.get(parts[1])
    return None


def _load_runs(runs_db_path: str | None, sources: list[str]) -> list[dict]:
    if not runs_db_path:
        return []
    p = Path(runs_db_path)
    if not p.is_file():
        return []
    sources_set = set(sources or [])
    try:
        conn = sqlite3.connect(str(p))
    except sqlite3.Error:
        return []
    conn.row_factory = sqlite3.Row
    try:
        try:
            meta_rows = conn.execute(
                "SELECT run_id, sim_name, params_json FROM runs_meta"
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        has_history = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='history'"
        ).fetchone() is not None
        if not has_history:
            return []
        out: list[dict] = []
        for r in meta_rows:
            sim_name = r["sim_name"] or "default"
            if sources_set and sim_name not in sources_set:
                continue
            try:
                params = json.loads(r["params_json"] or "{}")
            except (json.JSONDecodeError, TypeError):
                params = {}
            history_rows = conn.execute(
                "SELECT step, global_time, state FROM history "
                "WHERE simulation_id=? ORDER BY step ASC",
                (r["run_id"],),
            ).fetchall()
            # We accumulate all named series (including dot-notation resolved later)
            raw_states: list[dict] = []
            time_axis: list[float] = []
            for hrow in history_rows:
                try:
                    s = json.loads(hrow["state"]) if hrow["state"] else {}
                except (json.JSONDecodeError, TypeError):
                    s = {}
                raw_states.append(s)
                time_axis.append(s.get("time") or hrow["global_time"] or 0.0)
            out.append({
                "run_id": r["run_id"],
                "sim_name": sim_name,
                "params": params,
                "raw_states": raw_states,
                "time": time_axis,
            })
        return out
    finally:
        conn.close()


def _build_traces(runs: list[dict], observable_names: list[str]) -> list[dict]:
    dash_patterns = ["solid", "dash", "dot", "dashdot", "longdash"]
    traces: list[dict] = []
    for obs_idx, name in enumerate(observable_names):
        color = _PALETTE[obs_idx % len(_PALETTE)]
        for trace_idx, run in enumerate(runs):
            ys = [_resolve_nested(s, name) for s in run["raw_states"]]
            # Drop None / non-numeric values
            pairs = [(t, y) for t, y in zip(run["time"], ys)
                     if y is not None and isinstance(y, (int, float))]
            if not pairs:
                continue
            xs, ys_clean = zip(*pairs)
            dash = dash_patterns[trace_idx % len(dash_patterns)] if len(runs) > 1 else "solid"
            label = name if len(runs) == 1 else f"{name} ({run['sim_name']})"
            traces.append({
                "x": list(xs), "y": list(ys_clean),
                "type": "scatter", "mode": "lines",
                "name": label,
                "line": {"color": color, "width": 2, "dash": dash},
            })
    return traces


def _render_html(cfg: dict) -> str:
    observable_names = list(cfg.get("observables") or [])
    if not observable_names:
        return (
            '<div style="padding:12px;color:#991b1b">'
            "NFSimCaspuleMechanism: no observables declared in config. "
            "Set <code>observables: [name1, name2]</code>."
            "</div>"
        )
    runs_db_path = cfg.get("runs_db_path") or cfg.get("_runs_db_path") or ""
    sources = list(cfg.get("sources") or [])
    runs = _load_runs(runs_db_path, sources)
    if not runs:
        return (
            '<div style="padding:12px;color:#92400e;background:#fef3c7;'
            "border:1px solid #fcd34d;border-radius:4px\">"
            "<strong>No run data yet.</strong> "
            "Run a baseline and re-render."
            "</div>"
        )
    traces = _build_traces(runs, observable_names)
    if not traces:
        return (
            '<div style="padding:12px;color:#991b1b">'
            f"None of the declared observables ({', '.join(observable_names)}) "
            "were found in any run's emitted state."
            "</div>"
        )
    title = _html.escape(cfg.get("title") or "")
    layout = {
        "title": {"text": title, "font": {"size": 14}},
        "xaxis": {"title": {"text": "time"}},
        "yaxis": {"title": {"text": "value"}},
        "margin": {"l": 60, "r": 15, "t": 40, "b": 50},
        "legend": {"orientation": "h", "y": -0.25},
        "template": "plotly_white",
    }
    fig = {"data": traces, "layout": layout}
    fig_json = json.dumps(fig)
    return (
        '<div id="nfsim-caspule-mech-fig"></div>'
        "<script>"
        "(function(){"
        "var d=document.getElementById('nfsim-caspule-mech-fig');"
        f"var fig={fig_json};"
        "if(window.Plotly){Plotly.newPlot(d,fig.data,fig.layout,{responsive:true});}"
        "else{var s=document.createElement('script');"
        "s.src='https://cdn.plot.ly/plotly-latest.min.js';"
        "s.onload=function(){Plotly.newPlot(d,fig.data,fig.layout,{responsive:true});};"
        "document.head.appendChild(s);}"
        "})();"
        "</script>"
    )


class NFSimCaspuleMechanism(Visualization):
    """Self-contained mechanism time-series for CASPULE + NFSim composites.

    Supports dot-notation observable paths (e.g. "species.Dimer") to access
    nested dict values in the emitted state.
    """

    config_schema = {
        **Visualization.config_schema,
        "observables": {"_type": "list[string]", "_default": []},
        "sources": {"_type": "list[string]", "_default": []},
        "runs_db_path": {"_type": "string", "_default": ""},
        "run_id": {"_type": "string", "_default": ""},
        "study_yaml_path": {"_type": "string", "_default": ""},
    }

    def inputs(self) -> dict:
        return {}

    def update(self, state: dict) -> dict:
        cfg = dict(getattr(self, "config", None) or {})
        return {"html": _render_html(cfg)}
