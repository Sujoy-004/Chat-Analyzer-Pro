"""Interactive ECharts option specs for the HTML report (charts_json).

Research Pattern 4 extension: every report chart also ships an ECharts
option dict (hover tooltips, dataZoom detail zoom, a 3D scrollable network).
pipeline.build_chart_specs is the ONLY builder; write_report consumes the
specs as JSON and falls back to the base64 PNG charts when a spec is missing.

Design rules honored here:
- build_chart_specs NEVER raises: every chart is built behind its own
  try/except, a failure is logged and the chart is omitted from the returned
  dict, and write_report then renders the PNG fallback (Pitfall 6 spirit).
- Specs are strictly JSON-serializable: numpy scalars are converted with
  int()/float()/str(), NaN is dropped/None-converted, and dates become ISO
  strings — the boundary validation in report_html drops anything else.
- All tooltip formatters are JSON template strings ({b}, {d}, ...) — never
  JS functions through the JSON pipe.
"""

from __future__ import annotations

import logging

import networkx as nx
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_DAY_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

_DATAZOOM = [
    {
        "type": "inside",
        "start": 0,
        "end": 100,
        "zoomOnMouseWheel": True,
        "moveOnMouseMove": True,
        "moveOnMouseWheel": False,
    },
    {
        "type": "slider",
        "start": 0,
        "end": 100,
        "height": 24,
        "bottom": 20,
    },
]

_LINE_GRID = {"left": 64, "right": 24, "top": 36, "bottom": 76}


def build_chart_specs(
    df: pd.DataFrame,
    df_sent: pd.DataFrame,
    health_trend_df: pd.DataFrame,
    network_res: dict,
) -> dict[str, dict]:
    """Build the interactive ECharts spec for every report chart.

    Each chart is built defensively: a failing builder is logged and skipped
    (the chart is omitted from the returned dict and write_report renders its
    base64 PNG fallback). Returns only JSON-serializable option dicts.
    Emotion is excluded: the gated NLP stage builds that spec separately via
    build_emotion_spec when the models are available.
    """
    builders = (
        ("timeline", lambda: _build_timeline(df)),
        ("activity", lambda: _build_activity(df)),
        ("participants", lambda: _build_participants(df)),
        ("sentiment", lambda: _build_sentiment(df_sent)),
        ("health", lambda: _build_health(health_trend_df)),
        ("network", lambda: _build_network(network_res)),
    )
    specs: dict[str, dict] = {}
    for name, build in builders:
        try:
            spec = build()
            if spec:
                specs[name] = spec
        except Exception:
            logger.warning("interactive chart %s failed", name, exc_info=True)
    return specs


def build_emotion_spec(emotion_summary: dict | None) -> dict | None:
    """Build the emotion pie spec; None when there is no distribution."""
    try:
        if not emotion_summary:
            return None
        distribution = emotion_summary.get("distribution") or emotion_summary.get(
            "emotion_distribution"
        )
        if not distribution:
            return None
        data = [
            {"name": str(label), "value": int(count)}
            for label, count in distribution.items()
        ]
        return {
            "tooltip": {"trigger": "item", "formatter": "{b}: {c} ({d}%)"},
            "legend": {"bottom": 0, "type": "scroll"},
            "series": [
                {
                    "type": "pie",
                    "radius": ["35%", "70%"],
                    "center": ["50%", "44%"],
                    "data": data,
                }
            ],
        }
    except Exception:
        logger.warning("interactive chart emotion failed", exc_info=True)
        return None


def _build_timeline(df: pd.DataFrame) -> dict | None:
    """Daily message counts as a line with slider+inside dataZoom."""
    if df is None or df.empty or "timestamp" not in df.columns:
        return None
    series = df.set_index("timestamp").resample("D").size()
    if series.empty:
        return None
    return {
        "tooltip": {"trigger": "axis"},
        "grid": _LINE_GRID,
        "dataZoom": _DATAZOOM,
        "xAxis": {
            "type": "category",
            "data": [d.strftime("%Y-%m-%d") for d in series.index],
            "boundaryGap": False,
            "axisLabel": {"rotate": 45, "hideOverlap": True},
        },
        "yAxis": {"type": "value", "name": "messages"},
        "series": [
            {
                "name": "Daily messages",
                "type": "line",
                "data": [int(v) for v in series.values],
                "showSymbol": False,
                "smooth": True,
                "lineStyle": {"width": 2},
                "areaStyle": {"opacity": 0.15},
                "emphasis": {"focus": "series"},
            }
        ],
    }


def _build_activity(df: pd.DataFrame) -> dict | None:
    """Hour x day-of-week heatmap with per-cell tooltip via the {b} name."""
    if df is None or df.empty or "timestamp" not in df.columns:
        return None
    ts = pd.to_datetime(df["timestamp"])
    valid = ts[ts.notna()]
    counts: dict[tuple[int, int], int] = {}
    for dow, hour in zip(valid.dt.dayofweek.tolist(), valid.dt.hour.tolist()):
        key = (int(dow), int(hour))
        counts[key] = counts.get(key, 0) + 1
    if not counts:
        return None
    cells = []
    for (dow, hour), count in counts.items():
        cells.append(
            {
                "name": f"{_DAY_LABELS[dow]} \u00b7 {hour:02d}:00 \u2014 {count} msgs",
                "value": [dow, hour, count],
            }
        )
    return {
        "tooltip": {"trigger": "item", "formatter": "{b}"},
        "grid": {"left": 48, "right": 16, "top": 24, "bottom": 64},
        "xAxis": {
            "type": "category",
            "data": _DAY_LABELS,
            "splitArea": {"show": True},
        },
        "yAxis": {
            "type": "category",
            "data": [f"{h:02d}:00" for h in range(24)],
            "splitArea": {"show": True},
        },
        "visualMap": {
            "min": 0,
            "max": max(counts.values()),
            "calculable": True,
            "orient": "horizontal",
            "left": "center",
            "bottom": 8,
            "inRange": {
                "color": ["#ebedf0", "#c6dbef", "#6baed6", "#2171b5", "#08306b"]
            },
        },
        "series": [
            {
                "name": "Activity",
                "type": "heatmap",
                "data": cells,
                "label": {"show": False},
            }
        ],
    }


def _build_participants(df: pd.DataFrame) -> dict | None:
    """Horizontal bar of per-participant message counts, descending."""
    if df is None or df.empty or "sender" not in df.columns:
        return None
    counts = df["sender"].value_counts()
    if counts.empty:
        return None
    items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    return {
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
        "grid": {"left": 130, "right": 32, "top": 16, "bottom": 32},
        "xAxis": {"type": "value", "name": "messages"},
        "yAxis": {
            "type": "category",
            "data": [str(name) for name, _ in items],
            "axisLabel": {"width": 110, "overflow": "truncate"},
        },
        "series": [
            {
                "name": "Messages",
                "type": "bar",
                "data": [int(count) for _, count in items],
                "barMaxWidth": 28,
                "itemStyle": {"color": "#4f8ef0", "borderRadius": [0, 4, 4, 0]},
            }
        ],
    }


def _build_sentiment(df_sent: pd.DataFrame) -> dict | None:
    """VADER compound line plus a dashed 7-day average."""
    if df_sent is None or df_sent.empty or "timestamp" not in df_sent.columns:
        return None
    if "vader_compound" not in df_sent.columns:
        return None
    daily = df_sent.set_index("timestamp")["vader_compound"].resample("D").mean()
    if daily.empty:
        return None
    moving_avg = daily.rolling(7, min_periods=1).mean()
    return {
        "tooltip": {"trigger": "axis"},
        "grid": _LINE_GRID,
        "dataZoom": _DATAZOOM,
        "xAxis": {
            "type": "category",
            "data": [d.strftime("%Y-%m-%d") for d in daily.index],
            "boundaryGap": False,
            "axisLabel": {"rotate": 45, "hideOverlap": True},
        },
        "yAxis": {
            "type": "value",
            "min": -1,
            "max": 1,
            "splitNumber": 4,
            "name": "compound",
        },
        "series": [
            {
                "name": "VADER compound",
                "type": "line",
                "data": [_float_or_none(v) for v in daily.values],
                "showSymbol": False,
                "smooth": True,
                "lineStyle": {"width": 2},
                "emphasis": {"focus": "series"},
            },
            {
                "name": "7-day avg",
                "type": "line",
                "data": [_float_or_none(v) for v in moving_avg.values],
                "showSymbol": False,
                "smooth": True,
                "lineStyle": {"type": "dashed", "width": 2},
                "emphasis": {"focus": "series"},
            },
        ],
    }


def _build_health(health_trend_df: pd.DataFrame) -> dict | None:
    """Relationship-health score line with a reference line at 50.

    Only built when the trend frame carries a health_score column — the raw
    df fallback (no column) renders the PNG figure instead (D-07 spirit).
    """
    if health_trend_df is None or health_trend_df.empty:
        return None
    if (
        "timestamp" not in health_trend_df.columns
        or "health_score" not in health_trend_df.columns
    ):
        return None
    score = health_trend_df["health_score"]
    valid = pd.notna(score)
    if not valid.any():
        return None
    ts = pd.to_datetime(health_trend_df["timestamp"])
    dates = [pd.Timestamp(t).strftime("%Y-%m-%d") for t in ts[valid]]
    values = [_float_or_none(v) for v in score[valid]]
    return {
        "tooltip": {"trigger": "axis"},
        "grid": _LINE_GRID,
        "dataZoom": _DATAZOOM,
        "xAxis": {
            "type": "category",
            "data": dates,
            "boundaryGap": False,
            "axisLabel": {"rotate": 45, "hideOverlap": True},
        },
        "yAxis": {
            "type": "value",
            "min": 0,
            "max": 100,
            "splitNumber": 4,
            "name": "health score",
        },
        "series": [
            {
                "name": "Health score",
                "type": "line",
                "data": values,
                "showSymbol": False,
                "smooth": True,
                "lineStyle": {"width": 2},
                "areaStyle": {"opacity": 0.15},
                "emphasis": {"focus": "series"},
                "markLine": {
                    "symbol": "none",
                    "silent": True,
                    "lineStyle": {"type": "dashed", "color": "#d62728"},
                    "data": [{"yAxis": 50}],
                },
            }
        ],
    }


def _build_network(network_res: dict) -> dict | None:
    """TRUE 3D network via scatter3D + lines3D on grid3D.

    Positions come from nx.spring_layout(dim=3) centered and normalized to
    ~[-50, 50] to match the grid3D box size. Omitted entirely on any failure
    or when the graph has fewer than 2 nodes (PNG fallback renders).
    """
    graph = network_res.get("graph") if isinstance(network_res, dict) else None
    if graph is None or graph.number_of_nodes() < 2:
        return None

    pos = nx.spring_layout(graph, dim=3, seed=42)
    nodes = list(graph.nodes())
    coords = np.array([[pos[node][0], pos[node][1], pos[node][2]] for node in nodes])
    coords = coords - coords.mean(axis=0)
    span = float(np.abs(coords).max())
    if span > 0:
        coords = coords / span * 50.0

    node_data = []
    for i, node in enumerate(nodes):
        node_data.append(
            {
                "name": str(node),
                "value": [
                    float(round(coords[i][0], 1)),
                    float(round(coords[i][1], 1)),
                    float(round(coords[i][2], 1)),
                    int(graph.degree(node)),
                ],
            }
        )

    weights = [float(graph[u][v].get("weight", 1)) for u, v in graph.edges()]
    wmax = max(weights) if weights else 1.0
    coord_by_node = {node: coords[i].tolist() for i, node in enumerate(nodes)}
    lines = []
    for (u, v), weight in zip(graph.edges(), weights):
        u_pos = coord_by_node[u]
        v_pos = coord_by_node[v]
        lines.append(
            {
                "value": float(round(weight, 1)),
                "coords": [
                    [float(round(c, 1)) for c in u_pos],
                    [float(round(c, 1)) for c in v_pos],
                ],
                "lineStyle": {"opacity": round(0.2 + 0.8 * weight / wmax, 2)},
            }
        )

    max_degree = max((int(graph.degree(node)) for node in nodes), default=1)
    return {
        "tooltip": {"trigger": "item"},
        "grid3D": {
            "boxWidth": 100,
            "boxHeight": 100,
            "boxDepth": 100,
            "axisLine": {"show": False},
            "axisLabel": {"show": False},
            "splitLine": {"show": False},
        },
        "visualMap": {
            "show": False,
            "dimension": 3,
            "seriesIndex": [0],
            "min": 1,
            "max": max(1, max_degree),
            "inRange": {
                "symbolSize": [8, 28],
                "color": ["#74c476", "#31a354", "#006837"],
            },
        },
        "viewControl": {
            "projection": "perspective",
            "alpha": 25,
            "beta": 18,
            "autoRotate": True,
            "autoRotateAfterStill": 3,
            "rotateSensitivity": 1,
            "zoomSensitivity": 1,
            "panSensitivity": 0,
            "distance": 240,
            "minDistance": 80,
            "maxDistance": 600,
        },
        "xAxis3D": {"type": "value"},
        "yAxis3D": {"type": "value"},
        "zAxis3D": {"type": "value"},
        "series": [
            {
                "name": "participants",
                "type": "scatter3D",
                "coordinateSystem": "grid3D",
                "symbol": "circle",
                "itemStyle": {"opacity": 0.92, "borderWidth": 1, "borderColor": "#fff"},
                "label": {"show": False},
                "emphasis": {
                    "itemStyle": {"opacity": 1},
                    "label": {"show": True},
                },
                "data": node_data,
            },
            {
                "name": "ties",
                "type": "lines3D",
                "coordinateSystem": "grid3D",
                "polyline": False,
                "lineStyle": {"width": 2, "opacity": 0.5},
                "data": lines,
            },
        ],
    }


def _float_or_none(value) -> float | None:
    """Coerce a scalar to float, mapping NaN/None to None for JSON safety."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(number):  # NaN is never a valid JSON number
        return None
    return number