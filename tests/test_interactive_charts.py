"""Interactive ECharts report tests (charts_json specs + PNG fallback).

Exercises REAL modules: build_chart_specs against a synthetic canonical chat
DataFrame fed through analyze_network, the network's 3D spec shape, the
emotion pie spec, and write_report's two rendering paths — interactive div +
CHART_SPECS when specs exist, and the base64 PNG <img> fallback when they
don't (charts_json absent, a raising builder, or a non-serializable spec).
"""

import json

import numpy as np
import pandas as pd
import pytest

from chat_analyzer.analysis.network_graph import analyze_network
from chat_analyzer.cli.chart_json import build_chart_specs, build_emotion_spec
from chat_analyzer.cli.report_html import write_report
from tests.test_phase2_report import _results


@pytest.fixture
def chat_df() -> pd.DataFrame:
    """A tiny synthetic canonical chat DataFrame (pipeline analysis shape).

    timestamp/datetime drive the resampling charts, sender/message feed the
    participant + network charts, and vader_compound drives the sentiment
    line — the same columns run_pipeline's analysis stage produces.
    """
    n = 48
    ts = pd.to_datetime(
        [pd.Timestamp("2024-01-01") + pd.Timedelta(hours=i * 3) for i in range(n)]
    )
    senders = ["Alice", "Bob", "Claire"]
    return pd.DataFrame(
        {
            "timestamp": ts,
            "datetime": ts,
            "sender": [senders[i % 3] for i in range(n)],
            "message": [f"message {i}" for i in range(n)],
            "message_length": [5 + (i % 20) for i in range(n)],
            "vader_compound": np.linspace(-0.8, 0.9, n),
        }
    )


def _health_trend_df() -> pd.DataFrame:
    """The rolling-health Trend DataFrame shape (date renamed to timestamp)."""
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "health_score": [80.0, 72.5, 90.0],
            "grade": ["B", "B", "A"],
        }
    )


def test_build_chart_specs_keys_serializable(chat_df):
    """Every always-on chart gets a spec, and every spec is JSON-serializable."""
    health = _health_trend_df()
    network_res = analyze_network(chat_df)
    specs = build_chart_specs(chat_df, chat_df, health, network_res)

    assert set(specs) == {
        "timeline",
        "activity",
        "participants",
        "sentiment",
        "health",
        "network",
    }
    for name, spec in specs.items():
        assert isinstance(spec, dict), name
        json.dumps(spec, allow_nan=False)  # must never raise

    timeline = specs["timeline"]
    assert timeline["xAxis"]["type"] == "category"
    assert timeline["series"][0]["type"] == "line"
    assert [z["type"] for z in timeline["dataZoom"]] == ["inside", "slider"]

    activity = specs["activity"]
    assert activity["series"][0]["type"] == "heatmap"
    assert activity["xAxis"]["data"] == ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    assert len(activity["yAxis"]["data"]) == 24
    assert all("msgs" in cell["name"] for cell in activity["series"][0]["data"])

    participants = specs["participants"]
    descending = participants["series"][0]["data"]
    assert descending == sorted(descending, reverse=True)

    sentiment = specs["sentiment"]
    assert sentiment["yAxis"]["min"] == -1 and sentiment["yAxis"]["max"] == 1
    assert [s["name"] for s in sentiment["series"]] == ["VADER compound", "7-day avg"]

    health_spec = specs["health"]
    assert health_spec["yAxis"]["name"] == "health score"
    assert health_spec["series"][0]["markLine"]["data"] == [{"yAxis": 50}]


def test_network_spec_is_3d(chat_df):
    """Network is TRUE 3D: scatter3D + lines3D on grid3D, not 2D graphGL."""
    network_res = analyze_network(chat_df)
    specs = build_chart_specs(chat_df, chat_df, _health_trend_df(), network_res)
    net = specs["network"]

    assert "grid3D" in net
    assert net["viewControl"]["autoRotate"] is True
    types = [s["type"] for s in net["series"]]
    assert "scatter3D" in types
    assert "lines3D" in types

    nodes = net["series"][0]["data"]
    assert all(len(node["value"]) == 4 for node in nodes)  # x, y, z, degree
    for node in nodes:
        assert all(-50.5 <= coord <= 50.5 for coord in node["value"][:3])

    ties = net["series"][1]["data"]
    assert ties
    assert all(len(t["coords"]) == 2 for t in ties)
    assert all(0 < t["lineStyle"]["opacity"] <= 1 for t in ties)


def test_emotion_spec_pie():
    """The gated emotion spec is a donut pie with percentage formatter."""
    spec = build_emotion_spec({"distribution": {"joy": 10, "anger": 3}, "dominant": "joy"})
    assert spec["series"][0]["type"] == "pie"
    assert spec["tooltip"]["formatter"] == "{b}: {c} ({d}%)"
    assert [d["name"] for d in spec["series"][0]["data"]] == ["joy", "anger"]
    # The raw module summary spells the key emotion_distribution (Pattern 2).
    spec2 = build_emotion_spec({"emotion_distribution": {"joy": 1}})
    assert [d["name"] for d in spec2["series"][0]["data"]] == ["joy"]
    assert build_emotion_spec(None) is None


def test_interactive_report_renders_specs(chat_df, tmp_path, monkeypatch):
    """Specs present -> chart divs + CHART_SPECS, not the PNG <img> branches."""
    network_res = analyze_network(chat_df)
    specs = build_chart_specs(chat_df, chat_df, _health_trend_df(), network_res)

    res = _results()
    res["charts_json"] = specs
    src = tmp_path / "a.txt"
    src.write_text("x\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    out = write_report(res, src).read_text(encoding="utf-8")
    for chart_id in ("timeline", "activity", "participants", "sentiment", "health", "network"):
        assert f'id="chart-{chart_id}"' in out, chart_id
    assert "CHART_SPECS" in out
    assert out.count("echarts.init") >= 1
    assert "scatter3D" in out
    assert "grid3D" in out
    assert '<img class="chart" alt="Message timeline"' not in out


def test_png_fallback_when_spec_builder_raises(monkeypatch, tmp_path):
    """A build_chart_specs crash (caught upstream) leaves the PNG fallback.

    The pipeline wraps the builder, so a failing spec never reaches the
    report — written here is the honest integration path: the raise happens
    before adapt(), charts_json stays empty, and write_report renders the
    base64 PNG <img> branches with the interactive runtime inert.
    """
    from chat_analyzer.cli import chart_json as chart_json_mod

    def boom(*args, **kwargs):
        raise RuntimeError("spec build exploded")

    monkeypatch.setattr(chart_json_mod, "build_chart_specs", boom)
    with pytest.raises(RuntimeError):
        chart_json_mod.build_chart_specs(None, None, None, None)

    res = _results()  # no charts_json key -> empty, exactly like the crash path
    src = tmp_path / "a.txt"
    src.write_text("x\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    out = write_report(res, src).read_text(encoding="utf-8")
    assert "data:image/png;base64," in out
    assert '<img class="chart" alt="Message timeline"' in out
    assert 'id="chart-timeline"' not in out
    assert "echarts.init" in out  # runtime stays inlined but finds no specs


def test_non_serializable_spec_hits_boundary(tmp_path, monkeypatch):
    """A spec that cannot be JSON-serialized is dropped at the boundary."""
    res = _results()
    res["charts_json"] = {"timeline": object()}
    src = tmp_path / "a.txt"
    src.write_text("x\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    out = write_report(res, src).read_text(encoding="utf-8")
    assert '<img class="chart" alt="Message timeline"' in out
    assert 'id="chart-timeline"' not in out


def test_script_tag_in_participant_name_escaped(tmp_path, monkeypatch):
    """A </script> payload inside a chart spec must never break the inline JS.

    The participant name rides through build_chart_specs -> the participants
    bar spec -> Jinja |tojson. tojson escapes < as \u003c, so the raw
    `</script>` sequence must not appear anywhere in the report — the literal
    inlined bundles are scrubbed by _inline_js, leaving zero occurrences.
    """
    evil = "<script>alert(1)</script>Alice"
    n = 6
    ts = pd.to_datetime(
        [pd.Timestamp("2024-01-01") + pd.Timedelta(hours=i) for i in range(n)]
    )
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "datetime": ts,
            "sender": [evil, "Bob"] * 3,
            "message": ["hi"] * n,
            "message_length": [2] * n,
            "vader_compound": [0.1] * n,
        }
    )
    specs = build_chart_specs(df, df, _health_trend_df(), analyze_network(df))
    assert "participants" in specs
    names = specs["participants"]["yAxis"]["data"]
    assert evil in names

    res = _results()
    res["charts_json"] = specs
    src = tmp_path / "a.txt"
    src.write_text("x\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    out = write_report(res, src).read_text(encoding="utf-8")
    # The raw payload (unescaped <) must never survive — report legitimately
    # contains its own </script> closing tags, so the breakout test is the
    # VERBATIM payload, not the bare tag.
    assert "<script>alert(1)</script>Alice" not in out
    assert "\\u003cscript" in out  # tojson-escaped payload IS serialized