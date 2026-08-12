"""Single-file HTML report card for the chat-analyzer CLI (D-08..D-14).

research Pattern 4 — one self-contained HTML file: jinja2 autoescape
template (inline constant, no external assets, no CDN), base64 PNG chart
URIs, sanitized filename in the current working directory (D-09), utf-8
write, best-effort auto-open. Chat content is UNTRUSTED input: autoescape
is set explicitly (plain jinja2 defaults to False) and chart URIs are
validated at the boundary before they reach the template.
"""

from __future__ import annotations

import importlib.resources
import json
import logging
import os
import re
import webbrowser
from pathlib import Path

from jinja2 import Environment, select_autoescape

from chat_analyzer.cli.contracts import AnalysisResults

logger = logging.getLogger(__name__)

_INVALID_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f\x7f]')

_CHART_PREFIX = "data:image/png;base64,"

# The seven report chart slots, in template order. A spec key missing from
# charts_json renders the chart's base64 PNG fallback instead.
_CHART_KEYS = (
    "timeline",
    "activity",
    "participants",
    "sentiment",
    "health",
    "network",
    "emotion",
)


def _inline_js(js: str) -> str:
    """Neutralize strings that would terminate or comment out an inline <script>.

    `</script` must never appear inside an inlined bundle (it would close the
    HTML script element early), and `<!--` could start an HTML comment mode in
    ancient parsers. Scrubbing is applied to ALL inlined JS: the two vendored
    ECharts bundles and the report runtime.
    """
    return js.replace("</script", "<\\/script").replace("<!--", "<\\!--")


def _load_inline_asset(name: str) -> str:
    """Read one vendored ECharts bundle from the package assets, scrubbed.

    Assets ship inside package chat_analyzer (parent of the cli package) via
    the wheel's `artifacts` glob; importlib.resources locates them for both
    editable and regular installs. A missing bundle degrades to "" — the
    report still renders (interactive charts no-op, PNGs still show) rather
    than crashing write_report.
    """
    try:
        bundle = (
            importlib.resources.files("chat_analyzer")
            .joinpath("assets", name)
            .read_text(encoding="utf-8")
        )
    except (FileNotFoundError, ModuleNotFoundError, OSError):
        logger.warning("ECharts asset %s not found; interactive charts disabled", name)
        return ""
    return _inline_js(bundle)


_RUNTIME_JS = _inline_js(
    """\
function webglAvailable() {
  if (!window.WebGLRenderingContext) { return false; }
  var canvas = document.createElement('canvas');
  try {
    var gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
    return !!gl;
  } catch (e) { return false; }
}
function fallbackNetwork() {
  var el = document.getElementById('chart-network');
  var img = document.getElementById('img-network');
  if (el) { el.style.display = 'none'; }
  if (img) { img.style.display = ''; }
}
var _chartKeys = ['timeline', 'activity', 'participants', 'sentiment', 'health', 'network', 'emotion'];
var _charts = {};
function initCharts() {
  if (!webglAvailable()) { fallbackNetwork(); }
  for (var i = 0; i < _chartKeys.length; i++) {
    var key = _chartKeys[i];
    var el = document.getElementById('chart-' + key);
    if (!el) { continue; }
    var option = CHART_SPECS[key];
    if (!option || !option.series || !option.series.length) { continue; }
    if (key === 'network' && !webglAvailable()) { continue; }
    try {
      _charts[key] = echarts.init(el);
      _charts[key].setOption(option);
    } catch (e) { _charts[key] = null; }
  }
}
function resizeVisibleCharts() {
  for (var key in _charts) {
    if (!_charts[key]) { continue; }
    var el = document.getElementById('chart-' + key);
    if (el && el.offsetParent !== null) { _charts[key].resize(); }
  }
}
function showTab(id) {
  var panels = document.querySelectorAll('.panel');
  for (var i = 0; i < panels.length; i++) {
    panels[i].classList.toggle('active', panels[i].id === 'tab-' + id);
  }
  var buttons = document.querySelectorAll('button.tab');
  for (var j = 0; j < buttons.length; j++) {
    buttons[j].classList.toggle('active', buttons[j].dataset.tab === id);
  }
  if (window.requestAnimationFrame) {
    requestAnimationFrame(resizeVisibleCharts);
  } else {
    resizeVisibleCharts();
  }
}
window.addEventListener('resize', resizeVisibleCharts);
document.addEventListener('DOMContentLoaded', initCharts);
"""
)

TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{{ title }}</title>
<style>
  body { font-family: -apple-system, "Segoe UI", Roboto, sans-serif; margin: 0; background: #f6f7f9; color: #222; }
  header { padding: 24px 32px 8px; }
  h1 { margin: 0; font-size: 28px; }
  .subtitle { color: #666; margin: 6px 0 0; }
  nav { padding: 8px 32px; }
  button.tab { border: 0; background: #e8eaf0; padding: 8px 14px; margin-right: 6px; border-radius: 6px; cursor: pointer; font-size: 14px; }
  button.tab.active { background: #667eea; color: #fff; }
  main { padding: 0 32px 40px; }
  .panel { display: none; }
  .panel.active { display: block; }
  .card { background: #fff; border-radius: 10px; padding: 20px; margin-top: 16px; box-shadow: 0 1px 3px rgba(0,0,0,.08); }
  .lead { font-size: 17px; color: #333; }
  img.chart { max-width: 100%; height: auto; margin-top: 12px; border: 1px solid #eee; border-radius: 8px; }
  .chart { width: 100%; height: 400px; margin-top: 12px; }
  table { border-collapse: collapse; width: 100%; margin-top: 12px; }
  th, td { border: 1px solid #e4e7ec; padding: 8px 10px; text-align: left; font-size: 14px; }
  th { background: #f2f4f8; }
  ul { margin-top: 8px; }
  .skip-note { color: #b45309; background: #fef3c7; padding: 8px 12px; border-radius: 6px; margin: 12px 32px 0; }
  .narrative-summary { font-size: 15px; background: #eef2ff; border-left: 4px solid #667eea; padding: 10px 14px; border-radius: 6px; margin-top: 6px; }
  .obs-meta { color: #667; font-size: 12px; }
</style>
</head>
<body>
<header>
  <h1>{{ title }}</h1>
  <p class="subtitle">{{ subtitle }}</p>
</header>
{% if parse.skipped_lines > 0 %}
<p class="skip-note">Skipped {{ parse.skipped_lines }} lines that couldn't be parsed.</p>
{% endif %}
<nav>
  <button class="tab active" data-tab="overview" onclick="showTab('overview')">Overview</button>
  <button class="tab" data-tab="participants" onclick="showTab('participants')">Participants</button>
  <button class="tab" data-tab="flow" onclick="showTab('flow')">Flow</button>
  <button class="tab" data-tab="words" onclick="showTab('words')">Words</button>
  <button class="tab" data-tab="sentiment" onclick="showTab('sentiment')">Sentiment</button>
  <button class="tab" data-tab="health" onclick="showTab('health')">Relationship Health</button>
  <button class="tab" data-tab="network" onclick="showTab('network')">Network</button>
  <button class="tab" data-tab="emotion" onclick="showTab('emotion')">Emotion</button>
  <button class="tab" data-tab="narrative" onclick="showTab('narrative')">What's going on</button>
</nav>
<main>
  <div class="panel active" id="tab-overview">
    <div class="card">
      <p class="lead">{{ insights[0] }}</p>
      {% if charts_json.timeline %}<div id="chart-timeline" class="chart"></div>{% else %}{% if charts.timeline %}<img class="chart" alt="Message timeline" src="{{ charts.timeline }}">{% endif %}{% endif %}
      <table>
        <tr><th>Total messages</th><td>{{ stats.total_messages }}</td></tr>
        <tr><th>Participants</th><td>{{ stats.participants }}</td></tr>
        <tr><th>Date range</th><td>{{ stats.date_range.start }} to {{ stats.date_range.end }}</td></tr>
        <tr><th>Duration</th><td>{{ stats.duration_days }} days</td></tr>
        <tr><th>Media messages</th><td>{{ stats.media_messages }}</td></tr>
      </table>
    </div>
  </div>
  <div class="panel" id="tab-participants">
    <div class="card">
      <p class="lead">{{ insights[1] }}</p>
      {% if charts_json.participants %}<div id="chart-participants" class="chart"></div>{% else %}{% if charts.participants %}<img class="chart" alt="Participant activity" src="{{ charts.participants }}">{% endif %}{% endif %}
      <table>
        <tr><th>Participant</th><th>Messages</th><th>Avg message length</th><th>Share</th></tr>
        {% for name, data in participants.items() %}
        <tr><td>{{ name }}</td><td>{{ data.messages }}</td><td>{{ data.avg_message_length }}</td><td>{{ data.share_pct }}%</td></tr>
        {% endfor %}
      </table>
    </div>
  </div>
  <div class="panel" id="tab-flow">
    <div class="card">
      <p class="lead">{{ insights[2] }}</p>
      {% if charts_json.activity %}<div id="chart-activity" class="chart"></div>{% else %}{% if charts.activity %}<img class="chart" alt="Activity heatmap" src="{{ charts.activity }}">{% endif %}{% endif %}
      <table>
        <tr><th>Busiest day</th><td>{{ stats.busiest_day }}</td></tr>
        <tr><th>Peak hour</th><td>{{ stats.peak_hour }}:00</td></tr>
        {% if stats.avg_response_time %}
        <tr><th>Avg response time</th><td>{{ stats.avg_response_time }} minutes</td></tr>
        {% endif %}
      </table>
    </div>
  </div>
  <div class="panel" id="tab-words">
    <div class="card">
      <p class="lead">{{ insights[3] }}</p>
      <h3>Top words</h3>
      <table>
        {% for w in content.top_words %}<tr><td>{{ w }}</td></tr>{% endfor %}
      </table>
      <h3>Top emojis</h3>
      <ul>
        {% for e in content.top_emojis %}<li>{{ e }}</li>{% endfor %}
      </ul>
    </div>
  </div>
  <div class="panel" id="tab-sentiment">
    <div class="card">
      <p class="lead">{{ insights[4] }}</p>
      {% if charts_json.sentiment %}<div id="chart-sentiment" class="chart"></div>{% else %}{% if charts.sentiment %}<img class="chart" alt="Sentiment over time" src="{{ charts.sentiment }}">{% endif %}{% endif %}
      <table>
        <tr><th>Sentiment</th><th>Messages</th></tr>
        {% for label, count in sentiment.distribution.items() %}
        <tr><td>{{ label }}</td><td>{{ count }}</td></tr>
        {% endfor %}
      </table>
    </div>
  </div>
  <div class="panel" id="tab-health">
    <div class="card">
      <p class="lead">{{ insights[5] }}</p>
      {% if charts_json.health %}<div id="chart-health" class="chart"></div>{% else %}{% if charts.health %}<img class="chart" alt="Relationship health trend" src="{{ charts.health }}">{% endif %}{% endif %}
      {% if health %}
      <table>
        <tr><th>Overall health score</th><td>{{ health.overall_score }}</td></tr>
        <tr><th>Grade</th><td>{{ health.grade }}</td></tr>
        <tr><th>Initiator balance</th><td>{{ health.initiator_balance }}</td></tr>
        <tr><th>Avg response minutes</th><td>{{ health.avg_response_minutes }}</td></tr>
      </table>
      {% endif %}
    </div>
  </div>
  <div class="panel" id="tab-network">
    <div class="card">
      <p class="lead">{{ insights[6] }}</p>
      {% if charts_json.network %}<div id="chart-network" class="chart"></div>
      {% if charts.network %}<img id="img-network" class="chart" alt="Conversation network" src="{{ charts.network }}" style="display:none">{% endif %}
      {% else %}{% if charts.network %}<img class="chart" alt="Conversation network" src="{{ charts.network }}">{% endif %}{% endif %}
      {% if network %}
      <table>
        <tr><th>Nodes</th><td>{{ network.node_count }}</td></tr>
        <tr><th>Edges</th><td>{{ network.edge_count }}</td></tr>
        <tr><th>Density</th><td>{{ network.density }}</td></tr>
        {% if network.strongest_connections %}
        <tr><th>Strongest connection</th><td>{{ network.strongest_connections[0]['from'] }} &rarr; {{ network.strongest_connections[0]['to'] }}</td></tr>
        {% endif %}
      </table>
      {% endif %}
    </div>
  </div>
  <div class="panel" id="tab-emotion">
    <div class="card">
      {% if emotion %}
      <p class="lead">{{ insights[7] }}</p>
      {% if charts_json.emotion %}<div id="chart-emotion" class="chart"></div>{% else %}{% if charts.emotion %}<img class="chart" alt="Emotion distribution" src="{{ charts.emotion }}">{% endif %}{% endif %}
      <table>
        <tr><th>Emotion</th><th>Messages</th></tr>
        {% for label, count in emotion.distribution.items() %}
        <tr><td>{{ label }}</td><td>{{ count }}</td></tr>
        {% endfor %}
      </table>
      {% if emotion.dominant %}<p>Dominant emotion: {{ emotion.dominant }}</p>{% endif %}
      {% else %}
      <p>Emotion analysis unavailable. Install the optional NLP extras: <code>pip install chat-analyzer-pro[nlp]</code>.</p>
      {% endif %}
    </div>
  </div>
  <div class="panel" id="tab-narrative">
    <div class="card">
      <p class="lead">{{ narrative.lead }}</p>
      {% if narrative.narrative_summary %}
      <p class="narrative-summary">{{ narrative.narrative_summary }}</p>
      {% endif %}
      {% if narrative.observations %}
      <ul>
        {% for obs in narrative.observations %}
        <li>{{ obs.text }} <span class="obs-meta">confidence: {{ obs.confidence }}</span> <span class="obs-meta">{{ obs.kind }}</span></li>
        {% endfor %}
      </ul>
      {% endif %}
    </div>
  </div>
</main>
<script>{{ echarts_js | safe }}</script>
<script>{{ echarts_gl_js | safe }}</script>
<script>
var CHART_SPECS = {
  timeline: {{ charts_json.timeline | tojson }},
  activity: {{ charts_json.activity | tojson }},
  participants: {{ charts_json.participants | tojson }},
  sentiment: {{ charts_json.sentiment | tojson }},
  health: {{ charts_json.health | tojson }},
  network: {{ charts_json.network | tojson }},
  emotion: {{ charts_json.emotion | tojson }}
};
</script>
<script>{{ init_js | safe }}</script>
</body>
</html>
"""


def sanitize_filename(name: str) -> str:
    """Sanitize a filename stem for cross-platform safety (D-14).

    Strips path separators, Windows-invalid characters and control chars;
    strips leading dots/whitespace; falls back to "chat_analysis" if the
    result is empty.
    """
    safe = _INVALID_FILENAME_CHARS.sub("", name).strip(" .")
    return safe or "chat_analysis"


def _json_serializable(value) -> bool:
    """True when the value is strictly JSON-serializable (NaN drops out).

    The boundary guard for charts_json: a spec that cannot round-trip through
    JSON (numpy scalars, NaN, objects) is dropped here so write_report falls
    back to the chart's base64 PNG instead of emitting broken JS.
    """
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError):
        return False
    return True


def _validate_charts_json(charts_json) -> dict[str, dict]:
    """Boundary-validate the interactive ECharts specs before the template.

    Only dictionary, JSON-serializable specs reach the template; anything
    else degrades to an empty spec ({}) so the chart slot renders its PNG
    fallback (`{% if charts_json.NAME %}` is falsy for {}).
    """
    raw = charts_json if isinstance(charts_json, dict) else {}
    validated: dict[str, dict] = {}
    for name in _CHART_KEYS:
        spec = raw.get(name)
        validated[name] = spec if isinstance(spec, dict) and _json_serializable(spec) else {}
    return validated


def write_report(results: AnalysisResults, input_path: Path) -> Path:
    """Render the single-file HTML report to the cwd (D-09/D-14).

    The report is ALWAYS generated — no flags (D-08) — and lands in the
    current working directory as <sanitized-stem>_report.html.
    """
    # Validate chart URIs at the boundary — only internally generated PNG
    # data URIs reach the template (no |safe needed; T-02-01).
    charts = {
        name: (uri if uri.startswith(_CHART_PREFIX) else "")
        for name, uri in results["charts"].items()
    }

    # Validate the interactive specs at the boundary too: only dict,
    # JSON-serializable specs reach the template (`|tojson`), and dropped
    # specs render their PNG fallback (`{% if charts_json.NAME %}`).
    charts_json = _validate_charts_json(results.get("charts_json"))

    narrative = dict(results.get("narrative", {}))
    narrative_status = narrative.get("status") or {}
    if not narrative:
        # Legacy-shaped result with no narrative block at all — neutral placeholder.
        narrative["lead"] = "Analyzing the flow of this conversation."
    elif narrative_status.get("nlp_available") and narrative_status.get(
        "tier_b_generated"
    ):
        narrative["lead"] = "Tier B enabled (local generative model) \u2014 statistical inference is speculative."
    else:
        narrative["lead"] = "Tier A (statistical inference, speculative) \u2014 generative summary disabled."
    narrative.setdefault("narrative_summary", "")
    narrative.setdefault("observations", [])

    env = Environment(autoescape=select_autoescape(["html", "xml"]))
    stem = sanitize_filename(input_path.stem)
    title = stem.replace("_", " ").title()
    subtitle = (
        f"Source: {results['source']} export - "
        f"{results['parse']['parsed_messages']} messages from "
        f"{results['stats']['participants']} participants"
    )
    # The ECharts bundles and the report runtime are OUR trusted, scrubbed
    # constants (never user input) — the |safe filter is deliberate: Jinja
    # autoescape would otherwise escape < && into markup and break the JS.
    html = env.from_string(TEMPLATE).render(
        title=title,
        subtitle=subtitle,
        parse=results["parse"],
        stats=results["stats"],
        participants=results["participants"],
        content=results["content"],
        sentiment=results["sentiment"],
        charts=charts,
        charts_json=charts_json,
        insights=results["insights"],
        health=results.get("health", {}),
        network=results.get("network", {}),
        emotion=results.get("emotion", {}),
        narrative=narrative,
        echarts_js=_load_inline_asset("echarts.min.js"),
        echarts_gl_js=_load_inline_asset("echarts-gl.min.js"),
        init_js=_RUNTIME_JS,
    )

    report_path = Path.cwd() / f"{stem}_report.html"  # D-09: cwd, not input dir
    report_path.write_text(html, encoding="utf-8")  # Pitfall 11: never platform-default
    return report_path


def open_report(path: Path) -> bool:
    """Open the report in the default browser (D-09); degrade without crashing.

    Honors the CHAT_ANALYZER_NO_OPEN=1 opt-out: returns False without ever
    touching webbrowser (no exception, no log).
    """
    if os.environ.get("CHAT_ANALYZER_NO_OPEN") == "1":
        return False
    try:
        return bool(webbrowser.open("file://" + str(path.resolve())))
    except Exception:
        logger.exception("could not open report in browser")
        return False
