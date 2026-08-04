"""Single-file HTML report card for the chat-analyzer CLI (D-08..D-14).

research Pattern 4 — one self-contained HTML file: jinja2 autoescape
template (inline constant, no external assets, no CDN), base64 PNG chart
URIs, sanitized filename next to the input, utf-8 write, best-effort
auto-open. Chat content is UNTRUSTED input: autoescape is set explicitly
(plain jinja2 defaults to False) and chart URIs are validated at the
boundary before they reach the template.
"""

from __future__ import annotations

import logging
import re
import webbrowser
from pathlib import Path

from jinja2 import Environment, select_autoescape

from chat_analyzer.cli.contracts import AnalysisResults

logger = logging.getLogger(__name__)

_INVALID_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f\x7f]')

_CHART_PREFIX = "data:image/png;base64,"

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
  table { border-collapse: collapse; width: 100%; margin-top: 12px; }
  th, td { border: 1px solid #e4e7ec; padding: 8px 10px; text-align: left; font-size: 14px; }
  th { background: #f2f4f8; }
  ul { margin-top: 8px; }
  .skip-note { color: #b45309; background: #fef3c7; padding: 8px 12px; border-radius: 6px; margin: 12px 32px 0; }
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
</nav>
<main>
  <div class="panel active" id="tab-overview">
    <div class="card">
      <p class="lead">{{ insights[0] }}</p>
      {% if charts.timeline %}<img class="chart" alt="Message timeline" src="{{ charts.timeline }}">{% endif %}
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
      {% if charts.participants %}<img class="chart" alt="Participant activity" src="{{ charts.participants }}">{% endif %}
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
      {% if charts.activity %}<img class="chart" alt="Activity heatmap" src="{{ charts.activity }}">{% endif %}
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
      {% if charts.sentiment %}<img class="chart" alt="Sentiment over time" src="{{ charts.sentiment }}">{% endif %}
      <table>
        <tr><th>Sentiment</th><th>Messages</th></tr>
        {% for label, count in sentiment.distribution.items() %}
        <tr><td>{{ label }}</td><td>{{ count }}</td></tr>
        {% endfor %}
      </table>
    </div>
  </div>
</main>
<script>
function showTab(id) {
  var panels = document.querySelectorAll('.panel');
  for (var i = 0; i < panels.length; i++) {
    panels[i].classList.toggle('active', panels[i].id === 'tab-' + id);
  }
  var buttons = document.querySelectorAll('button.tab');
  for (var j = 0; j < buttons.length; j++) {
    buttons[j].classList.toggle('active', buttons[j].dataset.tab === id);
  }
}
</script>
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


def write_report(results: AnalysisResults, input_path: Path) -> Path:
    """Render the single-file HTML report next to the input (D-08/D-14)."""
    # Validate chart URIs at the boundary — only internally generated PNG
    # data URIs reach the template (no |safe needed; T-02-01).
    charts = {
        name: (uri if uri.startswith(_CHART_PREFIX) else "")
        for name, uri in results["charts"].items()
    }

    env = Environment(autoescape=select_autoescape(["html", "xml"]))
    stem = sanitize_filename(input_path.stem)
    title = stem.replace("_", " ").title()
    subtitle = (
        f"Source: {results['source']} export - "
        f"{results['parse']['parsed_messages']} messages from "
        f"{results['stats']['participants']} participants"
    )
    html = env.from_string(TEMPLATE).render(
        title=title,
        subtitle=subtitle,
        parse=results["parse"],
        stats=results["stats"],
        participants=results["participants"],
        content=results["content"],
        sentiment=results["sentiment"],
        charts=charts,
        insights=results["insights"],
    )

    report_path = input_path.parent / f"{stem}_report.html"
    report_path.write_text(html, encoding="utf-8")  # Pitfall 11: never platform-default
    return report_path


def open_report(path: Path) -> bool:
    """Open the report in the default browser (D-09); degrade without crashing."""
    try:
        return bool(webbrowser.open("file://" + str(path.resolve())))
    except Exception:
        logger.exception("could not open report in browser")
        return False
