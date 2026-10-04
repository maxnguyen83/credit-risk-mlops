"""Publish the run report the DAG's last task produces.

One self-contained HTML file per run: which candidates were trained, how they
compared, what the fairness gate decided, and what each mitigation strategy
cost. It exists because MLflow answers "what were the numbers" well and
"what did we decide, and why" badly -- and the second question is the one
asked in a review three weeks later.

Self-contained means no external CSS and no JavaScript: the file survives being
emailed, committed, or opened from a USB stick during a presentation.

Run it with `python -m credit_risk.models.report`.
"""

from __future__ import annotations

import argparse
import html
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from credit_risk import schema
from credit_risk.config import settings
from credit_risk.models.train import load_training_result, training_result_path

logger = logging.getLogger(__name__)

REPORT_FILE = "run_report.html"

_STYLE = """
body { font: 15px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
       max-width: 62rem; margin: 2rem auto; padding: 0 1.25rem; color: #1a1a1a; }
h1 { font-size: 1.6rem; margin-bottom: .25rem; }
h2 { font-size: 1.15rem; margin-top: 2.25rem; border-bottom: 1px solid #e5e5e5;
     padding-bottom: .3rem; }
.sub { color: #666; margin-top: 0; }
table { border-collapse: collapse; width: 100%; margin: .75rem 0; font-size: .92rem; }
th, td { text-align: left; padding: .45rem .6rem; border-bottom: 1px solid #ececec; }
th { background: #fafafa; font-weight: 600; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
.pass { color: #0a7d32; font-weight: 600; }
.fail { color: #b3261e; font-weight: 600; }
.note { background: #f7f7f9; border-left: 3px solid #bbb; padding: .7rem 1rem;
        margin: 1rem 0; font-size: .92rem; }
code { background: #f2f2f4; padding: .1rem .3rem; border-radius: 3px; font-size: .88em; }
"""


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:,.4f}"
    if isinstance(value, int):
        return f"{value:,}"
    return html.escape(str(value))


def _table(frame: pd.DataFrame, highlight: str | None = None) -> str:
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in frame.columns)
    rows = []
    for _, row in frame.iterrows():
        cells = []
        for col, value in row.items():
            css = "num" if isinstance(value, int | float) else ""
            if highlight and col == highlight:
                css = "pass" if value else "fail"
                value = "PASS" if value else "REFUSED"
            cells.append(f'<td class="{css}">{_fmt(value)}</td>')
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def build_report(result: dict[str, Any]) -> str:
    """Render the HTML. Pure: takes the artefact, returns a string."""
    gate_ok = bool(result.get("gate_passed"))
    reasons = result.get("gate_reasons") or []
    metrics = result.get("metrics", {})
    fairness = result.get("fairness", {})

    candidates = pd.DataFrame(result.get("candidates", []))
    tradeoff = pd.DataFrame(result.get("tradeoff", []))
    by_attr = result.get("fairness_by_attribute", {})

    metrics_frame = pd.DataFrame([{"metric": k, "value": v} for k, v in metrics.items()])
    fairness_frame = pd.DataFrame([{"metric": k, "value": v} for k, v in fairness.items()])

    gate_banner = (
        '<p class="pass">GATE PASSED &mdash; the candidate is registerable.</p>'
        if gate_ok
        else '<p class="fail">GATE REFUSED &mdash; ' + html.escape("; ".join(reasons)) + "</p>"
    )

    attr_section = ""
    if by_attr:
        attr_rows = [{"attribute": attr, **summary} for attr, summary in by_attr.items()]
        attr_section = (
            "<h2>Fairness across every protected attribute</h2>"
            '<p class="sub">The gate is applied to '
            f"<code>{html.escape(schema.PRIMARY_PROTECTED)}</code> only. The rest are audited "
            'so that "it passed" can never mean "it passed on the one attribute we chose to '
            'look at".</p>' + _table(pd.DataFrame(attr_rows))
        )

    ceiling = (
        metrics.get("capacity_fraction", settings.intervention_capacity_fraction)
        / metrics["base_rate"]
        if metrics.get("base_rate")
        else None
    )
    ceiling_note = ""
    if ceiling and metrics.get("recall_at_k") is not None:
        share = metrics["recall_at_k"] / ceiling
        ceiling_note = (
            '<div class="note"><strong>Recall is reported against a ceiling.</strong> '
            f"With {metrics.get('capacity_fraction', 0):.0%} intervention capacity and a base "
            f"rate of {metrics['base_rate']:.4f}, no ranker can exceed "
            f"<code>{ceiling:.4f}</code>. This model reaches {metrics['recall_at_k']:.4f}, "
            f"which is <strong>{share:.1%}</strong> of what is achievable. An absolute recall "
            "target above the ceiling is arithmetic nobody checked, not ambition.</div>"
        )

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Credit-risk run report</title><style>{_STYLE}</style></head><body>
<h1>Credit-risk training run</h1>
<p class="sub">
  winner <code>{html.escape(str(result.get('model')))}</code> &middot;
  run <code>{html.escape(str(result.get('run_id')))}</code> &middot;
  tracking <code>{html.escape(str(result.get('tracking_uri')))}</code>
  {' &middot; <strong>local fallback store</strong>' if result.get('used_fallback_store') else ''}
</p>
{gate_banner}

<h2>Candidates</h2>
<p class="sub">Both families are trained on every run. The baseline is not
ceremony &mdash; it is what turns the interpretability trade-off into a number.</p>
{_table(candidates, highlight="gate_passed") if not candidates.empty else "<p>none</p>"}

<h2>Winning model &mdash; held-out metrics</h2>
{_table(metrics_frame) if not metrics_frame.empty else "<p>none</p>"}
{ceiling_note}

<h2>Fairness &mdash; gated attribute</h2>
{_table(fairness_frame) if not fairness_frame.empty else "<p>none</p>"}
<div class="note">Gates: demographic parity difference &le;
<code>{schema.MAX_DEMOGRAPHIC_PARITY_DIFF}</code>, equalized odds difference &le;
<code>{schema.MAX_EQUALIZED_ODDS_DIFF}</code>. They block registration rather than
appearing in a report, because a number in a report protects only the model that
was examined.</div>

{attr_section}

<h2>Mitigation strategies and what each one costs</h2>
<p class="sub">Every strategy is run on every training run, so the trade-off is
measured rather than remembered.</p>
{_table(tradeoff) if not tradeoff.empty else "<p>none</p>"}
<div class="note"><strong>Read the <code>unawareness</code> row against the probe.</strong>
Dropping the protected column does not remove the attribute: credit limit and
repayment behaviour are proxies for it. The probe ROC-AUC logged with this run is
the evidence.</div>

</body></html>
"""


def write_report(result: dict[str, Any], path: Path | None = None) -> Path:
    target = path or (training_result_path().parent / REPORT_FILE)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(build_report(result), encoding="utf-8")
    logger.info("wrote %s", target)
    return target


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for `python -m credit_risk.models.report`."""
    parser = argparse.ArgumentParser(description="Publish the HTML run report")
    parser.add_argument("--result", type=Path, default=None, help="path to training_result.json")
    parser.add_argument("--out", type=Path, default=None, help="where to write the HTML")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s | %(message)s"
    )
    try:
        result = load_training_result(args.result)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    path = write_report(result, args.out)
    print(json.dumps({"report": str(path), "gate_passed": result.get("gate_passed")}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
