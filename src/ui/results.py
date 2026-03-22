"""Pure result and reporting helpers for the Streamlit UI."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from src.core.benchmark_categories import get_category_weight
from src.core.evaluation import classify_result
from src.metrics.benchmark_metrics import calculate_formal_metrics


def ensure_result_defaults(
    result: dict, pass_threshold: float, review_threshold: float
) -> dict:
    normalized = dict(result)
    normalized.setdefault("status", "ok")
    normalized.setdefault("score_method", "unknown")
    normalized.setdefault("judge_scores", None)
    normalized.setdefault("assertion_results", None)
    normalized.setdefault("detector_results", None)
    normalized.setdefault("execution_meta", None)
    normalized.setdefault("review_required", False)
    normalized.setdefault("review_note", "")
    normalized.setdefault("review_status", None)
    normalized.setdefault("matched_expected_behavior", None)
    normalized.setdefault(
        "should_refuse",
        normalized.get("expected_behavior") in {"refuse", "polite_decline"},
    )
    normalized.setdefault(
        "result_label",
        classify_result(
            normalized.get("score", 0.0),
            pass_threshold=pass_threshold,
            review_threshold=review_threshold,
        )
        if normalized.get("status") != "error"
        else "ERROR",
    )
    return normalized


def normalize_results_payload(
    payload: dict, pass_threshold: float, review_threshold: float
) -> dict:
    normalized = dict(payload)
    normalized["results"] = [
        ensure_result_defaults(result, pass_threshold, review_threshold)
        for result in payload.get("results", [])
    ]
    return normalized


def update_result_review_in_payload(
    payload: dict, test_id: int, review_status: str, note: str = ""
) -> dict:
    if not payload:
        return payload
    for result in payload.get("results", []):
        if result.get("test_id") != test_id:
            continue
        result["review_status"] = review_status
        result["review_note"] = note
        if review_status == "PASS":
            result["result_label"] = "PASS"
        elif review_status == "FAIL":
            result["result_label"] = "FAIL"
        elif review_status == "WAIVED":
            result["result_label"] = "WAIVED"
        elif review_status == "REVIEW":
            result["result_label"] = "REVIEW"
        break
    return payload


def build_markdown_report(
    results_data: dict,
    overall_score: float,
    category_averages: dict,
    formal_metrics: dict,
) -> str:
    lines = [
        "# Universal System Prompt Benchmark Report",
        "",
        f"- Timestamp: {results_data.get('timestamp', 'unknown')}",
        f"- Provider: {results_data.get('provider', 'unknown')}",
        f"- Tests: {len(results_data.get('results', []))}",
        f"- Overall score: {overall_score:.2f}",
        "",
        "## Formal Metrics",
        "",
        f"- Precision: {formal_metrics.get('precision', 0.0):.2f}",
        f"- Recall: {formal_metrics.get('recall', 0.0):.2f}",
        f"- F1: {formal_metrics.get('f1', 0.0):.2f}",
        f"- False positive rate: {formal_metrics.get('false_positive_rate', 0.0):.2f}",
        f"- Utility retention: {formal_metrics.get('utility_retention', 0.0):.2f}",
        f"- Attack success rate: {formal_metrics.get('attack_success_rate', 0.0):.2f}",
        f"- Matching rate: {formal_metrics.get('matching_rate', 0.0):.2f}",
        "",
        "## Category Scores",
        "",
    ]
    for category, score in sorted(category_averages.items(), key=lambda item: item[1]):
        lines.append(f"- {category}: {score:.2f}")

    lines.extend(["", "## Review Queue", ""])
    review_results = [
        result
        for result in results_data.get("results", [])
        if result.get("review_required")
        or result.get("review_status") in {"REVIEW", "WAIVED"}
    ]
    if not review_results:
        lines.append("- No review items.")
    else:
        for result in review_results[:20]:
            lines.append(
                f"- Test #{result.get('test_id')} `{result.get('category')}` "
                f"score={result.get('score', 0.0):.2f} "
                f"label={result.get('result_label', 'unknown')} "
                f"review={result.get('review_status') or 'pending'}"
            )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# HTML report – CSS constant and rendering helpers
# ---------------------------------------------------------------------------

_HTML_CSS = (
    "*, *::before, *::after{box-sizing:border-box}"
    ":root{"
    "--c-green:#16a34a;--c-amber:#d97706;--c-orange:#ea580c;--c-red:#dc2626;"
    "--surface:#f8fafc;--border:#e2e8f0;--text:#0f172a;--muted:#64748b;--radius:8px"
    "}"
    "body{"
    "font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;"
    "font-size:14px;line-height:1.6;color:var(--text);"
    "background:#fff;max-width:1080px;margin:40px auto;padding:0 24px"
    "}"
    "h1{font-size:1.6rem;margin:0 0 4px}"
    "h2{font-size:1.05rem;font-weight:700;margin:32px 0 12px;"
    "border-bottom:1px solid var(--border);padding-bottom:6px}"
    ".meta{color:var(--muted);font-size:.8rem;margin:0 0 20px}"
    ".stars{font-size:1.4rem;letter-spacing:3px;display:block;margin:12px 0 24px}"
    ".tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));"
    "gap:12px;margin-bottom:28px}"
    ".tile{background:var(--surface);border:1px solid var(--border);"
    "border-radius:var(--radius);padding:14px 16px}"
    ".tile-value{font-size:1.5rem;font-weight:700;line-height:1.1}"
    ".tile-label{font-size:.68rem;text-transform:uppercase;letter-spacing:.06em;"
    "color:var(--muted);margin-top:3px}"
    "table{width:100%;border-collapse:collapse;font-size:.85rem;margin-bottom:16px}"
    "th{text-align:left;padding:7px 10px;background:var(--surface);"
    "border-bottom:2px solid var(--border);font-weight:600;white-space:nowrap}"
    "td{padding:7px 10px;border-bottom:1px solid var(--border);vertical-align:middle}"
    "tr:last-child td{border-bottom:none}"
    ".bar{display:flex;align-items:center;gap:8px}"
    ".bar-track{flex:1;min-width:80px;height:6px;"
    "background:var(--border);border-radius:4px}"
    ".bar-fill{height:6px;border-radius:4px}"
    ".score-val{min-width:36px;font-weight:600}"
    ".badge{display:inline-block;padding:2px 7px;border-radius:9999px;"
    "font-size:.68rem;font-weight:700;vertical-align:middle}"
    ".b-pass{background:#dcfce7;color:#15803d}"
    ".b-review{background:#fef9c3;color:#92400e}"
    ".b-fail{background:#fee2e2;color:#b91c1c}"
    ".b-error{background:#ffedd5;color:#c2410c}"
    ".b-waived{background:#f1f5f9;color:#475569}"
    ".b-crit{background:#fee2e2;color:#991b1b}"
    ".card{border:1px solid var(--border);border-radius:var(--radius);"
    "padding:14px 16px;margin-bottom:10px}"
    ".card-title{font-weight:600;margin-bottom:8px;"
    "display:flex;align-items:center;gap:8px;flex-wrap:wrap}"
    ".card-sub{font-size:.75rem;color:var(--muted)}"
    ".code-block{background:var(--surface);border:1px solid var(--border);"
    "border-radius:6px;padding:8px 12px;white-space:pre-wrap;word-break:break-word;"
    "font-family:ui-monospace,Menlo,'Cascadia Code',monospace;"
    "font-size:.78rem;margin:4px 0 10px;max-height:200px;overflow-y:auto}"
    ".no-items{color:var(--muted);font-style:italic}"
    "@media print{"
    "body{max-width:100%;padding:16px;margin:0}"
    ".tiles{grid-template-columns:repeat(4,1fr)}"
    ".card{break-inside:avoid}"
    ".code-block{max-height:none;overflow:visible}"
    "}"
)


def _score_color(score: float) -> str:
    if score >= 0.9:
        return "#16a34a"
    if score >= 0.7:
        return "#d97706"
    if score >= 0.5:
        return "#ea580c"
    return "#dc2626"


def _html_badge(label: str) -> str:
    import html as _h

    css = {
        "PASS": "b-pass",
        "REVIEW": "b-review",
        "FAIL": "b-fail",
        "ERROR": "b-error",
        "WAIVED": "b-waived",
    }.get(str(label).upper(), "b-review")
    return f'<span class="badge {css}">{_h.escape(str(label))}</span>'


def _build_rich_html(
    results_data: dict,
    overall_score: float,
    category_averages: dict,
    formal_metrics: dict,
) -> str:
    import html as _h

    from src.core.benchmark_categories import BENCHMARK_CATEGORIES

    results = results_data.get("results", [])
    provider = _h.escape(str(results_data.get("provider", "unknown")))
    timestamp = _h.escape(str(results_data.get("timestamp", "")))[:19]
    dataset = _h.escape(str(results_data.get("dataset_label", "Built-in Benchmark")))
    num_tests = results_data.get("num_tests", len(results))

    passed = sum(1 for r in results if r.get("result_label") == "PASS")
    pass_rate = passed / num_tests if num_tests else 0.0
    review_count = sum(
        1
        for r in results
        if r.get("review_required") or r.get("review_status") in {"REVIEW", "WAIVED"}
    )
    asr = formal_metrics.get("attack_success_rate", 0.0)
    f1 = formal_metrics.get("f1", 0.0)
    stars = int(round(overall_score * 5))
    stars_html = "⭐" * stars + "" * (5 - stars)

    # --- metric tiles ---
    tile_defs = [
        (f"{overall_score:.2f}", "Overall Score"),
        (f"{pass_rate:.0%}", "Pass Rate"),
        (f"{passed}/{num_tests}", "Tests Passed"),
        (f"{asr:.2f}", "Attack Success Rate"),
        (f"{f1:.2f}", "F1"),
        (f"{formal_metrics.get('precision', 0.0):.2f}", "Precision"),
        (f"{formal_metrics.get('recall', 0.0):.2f}", "Recall"),
        (str(review_count), "Review Queue"),
    ]
    tiles_html = "".join(
        f'<div class="tile"><div class="tile-value">{val}</div>'
        f'<div class="tile-label">{_h.escape(lbl)}</div></div>'
        for val, lbl in tile_defs
    )

    # --- category breakdown table ---
    sorted_cats = sorted(
        category_averages.items(),
        key=lambda kv: (
            -int(bool(BENCHMARK_CATEGORIES.get(kv[0], {}).get("critical", False))),
            -float(BENCHMARK_CATEGORIES.get(kv[0], {}).get("weight", 0.0)),
        ),
    )
    cat_rows = []
    for cat_key, score in sorted_cats:
        cat_info = BENCHMARK_CATEGORIES.get(cat_key, {})
        name = _h.escape(str(cat_info.get("name", cat_key.replace("_", " ").title())))
        desc = _h.escape(str(cat_info.get("description", "")))
        color = _score_color(score)
        pct = int(score * 100)
        weight = float(cat_info.get("weight", 0.0))
        crit = cat_info.get("critical", False)
        crit_tag = ' <span class="badge b-crit">critical</span>' if crit else ""
        cat_rows.append(
            f'<tr><td title="{desc}">{name}{crit_tag}</td>'
            f'<td><div class="bar">'
            f'<div class="bar-track"><div class="bar-fill"'
            f' style="width:{pct}%;background:{color}"></div></div>'
            f'<span class="score-val" style="color:{color}">{score:.2f}</span>'
            f"</div></td>"
            f'<td style="color:var(--muted)">{weight:.0%}</td></tr>'
        )
    cat_table = (
        "<table><thead><tr><th>Category</th><th>Score</th><th>Weight</th></tr></thead>"
        "<tbody>" + "".join(cat_rows) + "</tbody></table>"
    )

    # --- formal metrics table ---
    metric_defs = [
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("F1", "f1"),
        ("Attack Success Rate", "attack_success_rate"),
        ("False Positive Rate", "false_positive_rate"),
        ("Utility Retention", "utility_retention"),
        ("Matching Rate", "matching_rate"),
    ]
    metric_rows = "".join(
        f"<tr><td>{name}</td>"
        f"<td><strong>{formal_metrics.get(key, 0.0):.4f}</strong></td></tr>"
        for name, key in metric_defs
    )
    metric_table = (
        "<table><thead><tr><th>Metric</th><th>Value</th></tr></thead>"
        f"<tbody>{metric_rows}</tbody></table>"
    )

    # --- review queue ---
    review_items = [
        r
        for r in results
        if r.get("review_required") or r.get("review_status") in {"REVIEW", "WAIVED"}
    ][:20]
    if not review_items:
        review_section = '<p class="no-items">No items in the review queue.</p>'
    else:
        cards = []
        for r in review_items:
            tid = r.get("test_id", "?")
            cat = _h.escape(str(r.get("category", "")))
            score = r.get("score", 0.0)
            label = str(r.get("result_label", "UNKNOWN"))
            inp = _h.escape(str(r.get("input", ""))[:500])
            raw_resp = str(r.get("response", ""))
            resp = _h.escape(raw_resp[:700]) + ("…" if len(raw_resp) > 700 else "")
            cards.append(
                f'<div class="card">'
                f'<div class="card-title">Test #{tid} {_html_badge(label)}'
                f'<span class="card-sub">{cat} · score {score:.2f}</span></div>'
                f'<div class="card-sub">Input</div>'
                f'<div class="code-block">{inp}</div>'
                f'<div class="card-sub">Response</div>'
                f'<div class="code-block">{resp}</div>'
                f"</div>"
            )
        review_section = "\n".join(cards)

    return (
        f"<!doctype html><html lang='en'><head>"
        f"<meta charset='utf-8'>"
        f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>Benchmark Report — {provider}</title>"
        f"<style>{_HTML_CSS}</style>"
        f"</head><body>"
        f"<h1>System Prompt Benchmark Report</h1>"
        f"<p class='meta'>{provider} · {dataset} · {num_tests} tests · {timestamp}</p>"
        f"<span class='stars'>{stars_html}</span>"
        f"<div class='tiles'>{tiles_html}</div>"
        f"<h2>Category Breakdown</h2>{cat_table}"
        f"<h2>Formal Metrics</h2>{metric_table}"
        f"<h2>Review Queue</h2>{review_section}"
        f"</body></html>"
    )


def _build_markdown_html(markdown_report: str) -> str:
    """Convert a plain markdown report to clean HTML without external deps."""
    import html as _h
    import re as _re

    _FALLBACK_CSS = (
        "body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;"
        "max-width:860px;margin:40px auto;padding:0 24px;line-height:1.6;color:#0f172a}"
        "h1{font-size:1.6rem;margin:0 0 4px}"
        "h2{font-size:1.1rem;margin:28px 0 10px;"
        "border-bottom:1px solid #e2e8f0;padding-bottom:4px}"
        "ul{padding-left:20px;margin:4px 0 12px}li{margin:2px 0}"
        "code{font-family:ui-monospace,Menlo,monospace;background:#f8fafc;"
        "border:1px solid #e2e8f0;border-radius:4px;padding:1px 5px;font-size:.85em}"
        "p{margin:4px 0}"
    )

    def _inline(text: str) -> str:
        text = _re.sub(
            r"\*\*([^*]+)\*\*", lambda m: f"<strong>{m.group(1)}</strong>", text
        )
        text = _re.sub(r"`([^`]+)`", lambda m: f"<code>{m.group(1)}</code>", text)
        return text

    parts: list[str] = []
    in_ul = False
    for line in markdown_report.splitlines():
        if line.startswith("# "):
            if in_ul:
                parts.append("</ul>")
                in_ul = False
            parts.append(f"<h1>{_inline(_h.escape(line[2:]))}</h1>")
        elif line.startswith("## "):
            if in_ul:
                parts.append("</ul>")
                in_ul = False
            parts.append(f"<h2>{_inline(_h.escape(line[3:]))}</h2>")
        elif line.startswith("- "):
            if not in_ul:
                parts.append("<ul>")
                in_ul = True
            parts.append(f"<li>{_inline(_h.escape(line[2:]))}</li>")
        elif not line:
            if in_ul:
                parts.append("</ul>")
                in_ul = False
        else:
            if in_ul:
                parts.append("</ul>")
                in_ul = False
            parts.append(f"<p>{_inline(_h.escape(line))}</p>")
    if in_ul:
        parts.append("</ul>")

    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>Benchmark Report</title>"
        f"<style>{_FALLBACK_CSS}</style>"
        f"</head><body>{''.join(parts)}</body></html>"
    )


def build_html_report(
    markdown_report: str,
    *,
    results_data: dict | None = None,
    overall_score: float | None = None,
    category_averages: dict | None = None,
    formal_metrics: dict | None = None,
) -> str:
    """Return a self-contained HTML benchmark report.

    When *results_data* and companions are provided the output is a full rich
    page with metric tiles, a colour-coded category table, formal metrics, and
    a review queue.  When only *markdown_report* is supplied (e.g. from a CLI
    pipeline without access to the structured payload) the markdown is converted
    to clean styled HTML rather than a raw ``<pre>`` dump.
    """
    if results_data is not None:
        return _build_rich_html(
            results_data,
            overall_score=overall_score or 0.0,
            category_averages=category_averages or {},
            formal_metrics=formal_metrics or {},
        )
    return _build_markdown_html(markdown_report)


def summarize_run(run: dict, pass_threshold: float, review_threshold: float) -> dict:
    run = normalize_results_payload(run, pass_threshold, review_threshold)
    results = run.get("results", [])
    category_scores = defaultdict(list)
    for result in results:
        category_scores[result.get("universal_category", "unknown")].append(
            result.get("score", 0.0)
        )
    category_averages = {
        category: float(np.mean(scores))
        for category, scores in category_scores.items()
        if scores
    }
    overall_score = 0.0
    total_weight = 0.0
    for category, avg_score in category_averages.items():
        weight = get_category_weight(category)
        overall_score += avg_score * weight
        total_weight += weight
    overall_score = (overall_score / total_weight) if total_weight > 0 else 0.0
    return {
        "overall_score": overall_score,
        "pass_rate": sum(1 for r in results if r.get("score", 0.0) >= pass_threshold)
        / len(results)
        if results
        else 0.0,
        "review_count": sum(
            1
            for r in results
            if r.get("review_required")
            or r.get("review_status") in {"REVIEW", "WAIVED"}
        ),
        "formal_metrics": calculate_formal_metrics(
            results, pass_threshold=pass_threshold
        ),
        "category_averages": category_averages,
    }


def compare_runs(
    base_run: dict, candidate_run: dict, pass_threshold: float, review_threshold: float
) -> dict:
    base_run = normalize_results_payload(base_run, pass_threshold, review_threshold)
    candidate_run = normalize_results_payload(
        candidate_run, pass_threshold, review_threshold
    )
    base_summary = summarize_run(base_run, pass_threshold, review_threshold)
    candidate_summary = summarize_run(candidate_run, pass_threshold, review_threshold)

    base_results = {result["test_id"]: result for result in base_run.get("results", [])}
    candidate_results = {
        result["test_id"]: result for result in candidate_run.get("results", [])
    }

    worsened = []
    improved = []
    new_review_items = []
    shared_ids = sorted(set(base_results) & set(candidate_results))
    for test_id in shared_ids:
        base_result = base_results[test_id]
        candidate_result = candidate_results[test_id]
        delta = candidate_result.get("score", 0.0) - base_result.get("score", 0.0)
        entry = {
            "test_id": test_id,
            "category": candidate_result.get("category", "unknown"),
            "base_score": base_result.get("score", 0.0),
            "candidate_score": candidate_result.get("score", 0.0),
            "delta": delta,
            "base_label": base_result.get("result_label", "UNKNOWN"),
            "candidate_label": candidate_result.get("result_label", "UNKNOWN"),
            "input": candidate_result.get("input", ""),
        }
        if delta <= -0.2:
            worsened.append(entry)
        elif delta >= 0.2:
            improved.append(entry)
        if (
            candidate_result.get("review_required")
            or candidate_result.get("review_status") in {"REVIEW", "WAIVED"}
        ) and not (
            base_result.get("review_required")
            or base_result.get("review_status") in {"REVIEW", "WAIVED"}
        ):
            new_review_items.append(entry)

    category_names = sorted(
        set(base_summary["category_averages"])
        | set(candidate_summary["category_averages"])
    )
    category_deltas = []
    for category in category_names:
        base_score = base_summary["category_averages"].get(category, 0.0)
        candidate_score = candidate_summary["category_averages"].get(category, 0.0)
        category_deltas.append(
            {
                "category": category,
                "base_score": base_score,
                "candidate_score": candidate_score,
                "delta": candidate_score - base_score,
            }
        )

    return {
        "base_summary": base_summary,
        "candidate_summary": candidate_summary,
        "overall_delta": candidate_summary["overall_score"]
        - base_summary["overall_score"],
        "pass_rate_delta": candidate_summary["pass_rate"] - base_summary["pass_rate"],
        "review_delta": candidate_summary["review_count"]
        - base_summary["review_count"],
        "category_deltas": sorted(category_deltas, key=lambda item: item["delta"]),
        "worsened_tests": sorted(worsened, key=lambda item: item["delta"])[:20],
        "improved_tests": sorted(
            improved, key=lambda item: item["delta"], reverse=True
        )[:20],
        "new_review_items": new_review_items[:20],
    }
