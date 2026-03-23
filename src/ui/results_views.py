"""Streamlit result-related views."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.core.benchmark_categories import BENCHMARK_CATEGORIES, get_category_weight
from src.metrics.benchmark_metrics import (
    calculate_formal_metrics,
    calculate_metrics_by_category,
)
from src.plugins.manager import get_plugin_manager
from src.ui.analytics_views import (
    render_attack_type_breakdown,
    render_detector_analytics,
)
from src.ui.results import (
    build_html_report,
    build_markdown_report,
    normalize_results_payload,
)
from src.utils.pdf_report import generate_pdf_report


def render_results_section(
    results_payload: dict,
    run_metadata: dict,
    last_run_logs: list[str],
    pass_threshold: float,
    review_threshold: float,
    on_update_review,
) -> dict:
    results_data = normalize_results_payload(
        results_payload, pass_threshold, review_threshold
    )
    results = results_data["results"]

    st.divider()
    st.header("Results")
    if run_metadata:
        st.info(
            f"Last run: **{run_metadata.get('mode', 'unknown')}** · "
            f"{run_metadata.get('num_tests', len(results))} tests · "
            f"{run_metadata.get('provider', 'provider')} ({run_metadata.get('model', 'model')}) · "
            f"{run_metadata.get('dataset_label', results_data.get('dataset_label', 'dataset'))} "
            f"v{run_metadata.get('dataset_version', results_data.get('dataset_metadata', {}).get('version', '1.0'))} · "
            f"{results_data['timestamp']}"
        )
    _render_provider_capabilities(run_metadata, results_data)
    _render_provider_test_result(run_metadata)
    _render_detector_summary(results)

    category_scores = defaultdict(list)
    for result in results:
        category_scores[result["universal_category"]].append(result["score"])

    category_averages = {
        category: float(np.mean(scores)) for category, scores in category_scores.items()
    }
    formal_metrics = calculate_formal_metrics(results, pass_threshold=pass_threshold)
    category_metrics = calculate_metrics_by_category(
        results, pass_threshold=pass_threshold
    )

    total_score = 0.0
    total_weight = 0.0
    for category, avg_score in category_averages.items():
        weight = get_category_weight(category)
        total_score += avg_score * weight
        total_weight += weight
    overall_score = total_score / total_weight if total_weight > 0 else 0.0

    passed = sum(1 for result in results if result["score"] >= pass_threshold)
    avg_latency = np.mean([result["latency"] for result in results])
    total_tokens = sum(result["tokens"] for result in results)
    review_needed = sum(
        1
        for result in results
        if result.get("review_required")
        or result.get("review_status") in {"REVIEW", "WAIVED"}
    )

    _render_score_hero(
        overall_score=overall_score,
        passed=passed,
        total=len(results),
        review_needed=review_needed,
        avg_latency=avg_latency,
        total_tokens=total_tokens,
        asr=formal_metrics.get("attack_success_rate", 0.0),
        f1=formal_metrics.get("f1", 0.0),
        precision=formal_metrics.get("precision", 0.0),
        recall=formal_metrics.get("recall", 0.0),
        matching_rate=formal_metrics.get("matching_rate", 0.0),
    )
    st.divider()

    result_tabs = st.tabs(
        [
            " Categories",
            " Charts",
            " Results",
            " Detectors",
            " Attacks",
            " Review",
            " Export",
            " Log",
        ]
    )

    with result_tabs[0]:
        _render_category_breakdown(category_averages, category_scores, category_metrics)
    with result_tabs[1]:
        _render_visualizations(category_averages)
    with result_tabs[2]:
        _render_detailed_results(results, category_averages)
    with result_tabs[3]:
        render_detector_analytics(results)
    with result_tabs[4]:
        render_attack_type_breakdown(results)
    with result_tabs[5]:
        _render_review_queue(results, on_update_review)
    with result_tabs[6]:
        _render_export_view(
            results_data, results, overall_score, category_averages, formal_metrics
        )
    with result_tabs[7]:
        _render_live_log(last_run_logs)

    return {
        "results_data": results_data,
        "results": results,
        "category_averages": category_averages,
        "formal_metrics": formal_metrics,
        "overall_score": overall_score,
    }


def _render_score_hero(
    *,
    overall_score: float,
    passed: int,
    total: int,
    review_needed: int,
    avg_latency: float,
    total_tokens: int,
    asr: float,
    f1: float,
    precision: float,
    recall: float,
    matching_rate: float,
) -> None:
    """Render the prominent score card + supporting metric row."""
    # Determine grade and color
    if overall_score >= 0.9:
        color, grade, badge_bg, badge_color = (
            "#16a34a",
            "Excellent",
            "#dcfce7",
            "#15803d",
        )
    elif overall_score >= 0.7:
        color, grade, badge_bg, badge_color = "#d97706", "Good", "#fef9c3", "#92400e"
    elif overall_score >= 0.5:
        color, grade, badge_bg, badge_color = (
            "#ea580c",
            "Needs Work",
            "#ffedd5",
            "#9a3412",
        )
    else:
        color, grade, badge_bg, badge_color = (
            "#dc2626",
            "Vulnerable",
            "#fee2e2",
            "#991b1b",
        )

    pct = int(overall_score * 100)
    pass_rate = passed / total if total else 0.0

    # Category bars (overall + pass rate)
    bars_html = ""
    for label, value, val_str in [
        ("Overall Score", overall_score, f"{overall_score:.2f}"),
        ("Pass Rate", pass_rate, f"{pass_rate:.0%}  ({passed}/{total})"),
        ("Attack Success Rate ↓", asr, f"{asr:.2f}"),
    ]:
        bar_color = (
            color
            if label != "Attack Success Rate ↓"
            else ("#16a34a" if asr <= 0.1 else "#d97706" if asr <= 0.3 else "#dc2626")
        )
        fill_pct = int(value * 100)
        bars_html += (
            f'<div class="score-hero-bar-row">'
            f'<span class="score-hero-bar-label">{label}</span>'
            f'<div class="score-hero-bar-track">'
            f'<div class="score-hero-bar-fill" style="width:{fill_pct}%;background:{bar_color}"></div>'
            f"</div>"
            f'<span class="score-hero-bar-val" style="color:{bar_color}">{val_str}</span>'
            f"</div>"
        )

    meta_html = (
        f"<span>⏱ <b>{avg_latency:.2f}s</b> avg latency</span>"
        f"<span> <b>{total_tokens:,}</b> tokens</span>"
        f"<span> <b>{review_needed}</b> in review queue</span>"
        f"<span> F1 <b>{f1:.2f}</b></span>"
        f"<span> Precision <b>{precision:.2f}</b></span>"
        f"<span> Recall <b>{recall:.2f}</b></span>"
        f"<span> Matching <b>{matching_rate:.2f}</b></span>"
    )

    st.markdown(
        f'<div class="score-hero">'
        f'  <div class="score-hero-left">'
        f'    <div class="score-hero-value" style="color:{color}">{overall_score:.2f}</div>'
        f'    <div class="score-hero-label">Overall Score</div>'
        f'    <div class="score-hero-grade" style="background:{badge_bg};color:{badge_color}">'
        f"      {grade}"
        f"    </div>"
        f"  </div>"
        f'  <div class="score-hero-right">'
        f"    {bars_html}"
        f'    <div class="score-hero-meta">{meta_html}</div>'
        f"  </div>"
        f"</div>",
        unsafe_allow_html=True,
    )


def _render_provider_capabilities(run_metadata: dict, results_data: dict) -> None:
    capabilities = (
        run_metadata.get("provider_capabilities")
        or results_data.get("provider_capabilities")
        or {}
    )
    if not capabilities:
        return

    transport = run_metadata.get("provider_transport") or capabilities.get(
        "transport", "unknown"
    )
    with st.expander("Provider Capabilities", expanded=False):
        metric_cols = st.columns(4)
        with metric_cols[0]:
            st.metric(
                "Provider",
                capabilities.get("provider", results_data.get("provider", "unknown")),
            )
        with metric_cols[1]:
            st.metric("Transport", transport)
        with metric_cols[2]:
            st.metric(
                "System Prompt",
                "yes" if capabilities.get("supports_system_prompt", True) else "no",
            )
        with metric_cols[3]:
            st.metric(
                "Model", capabilities.get("model", run_metadata.get("model", "unknown"))
            )

        if capabilities.get("base_url"):
            st.caption(f"Base URL: `{capabilities['base_url']}`")
        if capabilities.get("response_text_path"):
            st.caption(f"Response text path: `{capabilities['response_text_path']}`")
        if capabilities.get("response_tokens_path"):
            st.caption(
                f"Response tokens path: `{capabilities['response_tokens_path']}`"
            )
        st.json(capabilities)


def _render_provider_test_result(run_metadata: dict) -> None:
    test_result = run_metadata.get("provider_test_result") or {}
    if not test_result:
        return

    with st.expander("Last Provider Test", expanded=False):
        if test_result.get("status") == "ok":
            st.success(
                f"{test_result.get('provider', 'provider')} responded in "
                f"{test_result.get('latency', 0.0):.2f}s before the benchmark run."
            )
            cols = st.columns(3)
            with cols[0]:
                st.metric("Latency", f"{test_result.get('latency', 0.0):.2f}s")
            with cols[1]:
                st.metric("Tokens", int(test_result.get("tokens", 0)))
            with cols[2]:
                st.metric(
                    "Transport",
                    test_result.get("capabilities", {}).get("transport", "unknown"),
                )
            if test_result.get("user_message"):
                st.caption(f"Probe input: `{test_result['user_message']}`")
            st.code(test_result.get("response", ""))
        else:
            st.error(
                f"Provider test failed before run: {test_result.get('error', 'unknown error')}"
            )


def _render_detector_summary(results: list[dict]) -> None:
    detector_counts = defaultdict(int)
    detector_failures = defaultdict(int)
    for result in results:
        detector_payload = result.get("detector_results") or {}
        for item in detector_payload.get("results", []):
            name = item.get("name", "unknown")
            detector_counts[name] += 1
            if not item.get("matched", False):
                detector_failures[name] += 1
    if not detector_counts:
        return
    with st.expander("Detector Summary", expanded=False):
        rows = []
        for name in sorted(detector_counts):
            total = detector_counts[name]
            failed = detector_failures.get(name, 0)
            rows.append(
                {
                    "detector": name,
                    "evaluated": total,
                    "flagged_or_failed": failed,
                    "clean_rate": round((total - failed) / total, 4) if total else 0.0,
                }
            )
        st.dataframe(pd.DataFrame(rows), width="stretch")


def _render_category_breakdown(
    category_averages: dict, category_scores: dict, category_metrics: dict
) -> None:
    st.subheader("Category Scores")

    sorted_categories = []
    for category_name, category_info in BENCHMARK_CATEGORIES.items():
        if category_name in category_averages:
            sorted_categories.append(
                (
                    category_name,
                    category_info,
                    category_averages[category_name],
                    len(category_scores[category_name]),
                )
            )

    sorted_categories.sort(key=lambda item: (-item[1]["critical"], -item[1]["weight"]))

    for category_name, category_info, score, test_count in sorted_categories:
        if score >= 0.9:
            icon = ""
            color_class = "score-excellent"
        elif score >= 0.7:
            icon = ""
            color_class = "score-good"
        elif score >= 0.5:
            icon = ""
            color_class = "score-warning"
        else:
            icon = ""
            color_class = "score-poor"

        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"{icon} **{category_info['name']}** ({test_count} tests)")
            st.progress(score)
            st.caption(category_info["description"])
        with col2:
            st.markdown(
                f'<div class="category-score {color_class}">{score:.2f}</div>',
                unsafe_allow_html=True,
            )
            metrics = category_metrics.get(category_name, {})
            st.caption(
                f"Weight: {category_info['weight'] * 100:.0f}% · "
                f"ASR: {metrics.get('attack_success_rate', 0.0):.2f}"
            )


def _render_visualizations(category_averages: dict) -> None:
    st.subheader("Visualizations")
    sorted_categories = []
    for category_name, category_info in BENCHMARK_CATEGORIES.items():
        if category_name in category_averages:
            sorted_categories.append(
                (category_name, category_info["name"], category_averages[category_name])
            )
    categories = [category_name for _, category_name, _ in sorted_categories]
    scores = [score for _, _, score in sorted_categories]

    fig_radar = go.Figure()
    fig_radar.add_trace(
        go.Scatterpolar(
            r=scores,
            theta=categories,
            fill="toself",
            name="Score",
        )
    )
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=False,
        title="Category Scores (Radar)",
    )
    st.plotly_chart(fig_radar, width="stretch")

    fig_bar = px.bar(
        x=categories,
        y=scores,
        labels={"x": "Category", "y": "Score"},
        title="Category Scores (Bar)",
        color=scores,
        color_continuous_scale="RdYlGn",
    )
    fig_bar.update_layout(showlegend=False)
    st.plotly_chart(fig_bar, width="stretch")


def _render_detailed_results(results: list[dict], category_averages: dict) -> None:
    st.subheader("Detailed Test Results")
    selected_cat = st.selectbox(
        "Filter by category", ["All"] + list(category_averages.keys())
    )
    filtered_results = (
        results
        if selected_cat == "All"
        else [
            result for result in results if result["universal_category"] == selected_cat
        ]
    )

    label_filter = st.selectbox(
        "Filter by label", ["All", "PASS", "REVIEW", "FAIL", "ERROR", "WAIVED"]
    )
    if label_filter != "All":
        filtered_results = [
            result
            for result in filtered_results
            if result.get("result_label") == label_filter
        ]

    filtered_results.sort(key=lambda item: item["score"])
    for result in filtered_results[:20]:
        label = result.get("result_label", "UNKNOWN")
        if label == "PASS":
            badge = "🟢"
        elif label in {"REVIEW", "WAIVED"}:
            badge = "🟡"
        elif label == "ERROR":
            badge = "🟠"
        else:
            badge = ""

        with st.expander(
            f"{badge} Test #{result['test_id']} - {result['category']} "
            f"(Score: {result['score']:.2f}, {label})"
        ):
            st.write("**Input:**")
            st.code(result["input"])
            st.write("**Response:**")
            st.code(
                result["response"][:500]
                + ("..." if len(result["response"]) > 500 else "")
            )

            cols = st.columns(4)
            with cols[0]:
                st.metric("Score", f"{result['score']:.2f}")
            with cols[1]:
                st.metric("Tokens", result["tokens"])
            with cols[2]:
                st.metric("Latency", f"{result['latency']:.2f}s")
            with cols[3]:
                st.metric("Judge", result.get("score_method", "unknown"))

            st.caption(
                f"Expected behavior: {result.get('expected_behavior', 'unknown')} · "
                f"Review required: {'yes' if result.get('review_required') else 'no'} · "
                f"Matched expected: {'yes' if result.get('matched_expected_behavior') else 'no'}"
            )
            if result.get("execution_meta"):
                execution_meta = result["execution_meta"]
                st.caption(
                    f"Execution mode: {execution_meta.get('mode', 'unknown')} · "
                    f"Turns: {execution_meta.get('turn_count', 1)} · "
                    f"Artifacts: {execution_meta.get('artifacts_used', 0)}"
                )
                if execution_meta.get("trace"):
                    with st.expander("Execution Trace", expanded=False):
                        st.json(execution_meta["trace"])
            if result.get("assertion_results"):
                _render_assertion_results(result["assertion_results"])
            if result.get("detector_results"):
                _render_detector_results(result["detector_results"])
            if result.get("judge_scores"):
                st.json(result["judge_scores"])
            if result.get("review_note"):
                st.info(f"Review note: {result['review_note']}")


def _render_review_queue(results: list[dict], on_update_review) -> None:
    st.subheader("Manual Review Queue")
    review_results = [
        result
        for result in results
        if result.get("review_required")
        or result.get("review_status") in {"REVIEW", "WAIVED"}
    ]
    if not review_results:
        st.success("No review items.")
        return

    if "review_notes" not in st.session_state:
        st.session_state.review_notes = {}

    for result in review_results:
        test_id = result["test_id"]
        note_key = f"review_note_{test_id}"
        current_note = result.get("review_note", "")
        st.session_state.review_notes.setdefault(note_key, current_note)
        with st.expander(
            f"Test #{test_id} · {result['category']} · "
            f"score {result['score']:.2f} · current {result.get('result_label', 'UNKNOWN')}"
        ):
            st.write("**Input**")
            st.code(result["input"])
            st.write("**Response**")
            st.code(
                result["response"][:800]
                + ("..." if len(result["response"]) > 800 else "")
            )
            if result.get("execution_meta"):
                execution_meta = result["execution_meta"]
                st.caption(
                    f"Execution mode: {execution_meta.get('mode', 'unknown')} · "
                    f"Turns: {execution_meta.get('turn_count', 1)} · "
                    f"Artifacts: {execution_meta.get('artifacts_used', 0)}"
                )
                if execution_meta.get("trace"):
                    with st.expander("Execution Trace", expanded=False):
                        st.json(execution_meta["trace"])
            if result.get("assertion_results"):
                _render_assertion_results(result["assertion_results"])
            if result.get("detector_results"):
                _render_detector_results(result["detector_results"])
            if result.get("judge_scores"):
                st.write("**Judge Breakdown**")
                st.json(result["judge_scores"])
            note_value = st.text_area(
                "Review note",
                key=note_key,
                value=st.session_state.review_notes[note_key],
                height=100,
            )
            action_cols = st.columns(4)
            with action_cols[0]:
                if st.button("Mark PASS", key=f"mark_pass_{test_id}"):
                    on_update_review(test_id, "PASS", note_value)
                    st.rerun()
            with action_cols[1]:
                if st.button("Mark FAIL", key=f"mark_fail_{test_id}"):
                    on_update_review(test_id, "FAIL", note_value)
                    st.rerun()
            with action_cols[2]:
                if st.button("Keep REVIEW", key=f"mark_review_{test_id}"):
                    on_update_review(test_id, "REVIEW", note_value)
                    st.rerun()
            with action_cols[3]:
                if st.button("Waive", key=f"mark_waive_{test_id}"):
                    on_update_review(test_id, "WAIVED", note_value)
                    st.rerun()


def _render_export_view(
    results_data: dict,
    results: list[dict],
    overall_score: float,
    category_averages: dict,
    formal_metrics: dict,
) -> None:
    st.subheader("Export Results")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_data = json.dumps(results_data, indent=2)
    st.download_button(
        " Download JSON",
        json_data,
        file_name=f"benchmark_{timestamp}.json",
        mime="application/json",
    )

    csv_data = pd.DataFrame(results).to_csv(index=False)
    st.download_button(
        " Download CSV",
        csv_data,
        file_name=f"benchmark_{timestamp}.csv",
        mime="text/csv",
    )

    markdown_report = build_markdown_report(
        results_data, overall_score, category_averages, formal_metrics
    )
    st.download_button(
        " Download Markdown",
        markdown_report,
        file_name=f"benchmark_{timestamp}.md",
        mime="text/markdown",
    )

    html_report = build_html_report(
        markdown_report,
        results_data=results_data,
        overall_score=overall_score,
        category_averages=category_averages,
        formal_metrics=formal_metrics,
    )
    st.download_button(
        " Download HTML",
        html_report,
        file_name=f"benchmark_{timestamp}.html",
        mime="text/html",
    )

    pdf_payload = {
        "metadata": {
            "provider": results_data.get("provider"),
            "num_tests": results_data.get("num_tests", len(results)),
            "timestamp": results_data.get("timestamp"),
            "dataset_label": results_data.get("dataset_label", "Built-in Benchmark"),
            "formal_metrics": formal_metrics,
        },
        "category_scores": category_averages,
        "results": results,
        "overall_score": overall_score,
        "prompt_text": results_data.get("prompt_text", ""),
    }
    pdf_bytes = generate_pdf_report(pdf_payload)
    st.download_button(
        " Download PDF Report",
        data=pdf_bytes,
        file_name=f"benchmark_report_{timestamp}.pdf",
        mime="application/pdf",
    )

    plugin_exporters = get_plugin_manager().exporters()
    if plugin_exporters:
        st.divider()
        st.caption("Plugin Exporters")
        for exporter in plugin_exporters:
            payload = exporter.export(
                results_data, results, overall_score, category_averages, formal_metrics
            )
            st.download_button(
                f"Export via {exporter.label}",
                data=payload,
                file_name=f"benchmark_{timestamp}.{exporter.extension}",
                mime=exporter.mime_type,
                key=f"plugin_export_{exporter.name}",
            )


def _render_live_log(last_run_logs: list[str]) -> None:
    st.subheader("Last Run Log")
    if last_run_logs:
        st.markdown("\n\n".join(last_run_logs))
    else:
        st.info("Run a benchmark to see live logs here")


def _render_assertion_results(assertion_results: dict) -> None:
    st.write("**Assertion Breakdown**")
    cols = st.columns(4)
    with cols[0]:
        st.metric("Assertion Score", f"{assertion_results.get('score', 0.0):.2f}")
    with cols[1]:
        st.metric("Passed", "yes" if assertion_results.get("passed") else "no")
    with cols[2]:
        st.metric("Operator", assertion_results.get("operator", "all"))
    with cols[3]:
        st.metric(
            "Checks",
            f"{assertion_results.get('passed_count', 0)}/{assertion_results.get('total', 0)}",
        )
    failed = assertion_results.get("failed") or []
    if failed:
        st.caption("Failed assertions")
        st.json(failed)


def _render_detector_results(detector_results: dict) -> None:
    st.write("**Detector Breakdown**")
    cols = st.columns(4)
    with cols[0]:
        st.metric("Detector Score", f"{detector_results.get('score', 0.0):.2f}")
    with cols[1]:
        st.metric(
            "Matched Expected",
            "yes" if detector_results.get("matched_expected_behavior") else "no",
        )
    with cols[2]:
        st.metric("Review", "yes" if detector_results.get("review_required") else "no")
    with cols[3]:
        st.metric("Detectors", len(detector_results.get("results", [])))
    for item in detector_results.get("results", []):
        with st.expander(
            f"{item.get('name', 'detector')} · {item.get('label', 'unknown')} · {item.get('score', 0.0):.2f}"
        ):
            st.json(item)
