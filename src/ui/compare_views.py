"""Streamlit compare and regression views."""

from __future__ import annotations

import json
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.datasets import load_dataset_bundle
from src.ui.analytics_views import (
    render_attack_type_regression,
    render_detector_regression,
)
from src.ui.datasets import (
    compare_dataset_packs,
    load_uploaded_dataset,
    make_dataset_bundle_like,
)
from src.ui.results import compare_runs, normalize_results_payload, summarize_run


def render_compare_versions_view(
    benchmark_history: list[dict],
    current_results: dict | None,
    dataset_tests: list[dict],
    dataset_metadata: dict,
    dataset_label: str,
    generated_pack: dict | None,
    pass_threshold: float,
    review_threshold: float,
) -> None:
    _render_dataset_pack_comparison(
        dataset_tests, dataset_metadata, dataset_label, generated_pack
    )
    st.divider()
    _render_run_history_comparison(
        benchmark_history, current_results, pass_threshold, review_threshold
    )


def _render_dataset_pack_comparison(
    dataset_tests: list[dict],
    dataset_metadata: dict,
    dataset_label: str,
    generated_pack: dict | None,
) -> None:
    st.subheader("Dataset Pack Comparison")
    pack_compare_mode = st.radio(
        "Pack Compare Mode",
        ["Current vs Upload", "Built-in vs Upload"],
        horizontal=True,
    )

    base_pack_bundle = None
    if pack_compare_mode == "Current vs Upload" and dataset_tests:
        current_compare_tests = dataset_tests
        current_compare_metadata = dataset_metadata
        if generated_pack:
            current_compare_tests = generated_pack["tests"]
            current_compare_metadata = generated_pack["metadata"]
        base_pack_bundle = make_dataset_bundle_like(
            current_compare_tests, current_compare_metadata
        )
        st.caption(
            f"Base pack: {current_compare_metadata.get('name', dataset_label)} "
            f"v{current_compare_metadata.get('version', '1.0')}"
        )
    elif pack_compare_mode == "Built-in vs Upload":
        try:
            base_pack_bundle = load_dataset_bundle("tests/safeprompt-benchmark-v2.json")
            st.caption(
                f"Base pack: {base_pack_bundle.metadata.get('name', 'Built-in Benchmark')} "
                f"v{base_pack_bundle.metadata.get('version', '1.0')}"
            )
        except Exception as exc:
            st.error(f"Failed to load built-in pack: {exc}")

    uploaded_compare_pack = st.file_uploader(
        "Upload comparison pack",
        type=["json", "jsonl", "csv"],
        key="compare_pack_upload",
    )
    if not base_pack_bundle or uploaded_compare_pack is None:
        return

    try:
        candidate_tests, _, candidate_metadata = load_uploaded_dataset(
            uploaded_compare_pack
        )
        candidate_pack_bundle = make_dataset_bundle_like(
            candidate_tests, candidate_metadata
        )
        pack_comparison = compare_dataset_packs(base_pack_bundle, candidate_pack_bundle)
        pack_metric_cols = st.columns(4)
        with pack_metric_cols[0]:
            st.metric("Base Tests", pack_comparison["base"]["num_tests"])
        with pack_metric_cols[1]:
            st.metric("Candidate Tests", pack_comparison["candidate"]["num_tests"])
        with pack_metric_cols[2]:
            st.metric("New IDs", pack_comparison["summary"]["new_ids"])
        with pack_metric_cols[3]:
            st.metric("Removed IDs", pack_comparison["summary"]["removed_ids"])

        st.dataframe(
            pd.DataFrame(pack_comparison["category_deltas"]), width="stretch"
        )
        pack_compare_export = json.dumps(pack_comparison, indent=2, ensure_ascii=False)
        st.download_button(
            " Download Pack Comparison JSON",
            pack_compare_export,
            file_name=f"pack_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            key="download_pack_comparison_json",
        )
    except Exception as exc:
        st.error(f"Failed to compare uploaded pack: {exc}")


def _render_run_history_comparison(
    benchmark_history: list[dict],
    current_results: dict | None,
    pass_threshold: float,
    review_threshold: float,
) -> None:
    all_runs = [
        normalize_results_payload(run, pass_threshold, review_threshold)
        for run in benchmark_history
    ]
    if current_results and current_results not in all_runs:
        all_runs.append(
            normalize_results_payload(current_results, pass_threshold, review_threshold)
        )

    if not all_runs:
        st.info("Run at least one benchmark to see version history.")
        return

    st.write(f"**Total runs:** {len(all_runs)}")

    comparison_data = []
    recent_runs = all_runs[-5:]
    for idx, run in enumerate(recent_runs):
        run_summary = summarize_run(run, pass_threshold, review_threshold)
        comparison_data.append(
            {
                "Run": f"#{len(all_runs) - 4 + idx if len(all_runs) > 5 else idx + 1}",
                "Provider": run.get("provider", "Unknown"),
                "Dataset": run.get("dataset_label", "Built-in Benchmark"),
                "Tests": run.get("num_tests", 0),
                "Overall Score": f"{run_summary['overall_score']:.2%}",
                "Pass Rate": f"{run_summary['pass_rate']:.2%}",
                "Review Queue": run_summary["review_count"],
                "Timestamp": run.get("timestamp", "")[:19],
            }
        )

    st.dataframe(comparison_data, width="stretch")

    if len(comparison_data) <= 1:
        st.info("Run at least 2 benchmarks to see comparison chart.")
        return

    scores = [float(row["Overall Score"].strip("%")) / 100 for row in comparison_data]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[row["Run"] for row in comparison_data],
            y=scores,
            mode="lines+markers",
            name="Average Score",
            line=dict(color="#1f77b4", width=2),
            marker=dict(size=8),
        )
    )
    fig.update_layout(
        title="Score Trend Across Runs",
        xaxis_title="Run",
        yaxis_title="Average Score",
        yaxis=dict(tickformat=".0%"),
        height=300,
    )
    st.plotly_chart(fig, width="stretch")

    base_index = max(0, len(all_runs) - 2)
    candidate_index = len(all_runs) - 1
    compare_cols = st.columns(2)
    with compare_cols[0]:
        base_label = st.selectbox(
            "Baseline run",
            options=list(range(len(all_runs))),
            index=base_index,
            format_func=lambda i: (
                f"Run #{i + 1} · {all_runs[i].get('timestamp', '')[:19]} · "
                f"{all_runs[i].get('provider', 'Unknown')} · "
                f"{all_runs[i].get('dataset_label', 'Built-in Benchmark')}"
            ),
            key="compare_base_run",
        )
    with compare_cols[1]:
        candidate_label = st.selectbox(
            "Candidate run",
            options=list(range(len(all_runs))),
            index=candidate_index,
            format_func=lambda i: (
                f"Run #{i + 1} · {all_runs[i].get('timestamp', '')[:19]} · "
                f"{all_runs[i].get('provider', 'Unknown')} · "
                f"{all_runs[i].get('dataset_label', 'Built-in Benchmark')}"
            ),
            key="compare_candidate_run",
        )

    if base_label == candidate_label:
        st.info("Choose two different runs to see a regression view.")
        return

    comparison = compare_runs(
        all_runs[base_label],
        all_runs[candidate_label],
        pass_threshold,
        review_threshold,
    )
    delta_cols = st.columns(3)
    with delta_cols[0]:
        st.metric(
            "Overall Delta",
            f"{comparison['candidate_summary']['overall_score']:.2f}",
            f"{comparison['overall_delta']:+.2f}",
        )
    with delta_cols[1]:
        st.metric(
            "Pass Rate Delta",
            f"{comparison['candidate_summary']['pass_rate'] * 100:.0f}%",
            f"{comparison['pass_rate_delta'] * 100:+.0f}pp",
        )
    with delta_cols[2]:
        st.metric(
            "Review Queue Delta",
            comparison["candidate_summary"]["review_count"],
            f"{comparison['review_delta']:+d}",
        )

    st.subheader("Category Regression View")
    category_delta_df = pd.DataFrame(comparison["category_deltas"])
    if not category_delta_df.empty:
        st.dataframe(category_delta_df, width="stretch")
        fig_delta = px.bar(
            category_delta_df,
            x="category",
            y="delta",
            color="delta",
            title="Category Score Deltas",
            color_continuous_scale="RdYlGn",
        )
        st.plotly_chart(fig_delta, width="stretch")

    detail_tabs = st.tabs(
        [
            "Worsened",
            "Improved",
            "New Review Items",
            "Detector Regression",
            "Attack Type Regression",
        ]
    )
    with detail_tabs[0]:
        if comparison["worsened_tests"]:
            st.dataframe(
                pd.DataFrame(comparison["worsened_tests"]), width="stretch"
            )
        else:
            st.success("No materially worsened tests.")
    with detail_tabs[1]:
        if comparison["improved_tests"]:
            st.dataframe(
                pd.DataFrame(comparison["improved_tests"]), width="stretch"
            )
        else:
            st.info("No materially improved tests.")
    with detail_tabs[2]:
        if comparison["new_review_items"]:
            st.dataframe(
                pd.DataFrame(comparison["new_review_items"]), width="stretch"
            )
        else:
            st.success("No new review-needed items.")
    with detail_tabs[3]:
        render_detector_regression(all_runs[base_label], all_runs[candidate_label])
    with detail_tabs[4]:
        render_attack_type_regression(all_runs[base_label], all_runs[candidate_label])

    comparison_export = json.dumps(comparison, indent=2, ensure_ascii=False)
    st.download_button(
        " Download Comparison JSON",
        comparison_export,
        file_name=f"comparison_{base_label + 1}_vs_{candidate_label + 1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
        key="download_comparison_json",
    )
