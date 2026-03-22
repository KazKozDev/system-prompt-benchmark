"""Analytics views: detector performance, attack pattern analysis, and cross-run regression."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.core.benchmark_categories import BENCHMARK_CATEGORIES
from src.core.detectors import builtin_detector_catalog

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def _collect_detector_stats(results: list[dict]) -> dict[str, dict[str, Any]]:
    """Aggregate per-detector firing statistics across *results*.

    For each detector we track:
    - evaluated   : how many tests the detector ran on
    - fired       : times it flagged something (matched=False)
    - false_positives : fired but the overall test still PASSED
    - false_negatives : did NOT fire but the test still FAILED
    - by_category / by_attack_type : same counts broken down by bucket
    """
    catalog = builtin_detector_catalog()
    stats: dict[str, dict[str, Any]] = {}

    for result in results:
        overall_label = result.get("result_label", "UNKNOWN")
        category = result.get("universal_category", "unknown")
        attack_type = result.get("category", "unknown")

        detector_payload = result.get("detector_results") or {}
        for item in detector_payload.get("results", []):
            name = item.get("name", "unknown")
            if name not in stats:
                stats[name] = {
                    "description": catalog.get(name, {}).get("description", ""),
                    "evaluated": 0,
                    "fired": 0,
                    "false_positives": 0,
                    "false_negatives": 0,
                    "by_category": defaultdict(lambda: {"evaluated": 0, "fired": 0}),
                    "by_attack_type": defaultdict(lambda: {"evaluated": 0, "fired": 0}),
                }

            fired = not item.get(
                "matched", True
            )  # matched=False → detector flagged an issue

            stats[name]["evaluated"] += 1
            stats[name]["by_category"][category]["evaluated"] += 1
            stats[name]["by_attack_type"][attack_type]["evaluated"] += 1

            if fired:
                stats[name]["fired"] += 1
                stats[name]["by_category"][category]["fired"] += 1
                stats[name]["by_attack_type"][attack_type]["fired"] += 1
                # FP: detector fired but the model actually behaved correctly (PASS)
                if overall_label == "PASS":
                    stats[name]["false_positives"] += 1
            else:
                # FN: detector silent but the model was jailbroken/leaked (FAIL)
                if overall_label == "FAIL":
                    stats[name]["false_negatives"] += 1

    return stats


def _collect_attack_type_stats(results: list[dict]) -> dict[str, dict[str, Any]]:
    """Aggregate per-attack-type (specific test category) statistics."""
    stats: dict[str, dict[str, Any]] = {}

    for result in results:
        attack_type = result.get("category", "unknown")
        label = result.get("result_label", "UNKNOWN")
        score = result.get("score", 0.0)

        if attack_type not in stats:
            stats[attack_type] = {
                "total": 0,
                "pass": 0,
                "fail": 0,
                "review": 0,
                "scores": [],
                "detectors_fired": defaultdict(int),
            }

        stats[attack_type]["total"] += 1
        stats[attack_type]["scores"].append(score)

        if label == "PASS":
            stats[attack_type]["pass"] += 1
        elif label == "FAIL":
            stats[attack_type]["fail"] += 1
        elif label in {"REVIEW", "WAIVED"}:
            stats[attack_type]["review"] += 1

        detector_payload = result.get("detector_results") or {}
        for item in detector_payload.get("results", []):
            if not item.get("matched", True):  # fired
                stats[attack_type]["detectors_fired"][item.get("name", "unknown")] += 1

    return stats


# ---------------------------------------------------------------------------
# Detector Analytics
# ---------------------------------------------------------------------------


def render_detector_analytics(results: list[dict]) -> None:
    """Render comprehensive detector performance analytics."""
    st.subheader("Detector Analytics")

    has_detectors = any(result.get("detector_results") for result in results)
    if not has_detectors:
        st.info(
            "No detector data found. Enable **Pattern Detectors** in Evaluation Settings "
            "to populate this view."
        )
        return

    stats = _collect_detector_stats(results)
    if not stats:
        st.info("No detectors were evaluated in this run.")
        return

    # --- Summary table ---
    summary_rows = _build_summary_rows(stats)
    df_summary = pd.DataFrame(summary_rows)

    st.dataframe(
        df_summary.style.format(
            {
                "Fire Rate": "{:.0%}",
                "FP Rate": "{:.0%}",
                "Precision": "{:.2f}",
                "Recall": "{:.2f}",
            },
            na_rep="–",
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "**Fire Rate** = fraction of evaluations where the detector flagged something. "
        "**FP Rate** = fraction of firings that were false positives (test still PASSED). "
        "**Precision / Recall** treat confirmed failures as the positive class."
    )

    if len(summary_rows) < 2:
        return

    st.divider()
    tab_heatmap, tab_fp, tab_fn, tab_attack = st.tabs(
        [
            "Category Heatmap",
            "False Positives",
            "Missed Detections",
            "Per-Attack Breakdown",
        ]
    )

    with tab_heatmap:
        _render_detector_category_heatmap(stats, results)

    with tab_fp:
        _render_fp_chart(summary_rows)

    with tab_fn:
        _render_fn_chart(summary_rows)

    with tab_attack:
        _render_detector_per_attack(stats, results)


def _build_summary_rows(stats: dict[str, dict]) -> list[dict]:
    rows = []
    for name, s in sorted(stats.items()):
        evaluated = s["evaluated"]
        fired = s["fired"]
        fp = s["false_positives"]
        fn = s["false_negatives"]
        tp = fired - fp  # True positives: fired AND test ultimately FAIL

        precision = tp / fired if fired > 0 else float("nan")
        recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")

        rows.append(
            {
                "Detector": name,
                "Evaluated": evaluated,
                "Fired": fired,
                "Fire Rate": fired / evaluated if evaluated else 0.0,
                "False Positives": fp,
                "FP Rate": fp / fired if fired else 0.0,
                "Missed (FN)": fn,
                "Precision": precision,
                "Recall": recall,
                "Description": s["description"],
            }
        )
    return rows


def _render_detector_category_heatmap(stats: dict, results: list[dict]) -> None:
    categories = sorted({r.get("universal_category", "unknown") for r in results})
    detectors = sorted(stats.keys())

    if not categories or not detectors:
        st.info("Not enough data for a heatmap.")
        return

    matrix: list[list[float]] = []
    annotations: list[list[str]] = []

    for det in detectors:
        row: list[float] = []
        ann_row: list[str] = []
        for cat in categories:
            cat_data = stats[det]["by_category"].get(cat, {"evaluated": 0, "fired": 0})
            ev = cat_data["evaluated"]
            fi = cat_data["fired"]
            rate = fi / ev if ev > 0 else 0.0
            row.append(round(rate, 3))
            ann_row.append(f"{fi}/{ev}" if ev > 0 else "–")
        matrix.append(row)
        annotations.append(ann_row)

    cat_labels = [BENCHMARK_CATEGORIES.get(c, {}).get("name", c) for c in categories]

    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=cat_labels,
            y=detectors,
            colorscale="RdYlGn_r",
            zmin=0.0,
            zmax=1.0,
            hoverongaps=False,
            hovertemplate="<b>%{y}</b><br>Category: %{x}<br>Fire rate: %{z:.0%}<extra></extra>",
            text=annotations,
            texttemplate="%{text}",
            textfont={"size": 9},
        )
    )
    fig.update_layout(
        title="Detector Fire Rate per Category (fired / evaluated)",
        xaxis_title="Category",
        yaxis_title="Detector",
        height=max(420, len(detectors) * 24 + 120),
        margin=dict(l=250, r=20, t=60, b=60),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Each cell shows **fired / evaluated** for that detector × category pair. "
        "Red = high fire rate. A high rate in critical categories (security, jailbreak) is expected; "
        "a high rate in low-risk categories may indicate over-firing."
    )


def _render_fp_chart(summary_rows: list[dict]) -> None:
    fp_data = [r for r in summary_rows if r["False Positives"] > 0]
    if not fp_data:
        st.success(
            "No false positives — every detector firing corresponded to a test that ultimately failed."
        )
        return

    fp_sorted = sorted(fp_data, key=lambda r: r["False Positives"], reverse=True)
    df = pd.DataFrame(fp_sorted)

    fig = px.bar(
        df,
        x="Detector",
        y="False Positives",
        color="FP Rate",
        color_continuous_scale="Reds",
        title="False Positives by Detector",
        labels={
            "False Positives": "Count (fired on tests that PASSED)",
            "FP Rate": "FP Rate",
        },
        hover_data=["Evaluated", "Fired", "FP Rate", "Description"],
    )
    fig.update_layout(
        xaxis_tickangle=-40,
        coloraxis_colorbar_tickformat=".0%",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "**False positive**: the detector flagged a response as suspicious, "
        "but the test received a PASS verdict overall. "
        "A high FP rate suggests this detector over-fires on benign content and may be adding noise."
    )


def _render_fn_chart(summary_rows: list[dict]) -> None:
    fn_data = [r for r in summary_rows if r["Missed (FN)"] > 0]
    if not fn_data:
        st.success(
            "No missed detections — every failed test was flagged by at least one detector."
        )
        return

    fn_sorted = sorted(fn_data, key=lambda r: r["Missed (FN)"], reverse=True)
    df = pd.DataFrame(fn_sorted)

    fig = px.bar(
        df,
        x="Detector",
        y="Missed (FN)",
        color="Missed (FN)",
        color_continuous_scale="Blues",
        title="Missed Detections (False Negatives) by Detector",
        labels={"Missed (FN)": "Count (test FAILED but detector was silent)"},
        hover_data=["Evaluated", "Fired", "Description"],
    )
    fig.update_layout(xaxis_tickangle=-40)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "**Missed detection (false negative)**: the test failed (model jailbroken or leaked data) "
        "but this detector produced no signal. High values reveal blind spots in the detector stack."
    )


def _render_detector_per_attack(stats: dict, results: list[dict]) -> None:
    """Show for each detector which attack types trigger it most."""
    attack_types = sorted({r.get("category", "unknown") for r in results})
    detectors = sorted(stats.keys())

    if not attack_types or not detectors:
        st.info("Not enough data.")
        return

    # Build a matrix: detectors × attack_types → fire count
    matrix: list[list[int]] = []
    for det in detectors:
        row: list[int] = []
        for at in attack_types:
            at_data = stats[det]["by_attack_type"].get(at, {"evaluated": 0, "fired": 0})
            row.append(at_data["fired"])
        matrix.append(row)

    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=attack_types,
            y=detectors,
            colorscale="YlOrRd",
            hoverongaps=False,
            hovertemplate="<b>%{y}</b><br>Attack: %{x}<br>Fires: %{z}<extra></extra>",
            texttemplate="%{z}",
            textfont={"size": 9},
        )
    )
    fig.update_layout(
        title="Detector Fire Count per Attack Type",
        xaxis_title="Attack Type",
        yaxis_title="Detector",
        height=max(420, len(detectors) * 24 + 120),
        margin=dict(l=250, r=20, t=60, b=120),
    )
    fig.update_xaxes(tickangle=-50)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Absolute fire counts per detector × attack type pair. "
        "Useful for identifying which attack classes each detector is sensitive to."
    )


# ---------------------------------------------------------------------------
# Attack Type Breakdown
# ---------------------------------------------------------------------------


def render_attack_type_breakdown(results: list[dict]) -> None:
    """Render which attack types are most effective at breaking the prompt."""
    st.subheader("Attack Type Breakdown")

    stats = _collect_attack_type_stats(results)
    if not stats:
        st.info("No results to analyse.")
        return

    rows = _build_attack_rows(stats)
    rows_sorted = sorted(rows, key=lambda r: r["Fail Rate"], reverse=True)
    df = pd.DataFrame(rows_sorted)

    st.dataframe(
        df.style.format({"Fail Rate": "{:.0%}", "Avg Score": "{:.2f}"}, na_rep="–"),
        use_container_width=True,
        hide_index=True,
    )

    # Bar chart: fail rate per attack type
    fig_bar = px.bar(
        df,
        x="Attack Type",
        y="Fail Rate",
        color="Fail Rate",
        color_continuous_scale="RdYlGn_r",
        title="Fail Rate by Attack Type (higher = more effective attack)",
        labels={"Fail Rate": "Fail Rate"},
        hover_data=["Tests", "Failed", "In Review", "Avg Score", "Top Detectors Fired"],
    )
    fig_bar.update_layout(
        xaxis_tickangle=-45,
        yaxis_tickformat=".0%",
        coloraxis_colorbar_tickformat=".0%",
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # Scatter: avg score vs fail rate (only when there's enough variety)
    if len(rows_sorted) >= 4:
        fig_scatter = px.scatter(
            df,
            x="Avg Score",
            y="Fail Rate",
            text="Attack Type",
            size="Tests",
            color="Fail Rate",
            color_continuous_scale="RdYlGn_r",
            title="Score vs Fail Rate per Attack Type",
            hover_data=["Tests", "Failed", "Top Detectors Fired"],
        )
        fig_scatter.update_traces(textposition="top center", textfont_size=10)
        fig_scatter.update_layout(
            yaxis_tickformat=".0%",
            coloraxis_colorbar_tickformat=".0%",
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    # Top-N worst attack types callout
    worst = [r for r in rows_sorted if r["Fail Rate"] >= 0.5]
    if worst:
        st.error(f"**{len(worst)} attack type(s) break the prompt in ≥50% of tests:**")
        for r in worst:
            st.markdown(
                f"- `{r['Attack Type']}` — fail rate **{r['Fail Rate']:.0%}**, "
                f"avg score {r['Avg Score']:.2f}, "
                f"top detectors: {r['Top Detectors Fired']}"
            )


def _build_attack_rows(stats: dict) -> list[dict]:
    rows = []
    for attack_type, s in stats.items():
        total = s["total"]
        fail = s["fail"]
        avg_score = float(np.mean(s["scores"])) if s["scores"] else 0.0
        top_dets = sorted(
            s["detectors_fired"].items(), key=lambda x: x[1], reverse=True
        )[:3]
        top_det_str = ", ".join(f"{d} ×{n}" for d, n in top_dets) if top_dets else "—"
        rows.append(
            {
                "Attack Type": attack_type,
                "Tests": total,
                "Passed": s["pass"],
                "Failed": fail,
                "In Review": s["review"],
                "Fail Rate": round(fail / total, 3) if total else 0.0,
                "Avg Score": round(avg_score, 3),
                "Top Detectors Fired": top_det_str,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Cross-run Regression
# ---------------------------------------------------------------------------


def render_detector_regression(base_run: dict, candidate_run: dict) -> None:
    """Show how detector firing rates changed between two benchmark runs."""
    st.subheader("Detector Regression")

    base_results = base_run.get("results", [])
    candidate_results = candidate_run.get("results", [])

    if not base_results or not candidate_results:
        st.info("Both runs must have results to compare detector behaviour.")
        return

    base_stats = _collect_detector_stats(base_results)
    candidate_stats = _collect_detector_stats(candidate_results)
    all_detectors = sorted(set(base_stats) | set(candidate_stats))

    if not all_detectors:
        st.info(
            "No detector data found in either run. Enable detectors to use this view."
        )
        return

    rows = []
    for name in all_detectors:
        bs = base_stats.get(
            name,
            {"evaluated": 0, "fired": 0, "false_positives": 0, "false_negatives": 0},
        )
        cs = candidate_stats.get(
            name,
            {"evaluated": 0, "fired": 0, "false_positives": 0, "false_negatives": 0},
        )
        base_fr = bs["fired"] / bs["evaluated"] if bs["evaluated"] else 0.0
        cand_fr = cs["fired"] / cs["evaluated"] if cs["evaluated"] else 0.0
        base_fp = bs["false_positives"] / bs["fired"] if bs["fired"] else 0.0
        cand_fp = cs["false_positives"] / cs["fired"] if cs["fired"] else 0.0
        delta_fr = cand_fr - base_fr
        delta_fp = cand_fp - base_fp
        rows.append(
            {
                "Detector": name,
                "Base Fire Rate": base_fr,
                "Cand Fire Rate": cand_fr,
                "Δ Fire Rate": delta_fr,
                "Base FP Rate": base_fp,
                "Cand FP Rate": cand_fp,
                "Δ FP Rate": delta_fp,
                "Base Missed (FN)": bs["false_negatives"],
                "Cand Missed (FN)": cs["false_negatives"],
                "Δ Missed": cs["false_negatives"] - bs["false_negatives"],
            }
        )

    rows_sorted = sorted(rows, key=lambda r: r["Δ Fire Rate"])
    df = pd.DataFrame(rows_sorted)

    st.dataframe(
        df.style.format(
            {
                "Base Fire Rate": "{:.0%}",
                "Cand Fire Rate": "{:.0%}",
                "Δ Fire Rate": "{:+.0%}",
                "Base FP Rate": "{:.0%}",
                "Cand FP Rate": "{:.0%}",
                "Δ FP Rate": "{:+.0%}",
                "Δ Missed": "{:+d}",
            },
            na_rep="–",
        ),
        use_container_width=True,
        hide_index=True,
    )

    # Callouts
    newly_firing = [
        r for r in rows if r["Base Fire Rate"] == 0.0 and r["Cand Fire Rate"] > 0.0
    ]
    newly_silent = [
        r for r in rows if r["Base Fire Rate"] > 0.0 and r["Cand Fire Rate"] == 0.0
    ]
    more_fp = [r for r in rows if r["Δ FP Rate"] > 0.15]
    more_fn = [r for r in rows if r["Δ Missed"] > 0]

    callout_cols = st.columns(2)
    with callout_cols[0]:
        if newly_firing:
            st.warning(
                f"**{len(newly_firing)} detector(s) newly firing** in candidate run:"
            )
            for r in newly_firing:
                st.markdown(f"- `{r['Detector']}` — now at {r['Cand Fire Rate']:.0%}")
        else:
            st.success("No new detectors started firing.")

        if more_fp:
            st.warning(
                f"**{len(more_fp)} detector(s) with rising FP rate** (+15 pp or more):"
            )
            for r in sorted(more_fp, key=lambda x: x["Δ FP Rate"], reverse=True):
                st.markdown(
                    f"- `{r['Detector']}` FP rate: {r['Base FP Rate']:.0%} → {r['Cand FP Rate']:.0%}"
                )

    with callout_cols[1]:
        if newly_silent:
            st.success(
                f"**{len(newly_silent)} detector(s) went silent** in candidate run:"
            )
            for r in newly_silent:
                st.markdown(
                    f"- `{r['Detector']}` — was {r['Base Fire Rate']:.0%}, now 0%"
                )
        else:
            st.info("No detectors went fully silent.")

        if more_fn:
            st.error(
                f"**{len(more_fn)} detector(s) with more missed detections** in candidate:"
            )
            for r in sorted(more_fn, key=lambda x: x["Δ Missed"], reverse=True):
                st.markdown(
                    f"- `{r['Detector']}` missed: {r['Base Missed (FN)']} → {r['Cand Missed (FN)']} "
                    f"({r['Δ Missed']:+d})"
                )

    # Delta bar chart
    if len(rows) >= 2:
        fig = px.bar(
            df.assign(**{"Detector_label": df["Detector"]}),
            x="Detector",
            y="Δ Fire Rate",
            color="Δ Fire Rate",
            color_continuous_scale="RdYlGn_r",
            title="Change in Detector Fire Rate (candidate vs baseline)",
            labels={"Δ Fire Rate": "Δ Fire Rate"},
        )
        fig.update_layout(
            xaxis_tickangle=-45,
            yaxis_tickformat="+.0%",
            coloraxis_colorbar_tickformat="+.0%",
        )
        st.plotly_chart(fig, use_container_width=True)


def render_attack_type_regression(base_run: dict, candidate_run: dict) -> None:
    """Show how attack type fail rates changed between two benchmark runs."""
    st.subheader("Attack Type Regression")

    base_stats = _collect_attack_type_stats(base_run.get("results", []))
    candidate_stats = _collect_attack_type_stats(candidate_run.get("results", []))
    all_types = sorted(set(base_stats) | set(candidate_stats))

    if not all_types:
        st.info("No attack type data found in the selected runs.")
        return

    rows = []
    for at in all_types:
        bs = base_stats.get(at, {"total": 0, "fail": 0, "scores": []})
        cs = candidate_stats.get(at, {"total": 0, "fail": 0, "scores": []})
        base_fr = bs["fail"] / bs["total"] if bs["total"] else 0.0
        cand_fr = cs["fail"] / cs["total"] if cs["total"] else 0.0
        base_avg = float(np.mean(bs["scores"])) if bs["scores"] else 0.0
        cand_avg = float(np.mean(cs["scores"])) if cs["scores"] else 0.0
        rows.append(
            {
                "Attack Type": at,
                "Base Tests": bs["total"],
                "Cand Tests": cs["total"],
                "Base Fail Rate": base_fr,
                "Cand Fail Rate": cand_fr,
                "Δ Fail Rate": cand_fr - base_fr,
                "Base Avg Score": round(base_avg, 3),
                "Cand Avg Score": round(cand_avg, 3),
                "Δ Score": round(cand_avg - base_avg, 3),
            }
        )

    rows_sorted = sorted(rows, key=lambda r: r["Δ Fail Rate"], reverse=True)
    df = pd.DataFrame(rows_sorted)

    st.dataframe(
        df.style.format(
            {
                "Base Fail Rate": "{:.0%}",
                "Cand Fail Rate": "{:.0%}",
                "Δ Fail Rate": "{:+.0%}",
                "Base Avg Score": "{:.2f}",
                "Cand Avg Score": "{:.2f}",
                "Δ Score": "{:+.2f}",
            },
            na_rep="–",
        ),
        use_container_width=True,
        hide_index=True,
    )

    # Delta bar chart
    if len(rows) >= 2:
        fig = px.bar(
            df,
            x="Attack Type",
            y="Δ Fail Rate",
            color="Δ Fail Rate",
            color_continuous_scale="RdYlGn_r",
            title="Change in Fail Rate per Attack Type (positive = attack got more effective)",
            labels={"Δ Fail Rate": "Δ Fail Rate"},
        )
        fig.update_layout(
            xaxis_tickangle=-45,
            yaxis_tickformat="+.0%",
            coloraxis_colorbar_tickformat="+.0%",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Callouts
    worsened = [r for r in rows if r["Δ Fail Rate"] > 0.1]
    improved = [r for r in rows if r["Δ Fail Rate"] < -0.1]

    reg_cols = st.columns(2)
    with reg_cols[0]:
        if worsened:
            st.error(
                f"**{len(worsened)} attack type(s) became more effective** (Δ fail rate > +10 pp):"
            )
            for r in sorted(worsened, key=lambda x: x["Δ Fail Rate"], reverse=True):
                st.markdown(
                    f"- `{r['Attack Type']}`: {r['Base Fail Rate']:.0%} → {r['Cand Fail Rate']:.0%} "
                    f"(**{r['Δ Fail Rate']:+.0%}**)"
                )
        else:
            st.success("No attack types became significantly more effective.")

    with reg_cols[1]:
        if improved:
            st.success(
                f"**{len(improved)} attack type(s) became less effective** (Δ fail rate < −10 pp):"
            )
            for r in sorted(improved, key=lambda x: x["Δ Fail Rate"]):
                st.markdown(
                    f"- `{r['Attack Type']}`: {r['Base Fail Rate']:.0%} → {r['Cand Fail Rate']:.0%} "
                    f"(**{r['Δ Fail Rate']:+.0%}**)"
                )
        else:
            st.info("No attack types improved significantly.")
