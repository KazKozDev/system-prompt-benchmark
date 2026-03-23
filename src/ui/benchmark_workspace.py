"""Main benchmark workspace rendering and execution flow."""

from __future__ import annotations

import time
from collections import defaultdict
from datetime import datetime
import hashlib
import json
from textwrap import shorten
from typing import Any

import streamlit as st

from src.config import ProviderConfig
from src.core.evaluation import classify_result, evaluate_response
from src.core.run_universal_benchmark import UniversalBenchmark
from src.providers.run_benchmark import create_provider
from src.ui.compare_views import render_compare_versions_view
from src.ui.dataset_views import render_build_pack_view
from src.ui.help_views import render_help_intro
from src.ui.results import ensure_result_defaults, normalize_results_payload
from src.utils.prompt_analyzer import analyze_system_prompt


def render_benchmark_workspace(
    *,
    system_prompt: str,
    provider_config,
    provider_capabilities: dict[str, Any],
    dataset_state: dict[str, Any],
    mode: str,
    num_tests: int,
    auto_analyze: bool,
    judge_config_ui,
    pass_threshold: float,
    review_threshold: float,
    render_analyze_prompt_help,
    render_build_pack_help,
    render_compare_versions_help,
) -> None:
    dataset_tests = dataset_state["dataset_tests"]
    dataset_issues = dataset_state["dataset_issues"]
    dataset_label = dataset_state["dataset_label"]
    dataset_metadata = dataset_state["dataset_metadata"]

    if not dataset_tests:
        st.warning(
            " No benchmark dataset loaded — select one in the sidebar under "
            "**Dataset**."
        )
    if dataset_issues:
        st.error(
            " The loaded dataset has validation issues. Fix them before "
            "running."
        )
    if provider_capabilities.get("validation_errors"):
        st.error(
            " Provider configuration error — check the **Provider** section "
            "in the sidebar."
        )

    _sync_prompt_analysis_state(system_prompt, provider_config)

    action_tabs = st.tabs(
        [
            "Run Benchmark",
            "Analyze Prompt",
            "Build Pack",
            "Compare Versions",
        ]
    )

    with action_tabs[0]:
        render_help_intro(
            "Benchmark Workspace",
            key="help_link_run_benchmark_tab",
            text="Ready? Execute the benchmark with your current settings.",
        )
        run_benchmark = st.button(
            "Start Benchmark",
            disabled=bool(
                not dataset_tests
                or dataset_issues
                or provider_capabilities.get("validation_errors")
            ),
        )

    with action_tabs[1]:
        render_help_intro(
            "Benchmark Workspace",
            key="help_link_analyze_prompt_tab",
            text="Review the prompt before running the benchmark.",
        )
        with st.expander("About Analyze Prompt", expanded=False):
            render_analyze_prompt_help()

        if st.button("Run Analysis"):
            with st.spinner("Analyzing system prompt..."):
                st.session_state.prompt_analysis = analyze_system_prompt(
                    system_prompt,
                    provider_config=provider_config,
                    use_llm=True,
                )
                st.session_state.prompt_analysis_signature = (
                    _build_prompt_analysis_signature(system_prompt, provider_config)
                )
            st.success("Analysis complete!")

        if st.session_state.prompt_analysis:
            _render_prompt_analysis(st.session_state.prompt_analysis)

    with action_tabs[2]:
        render_help_intro(
            "Benchmark Workspace",
            key="help_link_build_pack_tab",
            text="Create a focused custom dataset pack from the current data.",
        )
        with st.expander("About Build Pack", expanded=False):
            render_build_pack_help()

        generated_pack = render_build_pack_view(
            dataset_tests,
            dataset_metadata,
            dataset_label,
        )
        if generated_pack:
            st.session_state.generated_pack = generated_pack
            st.success(
                " Custom pack ready: "
                f"{len(generated_pack['tests'])} tests. "
                "It will be used on the next run."
            )

    with action_tabs[3]:
        render_help_intro(
            "Benchmark Workspace",
            key="help_link_compare_versions_tab",
            text="Compare the current run with previous benchmark versions.",
        )
        with st.expander("About Compare Versions", expanded=False):
            render_compare_versions_help()

        render_compare_versions_view(
            st.session_state.benchmark_history,
            st.session_state.results,
            dataset_tests,
            dataset_metadata,
            dataset_label,
            st.session_state.generated_pack,
            pass_threshold,
            review_threshold,
        )

    if run_benchmark:
        if auto_analyze:
            _run_auto_prompt_analysis(system_prompt, provider_config)
        _execute_benchmark_run(
            system_prompt=system_prompt,
            provider_config=provider_config,
            provider_capabilities=provider_capabilities,
            dataset_state=dataset_state,
            mode=mode,
            num_tests=num_tests,
            judge_config_ui=judge_config_ui,
            pass_threshold=pass_threshold,
            review_threshold=review_threshold,
        )


def _render_prompt_analysis(analysis: dict[str, Any]) -> None:
    st.subheader(" Prompt Analysis")
    method = analysis.get("analysis_method", "unknown")
    provider_label = analysis.get("analysis_provider")
    fallback_reason = analysis.get("analysis_fallback_reason")

    if method == "llm":
        source_label = provider_label or "selected provider"
        st.caption(f"Analyzed with live provider: {source_label}")
    else:
        st.warning(
            "Live provider analysis was unavailable, so this view is showing "
            "heuristic fallback results."
        )
        if fallback_reason:
            st.caption(f"Fallback reason: {fallback_reason}")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.caption("Role")
        st.write(f"**{analysis.get('role', 'Unknown')}**")
        st.caption("Domain")
        st.write(f"**{analysis.get('domain', 'Unknown')}**")

    with col2:
        capabilities = analysis.get("capabilities", [])
        st.write("**Capabilities:**")
        for cap in capabilities[:3]:
            st.write(f"• {cap}")

    with col3:
        boundaries = analysis.get("boundaries", [])
        st.write("**Boundaries:**")
        for bound in boundaries[:3]:
            st.write(f"• {bound}")

    topics = analysis.get("core_topics", [])
    if topics:
        st.write(f"**Core Topics:** {', '.join(topics[:5])}")

    constraints = analysis.get("constraints") or {}
    constraint_items = [
        f"{label}: {value}"
        for label, value in (
            ("Format", constraints.get("format")),
            ("Length", constraints.get("length")),
            ("Tone", constraints.get("tone")),
            ("Language", constraints.get("language")),
        )
        if value
    ]
    if constraint_items:
        st.write("**Constraints:**")
        for item in constraint_items:
            st.write(f"• {item}")


def _build_prompt_analysis_signature(
    system_prompt: str,
    provider_config: ProviderConfig,
) -> str:
    payload = {
        "system_prompt": system_prompt,
        "provider": {
            "name": provider_config.name,
            "model": provider_config.model,
            "api_key": provider_config.api_key,
            "api_key_env": provider_config.api_key_env,
            "base_url": provider_config.base_url,
            "api_version": provider_config.api_version,
            "aws_region": provider_config.aws_region,
            "project_id": provider_config.project_id,
            "location": provider_config.location,
        },
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _sync_prompt_analysis_state(
    system_prompt: str,
    provider_config: ProviderConfig,
) -> None:
    current_signature = _build_prompt_analysis_signature(
        system_prompt,
        provider_config,
    )
    stored_signature = st.session_state.get("prompt_analysis_signature")
    if stored_signature != current_signature:
        st.session_state.prompt_analysis = None
        st.session_state.prompt_analysis_signature = None


def _run_auto_prompt_analysis(
    system_prompt: str,
    provider_config: ProviderConfig,
) -> None:
    current_signature = _build_prompt_analysis_signature(
        system_prompt,
        provider_config,
    )
    if st.session_state.get("prompt_analysis_signature") == current_signature:
        return

    with st.spinner("Auto-analyzing system prompt..."):
        st.session_state.prompt_analysis = analyze_system_prompt(
            system_prompt,
            provider_config=provider_config,
            use_llm=True,
        )
        st.session_state.prompt_analysis_signature = current_signature


def _execute_benchmark_run(
    *,
    system_prompt: str,
    provider_config,
    provider_capabilities: dict[str, Any],
    dataset_state: dict[str, Any],
    mode: str,
    num_tests: int,
    judge_config_ui,
    pass_threshold: float,
    review_threshold: float,
) -> None:
    dataset_tests = dataset_state["dataset_tests"]
    dataset_issues = dataset_state["dataset_issues"]
    dataset_label = dataset_state["dataset_label"]
    dataset_metadata = dataset_state["dataset_metadata"]
    dataset_source = dataset_state["dataset_source"]

    try:
        active_dataset_tests = dataset_tests
        active_dataset_metadata = dataset_metadata
        active_dataset_label = dataset_label
        if st.session_state.generated_pack:
            active_dataset_tests = st.session_state.generated_pack["tests"]
            active_dataset_metadata = st.session_state.generated_pack[
                "metadata"
            ]
            active_dataset_label = st.session_state.generated_pack["label"]

        if not active_dataset_tests:
            st.error("No benchmark dataset loaded.")
            st.stop()
        if dataset_issues:
            st.error(
                "Fix dataset validation issues before running the benchmark."
            )
            st.stop()
        if provider_capabilities.get("validation_errors"):
            st.error(
                "Fix provider configuration errors before running the "
                "benchmark."
            )
            st.stop()

        provider = create_provider(provider_config)

        temp_prompt_file = "/tmp/temp_system_prompt.txt"
        with open(temp_prompt_file, "w", encoding="utf-8") as handle:
            handle.write(system_prompt)

        benchmark = UniversalBenchmark(
            temp_prompt_file,
            provider,
            judge_config=judge_config_ui,
        )

        effective_num_tests = min(num_tests, len(active_dataset_tests))
        st.markdown(
            f"Running {effective_num_tests} tests from "
            f"**{active_dataset_label}**..."
        )

        progress_bar = st.progress(0)
        status_text = st.empty()
        with st.container():
            st.markdown("**Live Execution Log**")
            log_stream = st.empty()
        st.session_state.live_logs = []

        all_tests = _select_tests(active_dataset_tests, effective_num_tests)
        results = []
        for idx, test in enumerate(all_tests):
            status_text.text(
                f"Running test {idx + 1}/{len(all_tests)}: "
                f"{test.get('category', 'unknown')}"
            )

            try:
                response, tokens, latency, execution_meta = (
                    benchmark._execute_test_case(test)
                )
                evaluation = evaluate_response(
                    system_prompt,
                    test,
                    response,
                    judge_config_ui,
                )
                result = {
                    "test_id": test["id"],
                    "category": test.get(
                        "original_category",
                        test["category"],
                    ),
                    "universal_category": test.get(
                        "universal_category",
                        "unknown",
                    ),
                    "input": test["input"],
                    "response": response,
                    "score": evaluation.score,
                    "score_method": evaluation.score_method,
                    "judge_scores": evaluation.judge_scores,
                    "assertion_results": evaluation.assertion_results,
                    "detector_results": evaluation.detector_results,
                    "review_required": evaluation.review_required,
                    "matched_expected_behavior": (
                        evaluation.matched_expected_behavior
                    ),
                    "execution_meta": execution_meta,
                    "review_status": "REVIEW"
                    if evaluation.review_required
                    else None,
                    "review_note": "",
                    "result_label": classify_result(
                        evaluation.score,
                        pass_threshold=pass_threshold,
                        review_threshold=review_threshold,
                    ),
                    "status": "ok",
                    "should_refuse": bool(test.get("should_refuse")),
                    "expected_behavior": test.get(
                        "expected_behavior",
                        "refuse" if test.get("should_refuse") else "answer",
                    ),
                    "tokens": tokens,
                    "latency": latency,
                    "success_criteria": test.get("success_criteria"),
                }
                results.append(result)
                _append_log_entry(
                    log_stream,
                    idx,
                    all_tests,
                    test,
                    result,
                    tokens,
                    latency,
                    response,
                )
            except Exception as exc:
                error_result = ensure_result_defaults(
                    {
                        "test_id": test["id"],
                        "category": test.get(
                            "original_category",
                            test["category"],
                        ),
                        "universal_category": test.get(
                            "universal_category",
                            "unknown",
                        ),
                        "input": test["input"],
                        "response": f"ERROR: {exc}",
                        "score": 0.0,
                        "score_method": "error",
                        "judge_scores": None,
                        "assertion_results": None,
                        "detector_results": None,
                        "review_required": True,
                        "matched_expected_behavior": False,
                        "execution_meta": {"mode": "error"},
                        "review_status": "REVIEW",
                        "review_note": "",
                        "status": "error",
                        "tokens": 0,
                        "latency": 0.0,
                        "should_refuse": bool(test.get("should_refuse")),
                        "expected_behavior": test.get(
                            "expected_behavior",
                            (
                                "refuse"
                                if test.get("should_refuse")
                                else "answer"
                            ),
                        ),
                        "success_criteria": test.get("success_criteria"),
                    },
                    pass_threshold,
                    review_threshold,
                )
                results.append(error_result)
                st.session_state.live_logs.insert(
                    0,
                    f"**Test {idx + 1}/{len(all_tests)}** · "
                    f"`{test.get('category', 'unknown')}` · ERROR `{exc}`",
                )
                st.session_state.live_logs = st.session_state.live_logs[:25]
                log_stream.markdown("\n\n".join(st.session_state.live_logs))
                st.error(f"Error on test {test['id']}: {exc}")

            progress_bar.progress((idx + 1) / len(all_tests))
            time.sleep(0.1)

        status_text.text("Benchmark complete!")
        st.session_state.results = normalize_results_payload(
            {
                "timestamp": datetime.now().isoformat(),
                "provider": provider.get_model_name(),
                "provider_capabilities": provider.get_capabilities(),
                "num_tests": len(results),
                "dataset_label": active_dataset_label,
                "dataset_metadata": active_dataset_metadata,
                "dataset_source": dataset_source,
                "results": results,
                "prompt_text": system_prompt,
            },
            pass_threshold,
            review_threshold,
        )
        st.session_state.last_run_logs = list(st.session_state.live_logs)
        st.session_state.run_metadata = {
            "mode": mode,
            "num_tests": len(results),
            "provider": provider.get_model_name(),
            "model": provider_config.model or provider.get_model_name(),
            "dataset_label": active_dataset_label,
            "dataset_version": active_dataset_metadata.get("version", "1.0"),
            "provider_transport": provider_capabilities.get(
                "transport",
                "unknown",
            ),
            "provider_capabilities": provider.get_capabilities(),
            "provider_test_result": st.session_state.get(
                "provider_test_result"
            ),
        }
        st.session_state.benchmark_history.append(st.session_state.results)
        st.success(f"Completed {len(results)} tests!")
        st.rerun()
    except Exception as exc:
        st.error(f"Error: {exc}")
        import traceback

        st.code(traceback.format_exc())


def _select_tests(
    active_dataset_tests: list[dict],
    effective_num_tests: int,
) -> list[dict]:
    all_tests = list(active_dataset_tests)
    if effective_num_tests >= len(all_tests):
        return all_tests

    by_category: dict[str, list[dict]] = defaultdict(list)
    for test in all_tests:
        by_category[test.get("universal_category", "unknown")].append(test)

    selected_tests: list[dict] = []
    per_category = max(1, effective_num_tests // max(1, len(by_category)))
    for tests in by_category.values():
        selected_tests.extend(tests[:per_category])

    remaining = effective_num_tests - len(selected_tests)
    if remaining > 0:
        for test in all_tests:
            if test not in selected_tests:
                selected_tests.append(test)
                if len(selected_tests) >= effective_num_tests:
                    break
    return selected_tests[:effective_num_tests]


def _append_log_entry(
    log_stream,
    idx: int,
    all_tests: list[dict],
    test: dict[str, Any],
    result: dict[str, Any],
    tokens: int,
    latency: float,
    response: str,
) -> None:
    input_preview = shorten(
        test["input"].replace("\n", " "),
        width=140,
        placeholder="…",
    )
    response_preview = shorten(
        response.replace("\n", " "),
        width=160,
        placeholder="…",
    )
    log_entry = (
        f"**Test {idx + 1}/{len(all_tests)}** · "
        f"`{test.get('category', 'unknown')}`\n"
        f"- Score **{result['score']:.2f}** · "
        f"Label **{result['result_label']}**\n"
        f"- Input: {input_preview}\n"
        f"- Response: {response_preview}\n"
        f"- Tokens: {tokens} | Latency: {latency:.2f}s\n"
        f"- Judge: {result['score_method']}"
    )
    st.session_state.live_logs.insert(0, log_entry)
    st.session_state.live_logs = st.session_state.live_logs[:25]
    log_stream.markdown("\n\n".join(st.session_state.live_logs))
