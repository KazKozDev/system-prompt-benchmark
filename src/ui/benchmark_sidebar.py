"""Sidebar rendering and app-level benchmark state helpers."""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any

import streamlit as st

from src.ui.benchmark_presets import (
    BENCHMARK_AUTO_ANALYZE_KEY,
    BENCHMARK_CUSTOM_TESTS_KEY,
    BENCHMARK_MODE_KEY,
    BENCHMARK_USE_DEGRADATION_KEY,
    BENCHMARK_USE_SEMANTIC_KEY,
    apply_benchmark_preset_to_session,
    benchmark_preset_readiness,
    benchmark_preset_summary,
    build_benchmark_preset_payload,
    delete_benchmark_preset,
    list_benchmark_presets,
    load_benchmark_preset,
    parse_benchmark_preset_json,
    save_benchmark_preset,
    serialize_benchmark_preset,
)
from src.ui.help_views import (
    PENDING_WORKSPACE_MODE_KEY,
    render_help_intro,
)
from src.ui.dataset_views import render_dataset_selector
from src.ui.evaluation_views import render_evaluation_settings
from src.ui.provider_views import (
    PROVIDER_TEST_SESSION_KEY,
    render_provider_selector,
)


def initialize_benchmark_session_state() -> None:
    defaults = {
        "results": None,
        "benchmark_history": [],
        "prompt_analysis": None,
        "prompt_analysis_signature": None,
        "live_logs": [],
        "last_run_logs": [],
        "run_metadata": {},
        "review_notes": {},
        "generated_pack": None,
        "system_prompt": None,
        BENCHMARK_MODE_KEY: "Standard (100 tests)",
        BENCHMARK_CUSTOM_TESTS_KEY: 50,
        BENCHMARK_USE_SEMANTIC_KEY: True,
        BENCHMARK_USE_DEGRADATION_KEY: True,
        BENCHMARK_AUTO_ANALYZE_KEY: True,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def render_app_header() -> None:
    hero_html = (
        "<div style='text-align:center;padding:40px 0 6px'>"
        "<div style='font-size:2.2rem;font-weight:900;color:#0f172a;"
        "letter-spacing:-0.03em;line-height:1.1'>Audit prompt security</div>"
        "</div>"
    )
    st.markdown(hero_html, unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-divider" style="margin-top:14px"></div>',
        unsafe_allow_html=True,
    )


def render_benchmark_sidebar(
    *,
    pass_threshold: float,
    review_threshold: float,
    system_prompt_caption: str,
    provider_caption: str,
    test_mode_caption: str,
    dataset_caption: str,
    judge_caption: str,
    advanced_caption: str,
    sidebar_footer_html: str,
) -> dict[str, Any]:
    with st.sidebar:
        _render_logo()

        pending_workspace_mode = st.session_state.pop(
            PENDING_WORKSPACE_MODE_KEY,
            None,
        )
        if pending_workspace_mode in {"Benchmark", "Admin", "Help"}:
            st.session_state["workspace_mode"] = pending_workspace_mode

        view_mode = st.radio(
            "Workspace",
            ["Benchmark", "Admin", "Help"],
            horizontal=True,
            key="workspace_mode",
        )

        with st.expander(" System Prompt", expanded=True):
            render_help_intro(
                "Sidebar Sections",
                key="help_link_system_prompt",
                text=system_prompt_caption,
            )
            prompt_source = st.radio(
                "Source",
                ["Use Example", "Paste Text", "Upload File"],
                label_visibility="collapsed",
                horizontal=True,
                key="prompt_source_radio",
            )
            system_prompt = _render_prompt_source(prompt_source)

        with st.expander(" LLM Provider", expanded=True):
            render_help_intro(
                "Sidebar Sections",
                key="help_link_provider",
                text=provider_caption,
            )
            provider_config, provider_capabilities = render_provider_selector()

        with st.expander(" How Many Tests?", expanded=False):
            render_help_intro(
                "Sidebar Sections",
                key="help_link_test_count",
                text=test_mode_caption,
            )
            mode = st.radio(
                "Mode",
                [
                    "Quick (10 tests)",
                    "Standard (100 tests)",
                    "Full (300 tests)",
                    "Custom",
                ],
                index=1,
                label_visibility="collapsed",
                key=BENCHMARK_MODE_KEY,
            )
            if mode == "Custom":
                num_tests = int(
                    st.slider(
                        "Number of tests",
                        10,
                        300,
                        int(st.session_state.get(
                            BENCHMARK_CUSTOM_TESTS_KEY,
                            50,
                        )),
                        key=BENCHMARK_CUSTOM_TESTS_KEY,
                    )
                )
            else:
                num_tests = {
                    "Quick (10 tests)": 10,
                    "Standard (100 tests)": 100,
                    "Full (300 tests)": 300,
                }[mode]

        with st.expander(" Benchmark Dataset", expanded=False):
            render_help_intro(
                "Sidebar Sections",
                key="help_link_dataset",
                text=dataset_caption,
            )
            dataset_state = render_dataset_selector()

        with st.expander(" Judge & Detectors", expanded=False):
            render_help_intro(
                "Sidebar Sections",
                key="help_link_judge",
                text=judge_caption,
            )
            judge_config_ui = render_evaluation_settings(
                pass_threshold,
                review_threshold,
            )

        with st.expander(" Advanced", expanded=False):
            render_help_intro(
                "Sidebar Sections",
                key="help_link_advanced",
                text=advanced_caption,
            )
            use_semantic = st.checkbox(
                "Use semantic similarity",
                value=bool(
                    st.session_state.get(BENCHMARK_USE_SEMANTIC_KEY, True)
                ),
                key=BENCHMARK_USE_SEMANTIC_KEY,
            )
            use_degradation = st.checkbox(
                "Detailed degradation metrics",
                value=bool(
                    st.session_state.get(BENCHMARK_USE_DEGRADATION_KEY, True)
                ),
                key=BENCHMARK_USE_DEGRADATION_KEY,
            )
            auto_analyze = st.checkbox(
                "Auto-analyze prompt",
                value=bool(
                    st.session_state.get(BENCHMARK_AUTO_ANALYZE_KEY, True)
                ),
                key=BENCHMARK_AUTO_ANALYZE_KEY,
            )

        _render_benchmark_presets_section(
            system_prompt=system_prompt,
            prompt_source=prompt_source,
            provider_config=provider_config,
            judge_config=judge_config_ui,
            dataset_state=dataset_state,
            mode=mode,
            num_tests=num_tests,
            use_semantic=use_semantic,
            use_degradation=use_degradation,
            auto_analyze=auto_analyze,
        )

        st.markdown(sidebar_footer_html, unsafe_allow_html=True)

    return {
        "view_mode": view_mode,
        "system_prompt": system_prompt,
        "provider_config": provider_config,
        "provider_capabilities": provider_capabilities,
        "mode": mode,
        "num_tests": num_tests,
        "judge_config_ui": judge_config_ui,
        "dataset_state": dataset_state,
        "use_semantic": use_semantic,
        "use_degradation": use_degradation,
        "auto_analyze": auto_analyze,
    }


def _render_logo() -> None:
    logo_path = Path("assets/logo.png")
    if not logo_path.exists():
        return
    with logo_path.open("rb") as handle:
        encoded_logo = base64.b64encode(handle.read()).decode()
    st.markdown(
        (
            "<div "
            "style=\"text-align:center;margin-top:-60px;margin-bottom:20px;\">"
            f"<img src=\"data:image/png;base64,{encoded_logo}\" width=\"240\">"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def _render_benchmark_presets_section(
    *,
    system_prompt: str | None,
    prompt_source: str,
    provider_config,
    judge_config,
    dataset_state: dict[str, Any],
    mode: str,
    num_tests: int,
    use_semantic: bool,
    use_degradation: bool,
    auto_analyze: bool,
) -> None:
    with st.expander(" Benchmark Presets", expanded=False):
        render_help_intro(
            "Sidebar Sections",
            key="help_link_benchmark_presets",
            text=(
                "Load, import, export, or save reusable benchmark presets here. "
                "Mandatory: No."
            ),
        )
        presets = list_benchmark_presets()
        selected_preset = st.selectbox(
            "Saved presets",
            options=["None"] + presets,
            key="benchmark_preset_select",
        )
        selected_payload = None
        if selected_preset != "None":
            selected_payload = load_benchmark_preset(selected_preset)
            st.caption("Saved Preset Preview")
            _render_benchmark_preset_preview(selected_payload)

        action_cols = st.columns(3)
        with action_cols[0]:
            if st.button(
                "Load Preset",
                key="benchmark_load_preset",
                disabled=selected_preset == "None",
            ):
                apply_benchmark_preset_to_session(selected_payload or {})
                st.rerun()
        with action_cols[1]:
            if st.button(
                "Delete Preset",
                key="benchmark_delete_preset",
                disabled=selected_preset == "None",
            ):
                delete_benchmark_preset(selected_preset)
                st.rerun()
        with action_cols[2]:
            st.download_button(
                "Export JSON",
                data=(
                    serialize_benchmark_preset(selected_payload)
                    if selected_payload
                    else ""
                ),
                file_name=f"{selected_preset}.json",
                mime="application/json",
                key="benchmark_export_preset",
                disabled=selected_payload is None,
            )

        imported_file = st.file_uploader(
            "Import preset JSON",
            type=["json"],
            key="benchmark_import_preset_file",
        )
        if imported_file is not None:
            try:
                imported_payload = parse_benchmark_preset_json(
                    imported_file.getvalue().decode("utf-8")
                )
            except ValueError as exc:
                st.error(str(exc))
            else:
                st.caption("Import Preview")
                _render_benchmark_preset_preview(imported_payload)

                is_ready, readiness_issues = benchmark_preset_readiness(
                    imported_payload
                )

                if is_ready:
                    st.success("Imported preset is run-ready.")
                else:
                    for issue in readiness_issues:
                        st.warning(issue)

                import_name = st.text_input(
                    "Imported preset name",
                    value=Path(imported_file.name).stem,
                    key="benchmark_import_preset_name",
                )
                import_cols = st.columns(2)
                with import_cols[0]:
                    if st.button(
                        "Import To Session",
                        key="benchmark_import_to_session",
                    ):
                        apply_benchmark_preset_to_session(imported_payload)
                        st.rerun()
                with import_cols[1]:
                    if st.button(
                        "Save Imported Preset",
                        key="benchmark_save_imported_preset",
                    ):
                        saved_name = save_benchmark_preset(
                            import_name,
                            imported_payload,
                        )
                        st.success(
                            "Imported benchmark preset saved as "
                            f"config/benchmarks/{saved_name}.json"
                        )

        if provider_config is None or judge_config is None:
            return

        st.divider()
        preset_name = st.text_input(
            "Preset name",
            value="benchmark-preset",
            key="benchmark_preset_name",
        )
        payload = build_benchmark_preset_payload(
            system_prompt=system_prompt or "",
            prompt_source=prompt_source,
            provider_config=provider_config,
            judge_config=judge_config,
            dataset_state=dataset_state,
            mode=mode,
            num_tests=num_tests,
            custom_num_tests=int(
                st.session_state.get(BENCHMARK_CUSTOM_TESTS_KEY, num_tests)
            ),
            use_semantic=use_semantic,
            use_degradation=use_degradation,
            auto_analyze=auto_analyze,
            generated_pack=st.session_state.get("generated_pack"),
            provider_test_result=st.session_state.get(
                PROVIDER_TEST_SESSION_KEY
            ),
        )
        export_cols = st.columns(2)
        with export_cols[0]:
            st.download_button(
                "Download Current JSON",
                data=serialize_benchmark_preset(payload),
                file_name=f"{preset_name}.json",
                mime="application/json",
                key="benchmark_download_current_preset",
            )
        with export_cols[1]:
            if st.button("Save Current Preset", key="benchmark_save_preset"):
                saved_name = save_benchmark_preset(preset_name, payload)
                st.success(
                    "Saved benchmark preset to "
                    f"config/benchmarks/{saved_name}.json"
                )


def _render_benchmark_preset_preview(payload: dict[str, Any]) -> None:
    summary = benchmark_preset_summary(payload)
    summary_cols = st.columns(2)
    with summary_cols[0]:
        st.markdown(
            "**Provider:** "
            f"{summary['provider']} / {summary['model']}"
        )
        st.markdown(f"**Mode:** {summary['mode']}")
    with summary_cols[1]:
        st.markdown(f"**Dataset:** {summary['dataset']}")
        st.markdown(f"**Tests:** {summary['test_count']}")

    if summary["prompt_preview"]:
        preview_suffix = (
            "..."
            if summary["prompt_length"] > len(summary["prompt_preview"])
            else ""
        )
        st.markdown("**Prompt Preview**")
        st.caption(f"{summary['prompt_preview']}{preview_suffix}")


def _render_prompt_source(prompt_source: str) -> str | None:
    system_prompt = None
    if prompt_source == "Upload File":
        uploaded_file = st.file_uploader(
            "Upload system prompt (.txt)",
            type=["txt"],
            label_visibility="collapsed",
            key="prompt_file_uploader",
        )
        if uploaded_file:
            system_prompt = uploaded_file.read().decode("utf-8")
            st.session_state.system_prompt = system_prompt
    elif prompt_source == "Paste Text":
        system_prompt = st.text_area(
            "Paste your system prompt here",
            height=180,
            placeholder=(
                "You are a helpful customer support agent for Acme Inc..."
            ),
            key="prompt_text_area",
            value=(
                st.session_state.system_prompt
                if st.session_state.system_prompt
                and isinstance(st.session_state.system_prompt, str)
                else ""
            ),
        )
        if system_prompt:
            st.session_state.system_prompt = system_prompt
    else:
        prompt_files: list[str] = []
        try:
            prompt_files = sorted(
                [
                    filename
                    for filename in os.listdir("prompts")
                    if filename.lower().endswith(".txt")
                ]
            )
        except FileNotFoundError:
            st.warning("prompts/ folder not found")

        if prompt_files:

            def _pretty_name(filename: str) -> str:
                stem = Path(filename).stem.replace("_", " ").replace("-", " ")
                name = stem.title()
                name = name.replace("Hr ", "HR ")
                name = name.replace("Ai ", "AI ")
                name = name.replace("Api ", "API ")
                return name

            selected_file = st.selectbox(
                "Choose example",
                prompt_files,
                format_func=_pretty_name,
            )
            prompt_path = Path("prompts") / selected_file
            try:
                with prompt_path.open("r", encoding="utf-8") as handle:
                    system_prompt = handle.read()
                st.session_state.system_prompt = system_prompt
            except Exception:
                st.warning(f"Failed to load {prompt_path}")
        else:
            st.warning("No example prompts found")

    if not system_prompt and st.session_state.system_prompt:
        system_prompt = st.session_state.system_prompt

    if system_prompt:
        with st.expander("Preview prompt", expanded=False):
            preview_suffix = "…" if len(system_prompt) > 600 else ""
            preview_prompt = system_prompt[:600] + preview_suffix
            st.code(preview_prompt, language=None)
    return system_prompt
