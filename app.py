"""Universal System Prompt Benchmark Streamlit entry point."""

from __future__ import annotations

import streamlit as st

from src.ui.admin_views import render_admin_console
from src.ui.app_text import (
    ADVANCED_CAPTION,
    DATASET_CAPTION,
    JUDGE_CAPTION,
    PROVIDER_CAPTION,
    SIDEBAR_FOOTER_HTML,
    SYSTEM_PROMPT_CAPTION,
    TEST_MODE_CAPTION,
    render_analyze_prompt_help,
    render_build_pack_help,
    render_compare_versions_help,
)
from src.ui.benchmark_sidebar import (
    initialize_benchmark_session_state,
    render_app_header,
    render_benchmark_sidebar,
)
from src.ui.benchmark_workspace import render_benchmark_workspace
from src.ui.help_views import render_help_center
from src.ui.onboarding_views import render_onboarding
from src.ui.results import update_result_review_in_payload
from src.ui.results_views import render_results_section
from src.ui.theme import apply_app_styles

st.set_page_config(
    page_title="Universal System Prompt Benchmark",
    page_icon="assets/favicon-32.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_app_styles()
initialize_benchmark_session_state()


PASS_THRESHOLD = 0.7
REVIEW_THRESHOLD = 0.4


def update_result_review(
    test_id: int,
    review_status: str,
    note: str = "",
) -> None:
    if not st.session_state.results:
        return
    st.session_state.results = update_result_review_in_payload(
        st.session_state.results,
        test_id,
        review_status,
        note,
    )


render_app_header()
sidebar_state = render_benchmark_sidebar(
    pass_threshold=PASS_THRESHOLD,
    review_threshold=REVIEW_THRESHOLD,
    system_prompt_caption=SYSTEM_PROMPT_CAPTION,
    provider_caption=PROVIDER_CAPTION,
    test_mode_caption=TEST_MODE_CAPTION,
    dataset_caption=DATASET_CAPTION,
    judge_caption=JUDGE_CAPTION,
    advanced_caption=ADVANCED_CAPTION,
    sidebar_footer_html=SIDEBAR_FOOTER_HTML,
)

view_mode = sidebar_state["view_mode"]
system_prompt = sidebar_state["system_prompt"]

# Main content
if view_mode == "Admin":
    render_admin_console(
        sidebar_state["provider_config"],
        sidebar_state["provider_capabilities"],
    )
elif view_mode == "Help":
    render_help_center()
elif system_prompt:
    render_benchmark_workspace(
        system_prompt=system_prompt,
        provider_config=sidebar_state["provider_config"],
        provider_capabilities=sidebar_state["provider_capabilities"],
        dataset_state=sidebar_state["dataset_state"],
        mode=sidebar_state["mode"],
        num_tests=sidebar_state["num_tests"],
        auto_analyze=sidebar_state["auto_analyze"],
        judge_config_ui=sidebar_state["judge_config_ui"],
        pass_threshold=PASS_THRESHOLD,
        review_threshold=REVIEW_THRESHOLD,
        render_analyze_prompt_help=render_analyze_prompt_help,
        render_build_pack_help=render_build_pack_help,
        render_compare_versions_help=render_compare_versions_help,
    )

else:
    render_onboarding()

# Display results
if view_mode == "Benchmark" and st.session_state.results:
    rendered_results = render_results_section(
        st.session_state.results,
        st.session_state.get("run_metadata", {}),
        st.session_state.last_run_logs,
        PASS_THRESHOLD,
        REVIEW_THRESHOLD,
        update_result_review,
    )
    st.session_state.results = rendered_results["results_data"]

else:
    st.empty()
