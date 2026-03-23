import streamlit as st


SYSTEM_PROMPT_CAPTION = (
    "The core instructions your LLM must obey.\n"
    "Mandatory: Yes. You must provide a prompt to start testing."
)

PROVIDER_CAPTION = (
    "Configure the underlying model you want to test.\n"
    "Mandatory: Yes. You must select a model and enter an API key "
    "(unless using local Ollama). When you enter the provider API key, "
    "the system dynamically loads the current model list directly from "
    "that provider."
)

TEST_MODE_CAPTION = (
    "Choose the duration and coverage of the benchmark.\n"
    "Mandatory: No, default is Standard (100). Use Quick for fast "
    "iteration, and Full for final validation."
)

DATASET_CAPTION = (
    "Select the collection of attack prompts.\n"
    "Mandatory: No. The default pack handles most cases. Use custom "
    "packs to test specific domain vulnerabilities."
)

JUDGE_CAPTION = (
    "Tweak how strictly responses are evaluated.\n"
    "Mandatory: No. Adjust this only if the automated judge is being "
    "too harsh or too lenient for your use case."
)

ADVANCED_CAPTION = (
    "Toggle extra analytical tools (semantic similarity, etc.).\n"
    "Mandatory: No. Enable these for deeper insights into response "
    "degradation."
)

SIDEBAR_FOOTER_HTML = """
<style>
[data-testid="stSidebarUserContent"] {
    display: flex;
    flex-direction: column;
    height: 100%;
}
[data-testid="stSidebarUserContent"] > div {
    display: flex;
    flex-direction: column;
    flex-grow: 1;
}
.element-container:has(.sidebar-footer) {
    margin-top: auto;
}
.sidebar-footer {
    padding: 15px 0;
    text-align: center;
    color: #64748b !important;
    font-size: 0.7rem !important;
}
.sidebar-footer a {
    color: #64748b !important;
    text-decoration: none !important;
    font-size: 0.7rem !important;
}
</style>
<p class="sidebar-footer">
    Built by
    <a href="https://github.com/KazKozDev" target="_blank">@KazKozDev</a>
</p>
"""

ANALYZE_PROMPT_STEPS = """
1. Make sure you've loaded a system prompt in the sidebar
2. Click **Run Analysis** button below
3. Review the extracted components:
   - **Role & Domain**: Does it correctly identify what your AI is?
   - **Capabilities**: Are all intended features listed?
   - **Boundaries**: Are all restrictions captured?
   - **Core Topics**: Are domain-specific concepts recognized?
4. If important constraints are missing, rewrite your prompt more explicitly
"""

BUILD_PACK_STEPS = """
1. Expand **Dataset Row Editor** to filter or edit rows by category,
    ID range, or search
2. Select **Transforms** to apply variations:
   - `base64_encode`: Encode attack in Base64
   - `rot13`: Apply ROT13 cipher
   - `multilingual`: Translate to multiple languages
3. Expand **Success Criteria** to add assertions:
   - Choose scope (All Tests / By Category / By Test IDs)
   - Set operator (`all` = every assertion must pass, `any` = at least one)
    - Add assertions like `contains`, `not_contains`, `regex` on
      response or input fields
4. Click **Use Generated Pack For This Run** to use it in next benchmark
5. Or **Download/Save** the pack for reuse
"""

COMPARE_PACK_STEPS = """
1. Choose mode: **Current vs Upload** or **Built-in vs Upload**
2. Upload a comparison pack (JSON/JSONL/CSV)
3. Review metrics: Base/Candidate test counts, New/Removed IDs, category deltas
"""

COMPARE_HISTORY_STEPS = """
1. View recent runs table with scores, providers, datasets
2. Check **Score Trend** chart for overall progress
3. Select **Baseline run** and **Candidate run** from dropdowns
4. Review deltas: Overall Score, Pass Rate, Review Queue changes
5. Explore tabs:
   - **Worsened**: Tests that scored lower
   - **Improved**: Tests that scored higher
   - **New Review Items**: Tests newly flagged for review
   - **Detector/Attack Type Regression**: Detailed breakdowns
6. Download comparison JSON for records
"""


def render_analyze_prompt_help() -> None:
    st.markdown("**Mandatory:** No.")
    st.markdown(
        "**What it does:** Uses AI to automatically read your system prompt "
        "and extract its structure: Role (e.g., \"customer support agent\"), "
        "Domain (e.g., \"e-commerce\"), Capabilities (what it should do), "
        "Boundaries (what it must not do), and Constraints (tone, format, "
        "language rules)."
    )
    st.markdown(
        "**When to use:** Before running the benchmark, to sanity-check if "
        "your instructions are clear and complete. If the AI misses key "
        "constraints or boundaries in the analysis, your prompt might be too "
        "vague for reliable behavior."
    )
    st.markdown("**How to use:**")
    st.markdown(ANALYZE_PROMPT_STEPS)


def render_build_pack_help() -> None:
    st.markdown("**Mandatory:** No.")
    st.markdown(
        "**What it does:** Build custom test packs from the loaded dataset. "
        "You can filter tests by category, apply transforms (Base64 encoding, "
        "ROT13, multilingual variants, etc.) to generate new attack "
        "variations, and attach assertion-based success criteria."
    )
    st.markdown(
        "**When to use:** (1) Test only specific attack families. (2) "
        "Generate encoded or obfuscated variants of existing attacks. (3) "
        "Enforce strict pass/fail rules beyond the AI judge's scoring."
    )
    st.markdown("**How to use:**")
    st.markdown(BUILD_PACK_STEPS)


def render_compare_versions_help() -> None:
    st.markdown("**Mandatory:** No.")
    st.markdown(
        "**What it does:** Compare multiple benchmark runs side-by-side. "
        "Shows overall score trends across runs, category-level performance "
        "deltas, and test-by-test breakdowns of improved vs. worsened "
        "results. Also compares dataset packs to see which tests were added "
        "or removed."
    )
    st.markdown(
        "**When to use:** After modifying your system prompt and re-running "
        "the benchmark, to measure whether your changes improved security or "
        "introduced new vulnerabilities."
    )
    st.markdown("**How to use:**")
    st.markdown("**Dataset Pack Comparison:**")
    st.markdown(COMPARE_PACK_STEPS)
    st.markdown("**Run History Comparison:**")
    st.markdown(COMPARE_HISTORY_STEPS)
