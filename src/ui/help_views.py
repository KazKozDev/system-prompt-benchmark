"""Dedicated help and usage guidance for the Streamlit workspace."""

from __future__ import annotations

import streamlit as st


HELP_TOPIC_OPTIONS = [
    "Start Here",
    "Sidebar Sections",
    "Benchmark Workspace",
    "Results",
    "Admin",
]

PENDING_WORKSPACE_MODE_KEY = "pending_workspace_mode"
PENDING_HELP_TOPIC_KEY = "pending_help_topic"


def open_help_topic(topic: str) -> None:
    """Switch the workspace into Help mode and focus a specific topic."""
    st.session_state[PENDING_WORKSPACE_MODE_KEY] = "Help"
    st.session_state[PENDING_HELP_TOPIC_KEY] = (
        topic if topic in HELP_TOPIC_OPTIONS else "Start Here"
    )
    st.rerun()


def render_help_link(
    topic: str,
    *,
    key: str,
    label: str | None = None,
) -> None:
    """Render a compact contextual help button."""
    button_label = label or f"Help: {topic}"
    if st.button(button_label, key=key, width="stretch"):
        open_help_topic(topic)


def render_help_intro(
    topic: str,
    *,
    key: str,
    text: str,
) -> None:
    """Render a section intro with a compact help trigger."""
    text_col, icon_col = st.columns([28, 1], gap="small")
    with text_col:
        st.caption(text)
    with icon_col:
        st.markdown(
            (
                "<style>"
                "div[data-testid='stButton'] > button[kind='tertiary'] {"
                "min-height:1rem;padding:0 0.05rem;font-size:0.72rem;"
                "line-height:1;border:none;background:transparent;"
                "box-shadow:none;transform:translate(-0.1rem,-0.18rem);"
                "}"
                "</style>"
            ),
            unsafe_allow_html=True,
        )
        if st.button(
            "?",
            key=key,
            help=f"Open help: {topic}",
            type="tertiary",
        ):
            open_help_topic(topic)
    st.markdown(
        "<div style='margin-bottom:0.18rem'></div>",
        unsafe_allow_html=True,
    )


def render_help_center() -> None:
    """Render the top-level help center for the benchmark UI."""
    pending_topic = st.session_state.pop(PENDING_HELP_TOPIC_KEY, None)
    if pending_topic in HELP_TOPIC_OPTIONS:
        st.session_state["help_topic"] = pending_topic

    st.header("Help Center")
    st.caption(
        "How each section works, when to use it, and how to run the benchmark "
        "without guessing."
    )

    selected_topic = st.radio(
        "Help Topic",
        HELP_TOPIC_OPTIONS,
        horizontal=True,
        key="help_topic",
    )

    if selected_topic == "Start Here":
        _render_start_here()
    elif selected_topic == "Sidebar Sections":
        _render_sidebar_help()
    elif selected_topic == "Benchmark Workspace":
        _render_benchmark_workspace_help()
    elif selected_topic == "Results":
        _render_results_help()
    else:
        _render_admin_help()


def _render_start_here() -> None:
    st.subheader("How To Use The Benchmark")
    st.markdown(
        "### In simple words\n"
        "This app checks whether your system prompt really holds up under bad "
        "or tricky inputs.\n\n"
        "It does not just ask the model normal questions. It deliberately "
        "tries to break the rules: bypass restrictions, confuse roles, inject "
        "new instructions, or make the model leak something it should not say."
    )

    st.info(
        "If the score is low, it usually means the prompt is too vague, too "
        "permissive, or not explicit enough about boundaries and refusals."
    )

    st.markdown("### Step-By-Step")
    st.markdown(
        "1. Add your system prompt in the sidebar.\n"
        "2. Select the provider and model you want to test.\n"
        "3. Choose how many tests to run.\n"
        "4. Choose which attack dataset to use.\n"
        "5. Leave judge settings as default unless you have a clear reason to "
        "change them.\n"
        "6. Open **Run Benchmark** and click **Start Benchmark**.\n"
        "7. Read the score first, then open failed examples, then compare "
        "with older runs if needed."
    )

    metric_cols = st.columns(3)
    with metric_cols[0]:
        st.metric("Quick Mode", "10 tests")
    with metric_cols[1]:
        st.metric("Standard Mode", "100 tests")
    with metric_cols[2]:
        st.metric("Full Mode", "300 tests")

    st.markdown("### Which Mode To Use In Real Work")
    st.markdown(
        "- **Quick**: useful when a product or prompt team changed "
        "wording and "
        "wants a fast sanity check before spending tokens on a larger run.\n"
        "- **Standard**: useful for the normal day-to-day workflow, for "
        "example when a support team updated escalation rules and wants a "
        "meaningful score.\n"
        "- **Full**: useful before release, audit, or security sign-off, for "
        "example when a banking or enterprise copilot goes to production."
    )

    st.markdown("### What A Good Workflow Looks Like")
    st.markdown(
        "- Use **Quick** while iterating on your prompt.\n"
        "- Use **Standard** when you want a meaningful score.\n"
        "- Use **Full** before shipping or after major prompt changes.\n"
        "- Save the setup as a **Benchmark Preset** if you will rerun it "
        "later."
    )

    st.markdown("### What Each Important Word Means")
    st.markdown(
        "- **System prompt**: the main instruction that tells the model how "
        "to behave.\n"
        "- **Provider**: the model backend, for example OpenAI, Anthropic, "
        "Ollama, and so on.\n"
        "- **Dataset**: the list of test attacks or prompts the model will "
        "face.\n"
        "- **Judge**: the scoring logic that decides whether the answer "
        "passed or failed.\n"
        "- **Detector**: extra checks that flag risky output patterns.\n"
        "- **Preset**: a saved benchmark setup so you can rerun the same "
        "scenario later."
    )

    st.markdown("### If You Are New")
    st.markdown(
        "Start with this simple path: paste prompt, choose provider, keep "
        "Standard mode, use the default dataset, run the benchmark, then open "
        "failed examples and fix the prompt based on what broke."
    )


def _render_sidebar_help() -> None:
    st.subheader("Sidebar Sections")
    _render_help_card(
        "Workspace",
        "This is the main mode switch for the whole app.",
        "Choose **Benchmark** for normal prompt testing, **Admin** for system "
        "operations, and **Help** when you need explanations.",
        "If you only want to test a prompt, stay in Benchmark most of the "
        "time.",
        "Example: a product manager stays in Benchmark during prompt "
        "iteration, but switches to Admin only when a queued run looks stuck.",
    )
    _render_help_card(
        "Benchmark Presets",
        "A preset is a saved benchmark setup.",
        "Use this single section to load, import, export, or save benchmark "
        "presets. A preset can store the prompt, provider config, dataset "
        "choice, mode, and judge settings.",
        "This is useful when you compare prompt versions over time or share "
        "one repeatable test setup with teammates.",
        "Example: a support team saves a preset for the release candidate so "
        "every weekly rerun uses the same prompt, dataset, and scoring setup.",
    )
    _render_help_card(
        "System Prompt",
        "This is the text you actually want to test.",
        "Paste the prompt, upload it from a file, or start from an example. "
        "The benchmark checks whether the model follows the prompt when it is "
        "under pressure.",
        "If the prompt is vague, contradictory, or missing refusal rules, the "
        "benchmark will usually expose that very quickly.",
        "Example: a legal assistant prompt may sound fine in normal chat, "
        "but the benchmark can reveal that it still answers prohibited "
        "advice requests.",
    )
    _render_help_card(
        "LLM Provider",
        "This is the model that will answer the test attacks.",
        "Set the provider, model name, API key or endpoint, and any optional "
        "retrieval settings if your provider needs them. Available models "
        "load automatically when you change the provider or its connection "
        "settings.",
        "If there is a provider validation error here, the benchmark cannot "
        "run. Fix this section first before checking anything else.",
        "Example: a procurement team can point the same prompt at two "
        "different vendors and see which model behaves more safely under "
        "the same attacks.",
    )
    _render_help_card(
        "How Many Tests?",
        "This controls speed versus coverage.",
        "Use Quick when you want a fast check while editing the prompt. Use "
        "Standard for most real work. Use Full before release or after major "
        "changes. Use Custom only if you know exactly how many rows you want.",
        "A small run is faster but less representative. A larger run gives a "
        "more reliable picture.",
        "Example: a security team may use Full before launch, while a prompt "
        "writer uses Quick after every small wording change.",
    )
    _render_help_card(
        "Benchmark Dataset",
        "This decides what exactly will be thrown at the model.",
        "A dataset is a collection of test cases: jailbreaks, injections, "
        "role "
        "confusion, domain-specific attacks, and other tricky inputs.",
        "Use the built-in dataset for a general benchmark. Use custom or "
        "saved "
        "packs when you want to focus on one product area, one risk type, or "
        "one team workflow.",
        "Example: a healthcare team can use a narrower pack focused on "
        "unsafe medical guidance, while a support team uses refund and "
        "escalation attacks.",
    )
    _render_help_card(
        "Judge & Detectors",
        "This section decides how answers are evaluated.",
        "The judge gives scores. Detectors add extra checks for unsafe, weak, "
        "or suspicious behavior.",
        "Most users should leave the defaults alone. Change this only if your "
        "domain has special rules and you understand how the scoring should "
        "change.",
        "Example: a compliance team may tighten review behavior so borderline "
        "answers are not treated as automatically acceptable.",
    )
    _render_help_card(
        "Advanced",
        "These are extra analysis options.",
        "They give deeper detail such as semantic similarity and degradation, "
        "which can help explain why a score changed.",
        "You do not need this section for normal runs. Open it when you are "
        "debugging behavior in more detail.",
        "Example: a prompt engineer uses this after a rewrite to "
        "understand why the score dropped even though failed examples look "
        "similar at first glance.",
    )
    st.markdown("### Common Sidebar Mistakes")
    st.markdown(
        "- Running without a real system prompt loaded.\n"
        "- Ignoring provider validation errors.\n"
        "- Using Quick mode and treating that result like a final security "
        "score.\n"
        "- Changing judge settings without understanding the scoring effect.\n"
        "- Forgetting to save a good setup as a preset."
    )


def _render_benchmark_workspace_help() -> None:
    st.subheader("Benchmark Workspace")
    st.markdown(
        "This is the main work area. The tabs are not separate tools. They "
        "are different steps around the same benchmark workflow."
    )

    _render_help_card(
        "Run Benchmark",
        "This is where the real test happens.",
        "When you click **Start Benchmark**, the app sends your chosen test "
        "set "
        "to your configured model, scores the answers, and stores the run "
        "for later review.",
        "Use this tab after the sidebar is ready and there are no validation "
        "errors.",
        "Example: before shipping a banking or support assistant, the "
        "team runs this tab to verify the latest prompt still resists the "
        "selected attacks.",
    )
    _render_help_card(
        "Analyze Prompt",
        "This gives you a quick reading of the prompt before a full run.",
        "It tries to summarize the prompt's role, domain, boundaries, and "
        "capabilities so you can see whether the prompt is clear enough.",
        "Use it when you want a quick sanity check before spending time on a "
        "larger benchmark run.",
        "Example: a prompt engineer uses this to notice that the prompt never "
        "clearly says when the assistant must refuse or escalate.",
    )
    _render_help_card(
        "Build Pack",
        "This helps you build a narrower custom test pack.",
        "You can take the current dataset and narrow it to a smaller, more "
        "specific pack for one attack family, one domain, or a release gate.",
        "Use this when the default dataset is too broad and you want focused "
        "regression checks.",
        "Example: an insurance company builds a smaller pack around "
        "claims, policy interpretation, and PII leakage instead of "
        "rerunning the full set.",
    )
    _render_help_card(
        "Compare Versions",
        "This shows whether things got better or worse over time.",
        "Compare old and new runs after changing the prompt, model, dataset, "
        "or scoring setup.",
        "This is one of the most useful tabs when you are iterating, because "
        "it helps you see regressions instead of relying on memory.",
        "Example: a SaaS team rewrites a support prompt and uses this tab to "
        "check whether customer-friendly wording weakened refusal behavior.",
    )

    st.markdown("### When To Save A Benchmark Preset")
    st.markdown(
        "Save a preset after you have chosen a prompt, provider, dataset, and "
        "judge configuration that you want to run repeatedly. This is the "
        "easiest way to keep runs comparable over time."
    )

    st.markdown("### Recommended Order")
    st.markdown(
        "1. Load prompt and provider in the sidebar.\n"
        "2. Optionally run **Analyze Prompt** for a quick structural check.\n"
        "3. Run **Run Benchmark**.\n"
        "4. Read failures in **Results**.\n"
        "5. If needed, use **Build Pack** for a focused regression set.\n"
        "6. After changes, use **Compare Versions** to confirm improvement."
    )


def _render_results_help() -> None:
    st.subheader("Results")
    st.markdown(
        "After a run finishes, this area shows what passed, what failed, and "
        "where the prompt is weak."
    )

    st.markdown(
        "### How To Read Results Without Overthinking It\n"
        "Start from the top and move from summary to detail:\n"
        "1. Look at the overall score.\n"
        "2. Look at the weakest categories.\n"
        "3. Open the exact failed examples.\n"
        "4. Check whether failures repeat the same pattern.\n"
        "5. Change the prompt and compare with the previous run."
    )

    result_sections = [
        (
            "Score Hero",
            "The headline summary. It shows the main score, pass rate, attack "
            "success rate, speed, token use, and how many cases need review.",
            "Example: a release manager checks this first to see whether "
            "the run is good enough to continue or should be blocked "
            "immediately.",
        ),
        (
            "Categories",
            "Breaks the score into parts so you can see which type of attack "
            "causes the most trouble.",
            "Example: a support team may find that refund and escalation "
            "attacks are weak even when the total score looks acceptable.",
        ),
        (
            "Charts",
            "Shows the same story in charts, which is often faster to read "
            "than raw tables.",
            "Example: a product lead uses charts in a review meeting to show "
            "which categories improved after a prompt rewrite.",
        ),
        (
            "Results",
            "Shows the individual rows: the input, the model answer, the "
            "score, and the failure label.",
            "Example: a prompt engineer reads exact failed rows to see the "
            "words or attack style that caused the model to break policy.",
        ),
        (
            "Detectors",
            "Shows which automatic checks fired and where they noticed risky "
            "behavior.",
            "Example: a safety team uses this to see whether risky output was "
            "caught by detectors even when the main score looked borderline.",
        ),
        (
            "Attacks",
            "Groups problems by attack style so you can see patterns instead "
            "of isolated rows.",
            "Example: a security review may show that prompt injection "
            "attacks are the main issue, not generic adversarial phrasing.",
        ),
        (
            "Review",
            "Shows cases where a human should take a closer look because the "
            "result is borderline or suspicious.",
            "Example: a compliance lead reviews these before sign-off when "
            "the company does not want borderline answers auto-approved.",
        ),
        (
            "Export",
            "Lets you download the results for reporting, sharing, or keeping "
            "a record of the run.",
            "Example: an audit or governance team exports results to attach "
            "them to a release approval record.",
        ),
        (
            "Log",
            "Shows the execution trail, which helps when you want to debug "
            "what happened during the run.",
            "Example: a platform engineer checks this when a run looks "
            "incomplete or a provider call behaved unexpectedly.",
        ),
    ]
    for name, description, example in result_sections:
        st.markdown(f"### {name}")
        st.markdown(f"**What you see here:** {description}")
        st.markdown(f"**Example from real work:** {example}")

    st.markdown("### How To Read The Output")
    st.markdown(
        "- Start with the overall score and pass rate.\n"
        "- Then open the lowest categories.\n"
        "- Then inspect the exact failing tests.\n"
        "- Finally use Compare Versions to confirm whether prompt changes "
        "helped."
    )

    st.markdown("### Simple Rule Of Thumb")
    st.markdown(
        "A single weird failure is interesting. Repeated failures of the same "
        "kind usually mean the prompt needs a real rewrite, not a tiny "
        "wording tweak."
    )


def _render_admin_help() -> None:
    st.subheader("Admin")
    st.markdown(
        "Admin is for operating the system itself, not for day-to-day prompt "
        "writing. Use it when you need to inspect jobs, stored results, "
        "workers, "
        "plugins, webhooks, datasets, or health state."
    )

    st.markdown(
        "### In simple words\n"
        "If Benchmark is where you test prompts, Admin is where you watch the "
        "machinery behind the scenes."
    )

    admin_sections = [
        (
            "Jobs",
            "See benchmark jobs in the queue, running now, completed already, "
            "failed, or moved aside for retry/replay.",
            "Example: a platform team opens this when CI triggered many "
            "runs and one batch appears stuck or delayed.",
        ),
        (
            "Results",
            "Open saved benchmark results and compare stored result files "
            "without rerunning the whole benchmark.",
            "Example: a governance team reopens stored results during release "
            "review instead of paying to rerun the whole benchmark.",
        ),
        (
            "Webhooks",
            "Check failed webhook deliveries and resend them if the target "
            "service missed them.",
            "Example: an integration team uses this when a Slack, Jira, or "
            "internal approval callback did not arrive.",
        ),
        (
            "Redis",
            "Inspect pending Redis work items when the system uses Redis "
            "based "
            "workers.",
            "Example: an infra team checks this when background workers are "
            "distributed and a run stays pending too long.",
        ),
        (
            "Metrics",
            "See counters, runtime state, and backend health information.",
            "Example: an ops team checks queue depth and runtime counters "
            "when benchmark throughput drops.",
        ),
        (
            "Plugins",
            "See which providers, transforms, judges, and exporters are "
            "loaded through the plugin system.",
            "Example: after adding a custom provider plugin, the team "
            "checks here that it was actually discovered and loaded.",
        ),
        (
            "Smoke Tools",
            "Run quick checks for provider capabilities such as vision, "
            "embeddings, or retrieval.",
            "Example: before a wider benchmark, an ML engineer runs a "
            "quick smoke test to confirm embeddings or retrieval are wired "
            "correctly.",
        ),
        (
            "Presets & Datasets",
            "Validate datasets, convert packs, inspect catalogs, and maintain "
            "dataset-related assets.",
            "Example: a benchmark owner validates a custom domain pack "
            "before the team starts relying on it for release decisions.",
        ),
    ]
    for name, description, example in admin_sections:
        st.markdown(f"### {name}")
        st.markdown(f"**What it is for:** {description}")
        st.markdown(f"**Example from real work:** {example}")

    st.warning(
        "Use Admin when you are operating the benchmark system itself. Use "
        "the "
        "Benchmark workspace when your goal is just to test a prompt."
    )

    st.markdown("### When You Probably Do Not Need Admin")
    st.markdown(
        "- You just want to run a normal benchmark.\n"
        "- You only need to paste a prompt and get a score.\n"
        "- You are not debugging jobs, workers, or stored results."
    )

    st.markdown("### When Admin Is The Right Place")
    st.markdown(
        "- A run is stuck or missing.\n"
        "- You need to inspect saved results outside the current UI session.\n"
        "- A webhook failed and needs replay.\n"
        "- You want to validate or manage datasets.\n"
        "- You are checking whether workers or plugins are healthy."
    )


def _render_help_card(
    title: str,
    plain_words: str,
    how_it_works: str,
    when_to_use: str,
    example: str,
) -> None:
    """Render one consistent help card."""
    st.markdown(f"### {title}")
    st.markdown(f"**In simple words:** {plain_words}")
    st.markdown(f"**How it works:** {how_it_works}")
    st.markdown(f"**When to use it:** {when_to_use}")
    st.markdown(f"**Example from real work:** {example}")
