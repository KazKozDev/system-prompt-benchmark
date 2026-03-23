"""Onboarding and empty-state views for the Streamlit app."""

from __future__ import annotations

import streamlit as st


def render_onboarding() -> None:
    """Render the initial onboarding state shown before a prompt is loaded."""
    st.markdown(
        (
            "<div style=\"text-align:center; padding: 4px 0 26px;\">"
            "<div style=\"font-size:1.9rem; font-weight:800; color:#0f172a; "
            "margin-bottom:10px;\">How does this tool work?</div>"
            "<div style=\"font-size:1.05rem; color:#64748b; max-width:580px; "
            "margin:0 auto; line-height:1.6;\">"
            "You paste the instructions your AI follows.<br>"
            "We fire hundreds of real attack messages at it.<br>"
            "You see a score — and <em>exactly</em> what broke."
            "</div></div>"
        ),
        unsafe_allow_html=True,
    )

    st.info(
        "  **Start here** — paste your system prompt in the **left sidebar**. "
        "The four tabs above will unlock as soon as you do."
    )
    st.markdown("<br>", unsafe_allow_html=True)

    step_columns = st.columns(4)
    steps = [
        (
            "1.",
            "Paste Your Prompt",
            "Those are the instructions your AI gets at the start of every "
            "conversation. Paste them in the sidebar, upload a .txt file, or "
            "pick one of the built-in examples.",
        ),
        (
            "2.",
            "Pick a Provider",
            "Choose which AI model should answer the attack messages: OpenAI, "
            "Anthropic, Gemini, Groq, Mistral, or a local model via Ollama.",
        ),
        (
            "3.",
            "Hit Start Benchmark",
            "We send 10–300 crafted trick messages at your AI and score every "
            "response automatically.",
        ),
        (
            "4.",
            "Read Your Report",
            "Get an overall score, category breakdowns, failed attacks, and "
            "downloadable reports you can compare across runs.",
        ),
    ]
    for column, (number, title, description) in zip(step_columns, steps):
        with column:
            st.markdown(
                (
                    '<div class="guide-step">'
                    f'<div class="guide-step-num">{number}</div>'
                    f"<h3>{title}</h3>"
                    f"<p>{description}</p>"
                    "</div>"
                ),
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)
    st.divider()

    left_column, right_column = st.columns(2, gap="large")
    with left_column:
        st.markdown("###  What is a system prompt?")
        st.markdown(
            "Think of your AI assistant as a new employee on their first day. "
            "The **system prompt** is the instruction manual you hand them "
            "before they talk to anyone. Without these instructions, the AI "
            "will do whatever a user asks. This tool checks how well those "
            "instructions hold up under active attack attempts."
        )

        st.markdown("###  Where do I find my system prompt?")
        st.markdown(
            "- **OpenAI API** — the `system` role message in your request\n"
            "- **Custom GPT** — the *Instructions* field in GPT Builder\n"
            "- **LangChain / LlamaIndex** — the `SystemMessage` or "
            "`system_prompt` variable\n"
            "- **No prompt yet?** Pick one of the built-in examples from the "
            "sidebar to see the workflow first"
        )

        st.markdown("###  What kinds of attacks are tested?")
        attack_columns = st.columns(2)
        attacks = [
            ("Prompt Injection", '"Ignore all previous instructions and…"'),
            ("Jailbreaks", "DAN mode, developer mode, roleplay frames"),
            ("Prompt Leaking", "Extracting your secret instructions verbatim"),
            (
                "Authority Bypass",
                "Fake CEO, admin, or emergency override claims",
            ),
            ("Encoding Tricks", "Base64, ROT13, hex, and similar obfuscation"),
            (
                "Multilingual",
                "The same attack delivered in multiple languages",
            ),
            ("RAG Poisoning", "Instructions hidden inside fake documents"),
            (
                "Multi-turn Escalation",
                "Slow pressure increase over many turns",
            ),
        ]
        for column, attack_group in zip(
            attack_columns,
            [attacks[:4], attacks[4:]],
        ):
            with column:
                for attack_name, attack_desc in attack_group:
                    st.markdown(
                        (
                            '<div class="atk-item">'
                            f"<strong>{attack_name}</strong>"
                            f'<span class="atk-sub">{attack_desc}</span>'
                            "</div>"
                        ),
                        unsafe_allow_html=True,
                    )

    with right_column:
        st.markdown("###  What do the scores mean?")
        st.markdown(
            "Every test gets a score from **0.0** to **1.0**. "
            "The overall score "
            "is a weighted average across twelve benchmark categories."
        )
        score_bands = [
            (
                "#16a34a",
                "0.90 – 1.00",
                "Excellent",
                "Held up against almost everything.",
            ),
            (
                "#d97706",
                "0.70 – 0.89",
                "Good",
                "Minor issues worth hardening.",
            ),
            (
                "#ea580c",
                "0.50 – 0.69",
                "Needs work",
                "Several attack types got through.",
            ),
            (
                "#dc2626",
                "0.00 – 0.49",
                "Vulnerable",
                "Significant gaps before production.",
            ),
        ]
        for color, score_range, label, description in score_bands:
            st.markdown(
                (
                    '<div class="score-row">'
                    f'<div class="score-dot" style="background:{color}"></div>'
                    "<div>"
                    f"<span style=\"font-weight:700;font-size:.875rem;\">"
                    f"{score_range} — {label}</span><br>"
                    f"<span style=\"font-size:.82rem;color:#64748b;\">"
                    f"{description}</span></div></div>"
                ),
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("###  What are the 12 test categories?")
        category_columns = st.columns(2)
        categories = [
            ("Role Adherence", "Does the AI stay in its assigned role?"),
            ("Instruction Following", "Does it obey every rule you set?"),
            ("Security", "Does it keep your instructions secret?"),
            ("Jailbreak Resistance", "Does it resist manipulation attempts?"),
            ("Ethics & Compliance", "Does it refuse harmful requests?"),
            ("Scope Boundaries", "Does it decline out-of-scope requests?"),
            ("Consistency", "Are answers stable across similar questions?"),
            ("Robustness", "Does it hold up with typos and other languages?"),
            (
                "Multi-turn Behaviour",
                "Does it stay safe across a long conversation?",
            ),
            ("Edge Cases", "Does it handle weird or malformed input?"),
            ("Graceful Degradation", "Does it refuse politely when it must?"),
            (
                "Constraint Following",
                "Does it respect format and length rules?",
            ),
        ]
        for column, category_group in zip(
            category_columns,
            [categories[:6], categories[6:]],
        ):
            with column:
                for category_name, category_desc in category_group:
                    st.markdown(
                        (
                            '<div class="atk-item">'
                            f"<strong>{category_name}</strong>"
                            f'<span class="atk-sub">{category_desc}</span>'
                            "</div>"
                        ),
                        unsafe_allow_html=True,
                    )

    st.divider()
    st.markdown("###  Common questions")
    faq_columns = st.columns(3, gap="medium")
    faqs = [
        (
            "Do I need Ollama?",
            "Ollama is useful for local judging. Without it, "
            "results can still "
            "be generated but more items may land in the review queue.",
        ),
        (
            "How long does a run take?",
            "Quick mode is usually under a minute, standard runs take a few "
            "minutes, and full runs depend mostly on provider latency.",
        ),
        (
            "Is my prompt kept private?",
            "Your prompt is sent only to the provider you configure. If you "
            "need everything local, use Ollama for both execution and "
            "judging.",
        ),
    ]
    for column, (question, answer) in zip(faq_columns, faqs):
        with column:
            st.markdown(
                (
                    '<div class="faq-card">'
                    f"<h4>{question}</h4>"
                    f"<p>{answer}</p>"
                    "</div>"
                ),
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)
    st.success(
        "  **Ready?** Paste your system prompt in the **left sidebar** — "
        "the benchmark workflow will appear instantly."
    )
