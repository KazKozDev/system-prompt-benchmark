"""Shared Streamlit styling helpers."""

from __future__ import annotations

from pathlib import Path

import streamlit as st


CSS_PATH = Path("assets/app.css")


def apply_app_styles() -> None:
    """Inject the shared application stylesheet into Streamlit."""
    if not CSS_PATH.exists():
        return
    css = CSS_PATH.read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
