"""Tests for admin UI helper behavior."""

from __future__ import annotations


class _FakeColumn:
    def __enter__(self) -> "_FakeColumn":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


class _FakeStreamlit:
    def __init__(self, pressed_keys: set[str] | None = None) -> None:
        self.pressed_keys = pressed_keys or set()
        self.session_state: dict[str, object] = {}
        self.markdown_calls: list[tuple[str, bool]] = []
        self.rerun_count = 0

    def columns(self, spec: list[float]) -> list[_FakeColumn]:
        return [_FakeColumn() for _ in spec]

    def markdown(self, body: str, unsafe_allow_html: bool = False) -> None:
        self.markdown_calls.append((body, unsafe_allow_html))

    def button(
        self,
        label: str,
        *,
        key: str,
        width: str | None = None,
    ) -> bool:
        del label, width
        return key in self.pressed_keys

    def rerun(self) -> None:
        self.rerun_count += 1


def test_render_compact_state_chips_outputs_expected_html(monkeypatch) -> None:
    from src.ui import admin_views

    fake_st = _FakeStreamlit()
    monkeypatch.setattr(admin_views, "st", fake_st)

    admin_views._render_compact_state_chips(
        "Runtime",
        ["Backend redis", "Workers 4", "Redis healthy"],
    )

    assert len(fake_st.markdown_calls) == 1
    html, unsafe_allow_html = fake_st.markdown_calls[0]
    assert unsafe_allow_html is True
    assert "Runtime" in html
    assert "Backend redis" in html
    assert "Workers 4" in html
    assert "Redis healthy" in html


def test_clear_sort_resets_session(monkeypatch) -> None:
    from src.ui import admin_views

    fake_st = _FakeStreamlit(pressed_keys={"clear-sort"})
    fake_st.session_state.update(
        {
            "sort_field": "Owner",
            "sort_direction": "Ascending",
        }
    )
    monkeypatch.setattr(admin_views, "st", fake_st)

    admin_views._render_table_state_badges(
        sort_field="Owner",
        sort_direction="Ascending",
        filter_summary="owner contains 'artem'",
        clear_button_key="clear-filters",
        clear_filter_defaults={"owner_filter": "", "page": 1},
        clear_sort_button_key="clear-sort",
        clear_sort_defaults={
            "sort_field": "Created Time",
            "sort_direction": "Descending",
        },
        sort_is_default=False,
    )

    assert len(fake_st.markdown_calls) == 1
    html, unsafe_allow_html = fake_st.markdown_calls[0]
    assert unsafe_allow_html is True
    assert "Active sort" in html
    assert "Owner" in html
    assert "Ascending" in html
    assert "Filtered: owner contains 'artem'" in html
    assert fake_st.session_state["sort_field"] == "Created Time"
    assert fake_st.session_state["sort_direction"] == "Descending"
    assert fake_st.rerun_count == 1


def test_clear_filters_reset_session(monkeypatch) -> None:
    from src.ui import admin_views

    fake_st = _FakeStreamlit(pressed_keys={"clear-filters"})
    fake_st.session_state.update(
        {
            "owner_filter": "artem",
            "page": 3,
        }
    )
    monkeypatch.setattr(admin_views, "st", fake_st)

    admin_views._render_table_state_badges(
        sort_field="Created Time",
        sort_direction="Descending",
        filter_summary="owner contains 'artem'",
        clear_button_key="clear-filters",
        clear_filter_defaults={"owner_filter": "", "page": 1},
        clear_sort_button_key="clear-sort",
        clear_sort_defaults={
            "sort_field": "Created Time",
            "sort_direction": "Descending",
        },
        sort_is_default=True,
    )

    assert fake_st.session_state["owner_filter"] == ""
    assert fake_st.session_state["page"] == 1
    assert fake_st.rerun_count == 1


def test_row_range_text_formats_empty_and_paginated_ranges() -> None:
    from src.ui import admin_views

    assert admin_views._row_range_text(0, 1, 50) == "Showing 0 of 0"
    assert admin_views._row_range_text(125, 1, 50) == "Showing 1-50 of 125"
    assert admin_views._row_range_text(125, 3, 50) == "Showing 101-125 of 125"


def test_job_filter_summary_formats_active_filters() -> None:
    from src.ui import admin_views

    summary = admin_views._job_filter_summary(
        selected_statuses=["queued", "failed"],
        available_statuses=["queued", "running", "failed"],
        owner_filter="artem",
        result_filter="Webhook Failures",
        search_term="dead-letter",
    )

    assert summary == (
        "statuses 2 selected | owner contains 'artem' | "
        "Webhook Failures | search 'dead-letter'"
    )
    assert (
        admin_views._job_filter_summary(
            selected_statuses=["queued", "running"],
            available_statuses=["running", "queued"],
            owner_filter="",
            result_filter="All",
            search_term="",
        )
        is None
    )


def test_result_filter_summary_formats_active_filters() -> None:
    from src.ui import admin_views

    summary = admin_views._result_filter_summary(
        owner_filter="artem",
        provider_filter="openai",
        dataset_filter="support",
        search_term="job-123",
    )

    assert summary == (
        "owner contains 'artem' | provider contains 'openai' | "
        "dataset contains 'support' | search 'job-123'"
    )
    assert (
        admin_views._result_filter_summary(
            owner_filter="",
            provider_filter="",
            dataset_filter="",
            search_term="",
        )
        is None
    )


def test_compare_result_payloads_returns_expected_deltas() -> None:
    from src.ui import admin_views

    base_payload = {
        "results": [
            {
                "test_id": 1,
                "category": "security",
                "universal_category": "security",
                "score": 0.4,
                "review_required": True,
            },
            {
                "test_id": 2,
                "category": "instruction_following",
                "universal_category": "instruction_following",
                "score": 0.8,
                "review_required": False,
            },
        ]
    }
    candidate_payload = {
        "results": [
            {
                "test_id": 1,
                "category": "security",
                "universal_category": "security",
                "score": 0.9,
                "review_required": False,
            },
            {
                "test_id": 2,
                "category": "instruction_following",
                "universal_category": "instruction_following",
                "score": 0.7,
                "review_required": False,
            },
        ]
    }

    comparison = admin_views._compare_result_payloads(
        base_payload,
        candidate_payload,
        base_label="base.json",
        candidate_label="candidate.json",
    )

    assert comparison["base"]["label"] == "base.json"
    assert comparison["candidate"]["label"] == "candidate.json"
    assert comparison["delta"]["overall_score"] > 0
    assert comparison["delta"]["pass_rate"] == 0.5
    assert comparison["delta"]["review_count"] == -1
    assert comparison["category_deltas"]["security"] == 0.5
    assert comparison["category_deltas"]["instruction_following"] == -0.1
