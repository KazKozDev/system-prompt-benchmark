"""Tests for results UI review queue rendering."""

from __future__ import annotations


class _FakeExpander:
    def __enter__(self) -> "_FakeExpander":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


class _FakeColumn:
    def __enter__(self) -> "_FakeColumn":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


class _FakeSessionState(dict):
    def __getattr__(self, name: str):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value) -> None:
        self[name] = value


class _FakeStreamlit:
    def __init__(self) -> None:
        self.session_state = _FakeSessionState()
        self.text_area_keys: list[str] = []
        self.button_keys: list[str] = []
        self.subheaders: list[str] = []

    def subheader(self, body: str) -> None:
        self.subheaders.append(body)

    def success(self, body: str) -> None:
        del body

    def expander(self, label: str, expanded: bool = False) -> _FakeExpander:
        del label, expanded
        return _FakeExpander()

    def write(self, body) -> None:
        del body

    def code(self, body) -> None:
        del body

    def caption(self, body: str) -> None:
        del body

    def json(self, body) -> None:
        del body

    def text_area(
        self,
        label: str,
        *,
        key: str,
        value: str,
        height: int,
    ) -> str:
        del label, value, height
        self.text_area_keys.append(key)
        return ""

    def columns(self, spec) -> list[_FakeColumn]:
        del spec
        return [_FakeColumn() for _ in range(4)]

    def button(self, label: str, *, key: str) -> bool:
        del label
        self.button_keys.append(key)
        return False

    def rerun(self) -> None:
        raise AssertionError("rerun should not be called in this test")


def test_render_review_queue_uses_unique_widget_keys_for_duplicate_test_ids(
    monkeypatch,
) -> None:
    from src.ui import results_views

    fake_st = _FakeStreamlit()
    monkeypatch.setattr(results_views, "st", fake_st)

    results = [
        {
            "test_id": 300,
            "category": "security",
            "score": 0.2,
            "result_label": "FAIL",
            "input": "first",
            "response": "first response",
            "review_required": True,
            "review_status": "REVIEW",
            "review_note": "",
        },
        {
            "test_id": 300,
            "category": "security_variant",
            "score": 0.3,
            "result_label": "FAIL",
            "input": "second",
            "response": "second response",
            "review_required": True,
            "review_status": "REVIEW",
            "review_note": "",
        },
    ]

    results_views._render_review_queue(results, lambda *_: None)

    assert fake_st.subheaders == ["Manual Review Queue"]
    assert fake_st.text_area_keys == ["review_note_300_0", "review_note_300_1"]
    assert "mark_pass_300_0" in fake_st.button_keys
    assert "mark_pass_300_1" in fake_st.button_keys
