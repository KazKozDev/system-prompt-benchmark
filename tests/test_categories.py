"""Tests for benchmark category configuration."""

from __future__ import annotations

import pytest

from src.core.benchmark_categories import (
    BENCHMARK_CATEGORIES,
    CATEGORY_MAPPING,
    get_all_categories,
    get_category_for_test,
    get_category_info,
    get_category_weight,
    is_critical_category,
)


def test_all_categories_have_required_keys():
    required = {"name", "description", "weight", "critical"}
    for cat_name, cat_info in BENCHMARK_CATEGORIES.items():
        for key in required:
            assert key in cat_info, f"Category '{cat_name}' missing key '{key}'"


def test_weights_sum_to_one():
    total = sum(info["weight"] for info in BENCHMARK_CATEGORIES.values())
    assert total == pytest.approx(1.0, abs=0.001), (
        f"Category weights sum to {total}, expected 1.0"
    )


def test_get_category_weight_known():
    assert get_category_weight("role_adherence") == pytest.approx(0.15)
    assert get_category_weight("security") == pytest.approx(0.15)
    assert get_category_weight("jailbreak_resistance") == pytest.approx(0.15)


def test_get_category_weight_unknown_returns_zero():
    assert get_category_weight("nonexistent_category") == 0.0


def test_critical_categories():
    assert is_critical_category("role_adherence") is True
    assert is_critical_category("security") is True
    assert is_critical_category("jailbreak_resistance") is True
    assert is_critical_category("ethics_compliance") is True


def test_non_critical_categories():
    assert is_critical_category("consistency") is False
    assert is_critical_category("edge_cases") is False


def test_get_category_for_test_known_mappings():
    assert get_category_for_test("jailbreak") == "jailbreak_resistance"
    assert get_category_for_test("prompt_leaking") == "security"
    assert get_category_for_test("authority_bypass") == "instruction_following"
    assert get_category_for_test("basic_identity") == "role_adherence"
    assert get_category_for_test("out_of_scope") == "scope_boundaries"


def test_get_category_for_test_unknown_falls_back():
    # unknown category should return fallback (not raise)
    result = get_category_for_test("completely_unknown_category_xyz")
    assert isinstance(result, str)
    assert len(result) > 0


def test_all_category_mappings_resolve_to_valid_categories():
    valid = set(BENCHMARK_CATEGORIES.keys())
    for test_cat, universal_cat in CATEGORY_MAPPING.items():
        assert universal_cat in valid, (
            f"CATEGORY_MAPPING['{test_cat}'] = '{universal_cat}' is not a valid category"
        )


def test_get_all_categories_returns_all():
    cats = get_all_categories()
    assert set(cats) == set(BENCHMARK_CATEGORIES.keys())


def test_get_category_info_returns_dict():
    info = get_category_info("security")
    assert info["name"] == "Security"
    assert info["weight"] == pytest.approx(0.15)


def test_get_category_info_unknown_returns_empty():
    info = get_category_info("does_not_exist")
    assert info == {}
