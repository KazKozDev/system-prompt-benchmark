"""Custom success criteria and assertion helpers."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


SUPPORTED_ASSERTION_TYPES = {
    "contains",
    "not_contains",
    "regex",
    "not_regex",
    "equals",
    "not_equals",
}


@dataclass
class AssertionEvaluation:
    score: float
    passed: bool
    operator: str
    total: int
    passed_count: int
    failed: list[dict]


def normalize_success_criteria(raw: Any) -> dict | None:
    if raw in (None, "", []):
        return None
    if isinstance(raw, str):
        raw = json.loads(raw)
    if isinstance(raw, list):
        raw = {"operator": "all", "assertions": raw}
    if not isinstance(raw, dict):
        raise ValueError("success_criteria must be a dict, list, or JSON string")
    assertions = raw.get("assertions", [])
    if not isinstance(assertions, list):
        raise ValueError("success_criteria.assertions must be a list")
    operator = str(raw.get("operator", "all")).lower()
    if operator not in {"all", "any"}:
        raise ValueError("success_criteria.operator must be 'all' or 'any'")

    normalized_assertions = []
    for index, assertion in enumerate(assertions):
        if not isinstance(assertion, dict):
            raise ValueError(f"assertion {index} must be an object")
        assertion_type = str(assertion.get("type", "")).lower()
        if assertion_type not in SUPPORTED_ASSERTION_TYPES:
            raise ValueError(f"assertion {index}: unsupported type '{assertion_type}'")
        if "value" not in assertion:
            raise ValueError(f"assertion {index}: missing value")
        normalized_assertions.append(
            {
                "type": assertion_type,
                "value": assertion["value"],
                "field": assertion.get("field", "response"),
                "flags": assertion.get("flags", ""),
                "description": assertion.get("description", ""),
            }
        )
    return {"operator": operator, "assertions": normalized_assertions}


def evaluate_success_criteria(response: str, test: dict) -> AssertionEvaluation | None:
    criteria = normalize_success_criteria(test.get("success_criteria"))
    if not criteria or not criteria.get("assertions"):
        return None

    failed = []
    passed_count = 0
    for assertion in criteria["assertions"]:
        matched = _evaluate_single_assertion(assertion, response, test)
        if matched:
            passed_count += 1
        else:
            failed.append(assertion)

    total = len(criteria["assertions"])
    operator = criteria["operator"]
    if operator == "all":
        passed = passed_count == total
    else:
        passed = passed_count > 0

    return AssertionEvaluation(
        score=round(passed_count / total, 4) if total else 0.0,
        passed=passed,
        operator=operator,
        total=total,
        passed_count=passed_count,
        failed=failed,
    )


def _evaluate_single_assertion(assertion: dict, response: str, test: dict) -> bool:
    field_name = assertion.get("field", "response")
    if field_name == "response":
        actual = response
    else:
        actual = test.get(field_name, "")
    actual_text = str(actual)
    expected = str(assertion.get("value", ""))
    assertion_type = assertion["type"]

    if assertion_type == "contains":
        return expected in actual_text
    if assertion_type == "not_contains":
        return expected not in actual_text
    if assertion_type == "equals":
        return actual_text == expected
    if assertion_type == "not_equals":
        return actual_text != expected

    flags = 0
    if "i" in str(assertion.get("flags", "")):
        flags |= re.IGNORECASE
    regex = re.compile(expected, flags)
    if assertion_type == "regex":
        return bool(regex.search(actual_text))
    if assertion_type == "not_regex":
        return not bool(regex.search(actual_text))
    return False
