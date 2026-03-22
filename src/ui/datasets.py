"""Dataset UI helpers that are independent from Streamlit widgets."""

from __future__ import annotations

import os
import tempfile
from types import SimpleNamespace
from pathlib import Path

from src.datasets import load_dataset_bundle, validate_dataset_rows


def load_uploaded_dataset(uploaded_file) -> tuple[list[dict], list[str], dict]:
    suffix = Path(uploaded_file.name).suffix or ".json"
    with tempfile.NamedTemporaryFile("wb", suffix=suffix, delete=False) as handle:
        handle.write(uploaded_file.getvalue())
        temp_path = handle.name
    try:
        bundle = load_dataset_bundle(temp_path)
        bundle.metadata["name"] = uploaded_file.name
        issues = validate_dataset_rows(bundle.tests)
        return bundle.tests, issues, bundle.metadata
    finally:
        try:
            os.unlink(temp_path)
        except OSError:
            pass


def make_dataset_bundle_like(tests: list[dict], metadata: dict):
    return SimpleNamespace(tests=tests, metadata=metadata)


def compare_dataset_packs(base_bundle, candidate_bundle) -> dict:
    base_tests = base_bundle.tests
    candidate_tests = candidate_bundle.tests

    def _category_counts(rows):
        counts = {}
        for row in rows:
            category = row.get("universal_category", "unknown")
            counts[category] = counts.get(category, 0) + 1
        return counts

    base_ids = {row["id"] for row in base_tests}
    candidate_ids = {row["id"] for row in candidate_tests}
    base_categories = _category_counts(base_tests)
    candidate_categories = _category_counts(candidate_tests)
    category_names = sorted(set(base_categories) | set(candidate_categories))

    category_rows = []
    for name in category_names:
        category_rows.append(
            {
                "category": name,
                "base_count": base_categories.get(name, 0),
                "candidate_count": candidate_categories.get(name, 0),
                "delta": candidate_categories.get(name, 0) - base_categories.get(name, 0),
            }
        )

    base_transforms = set(base_bundle.metadata.get("transforms", []))
    candidate_transforms = set(candidate_bundle.metadata.get("transforms", []))

    return {
        "base": {
            "name": base_bundle.metadata.get("name", "Base"),
            "version": base_bundle.metadata.get("version", "1.0"),
            "num_tests": len(base_tests),
            "source": base_bundle.metadata.get("source", ""),
        },
        "candidate": {
            "name": candidate_bundle.metadata.get("name", "Candidate"),
            "version": candidate_bundle.metadata.get("version", "1.0"),
            "num_tests": len(candidate_tests),
            "source": candidate_bundle.metadata.get("source", ""),
        },
        "summary": {
            "test_count_delta": len(candidate_tests) - len(base_tests),
            "shared_ids": len(base_ids & candidate_ids),
            "new_ids": len(candidate_ids - base_ids),
            "removed_ids": len(base_ids - candidate_ids),
            "shared_transforms": sorted(base_transforms & candidate_transforms),
            "candidate_only_transforms": sorted(candidate_transforms - base_transforms),
            "base_only_transforms": sorted(base_transforms - candidate_transforms),
        },
        "category_deltas": category_rows,
    }
