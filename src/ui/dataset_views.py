"""Streamlit dataset-related views."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pandas as pd
import streamlit as st

from src.core.assertions import normalize_success_criteria
from src.datasets import (
    apply_transforms,
    available_transforms,
    build_local_dataset_registry,
    compose_dataset_bundles,
    compose_dataset_preset,
    evaluate_remote_catalog_status,
    export_dataset,
    fetch_remote_dataset_catalog,
    list_dataset_presets,
    load_dataset_bundle,
    sync_remote_dataset_catalog,
)
from src.ui.datasets import load_uploaded_dataset


def render_dataset_selector(
    default_dataset_path: str = "tests/safeprompt-benchmark-v2.json",
) -> dict:
    with st.expander("Registry & Remote Catalog", expanded=False):
        registry_cols = st.columns([1.4, 1])
        with registry_cols[0]:
            remote_catalog_url = st.text_input(
                "Remote Catalog URL",
                key="dataset_remote_catalog_url",
                placeholder="https://example.com/catalog.json",
            )
        with registry_cols[1]:
            remote_catalog_timeout = st.number_input(
                "Timeout",
                min_value=1.0,
                value=20.0,
                step=1.0,
                key="dataset_remote_catalog_timeout",
            )
        remote_catalog = None
        if st.button("Fetch Remote Catalog", key="dataset_fetch_remote_catalog"):
            if not remote_catalog_url.strip():
                st.warning("Enter a remote catalog URL first.")
            else:
                try:
                    remote_catalog = fetch_remote_dataset_catalog(
                        remote_catalog_url.strip(),
                        timeout_seconds=float(remote_catalog_timeout),
                    )
                    st.session_state["dataset_remote_catalog_payload"] = remote_catalog
                    st.success(
                        f"Loaded {len(remote_catalog.get('packs', []))} remote pack(s)."
                    )
                except Exception as exc:
                    st.error(f"Failed to fetch remote catalog: {exc}")
        remote_catalog = st.session_state.get("dataset_remote_catalog_payload")
        if remote_catalog:
            pack_options = {
                f"{pack.get('id', 'unknown')} · {pack.get('metadata', {}).get('name', pack.get('name', 'Unnamed'))}": pack.get(
                    "id"
                )
                for pack in remote_catalog.get("packs", [])
            }
            selected_pack_labels = st.multiselect(
                "Remote Packs",
                list(pack_options.keys()),
                key="dataset_remote_catalog_selection",
            )
            destination_dir = st.text_input(
                "Sync Destination",
                value="datasets/remote",
                key="dataset_remote_catalog_destination",
            )
            sync_force = st.checkbox(
                "Overwrite existing synced packs",
                value=False,
                key="dataset_remote_catalog_force",
            )
            allow_untrusted = st.checkbox(
                "Allow packs without sha256",
                value=False,
                key="dataset_remote_catalog_allow_untrusted",
                help="By default, sync requires a declared sha256 checksum in the remote catalog.",
            )
            statuses = evaluate_remote_catalog_status(
                remote_catalog, destination_dir=destination_dir
            )
            if statuses:
                st.dataframe(
                    pd.DataFrame(statuses)[
                        [
                            "id",
                            "name",
                            "status",
                            "trust_status",
                            "available_version",
                            "installed_version",
                            "path",
                        ]
                    ],
                    use_container_width=True,
                    hide_index=True,
                )
            if st.button("Sync Selected Packs", key="dataset_sync_remote_catalog"):
                try:
                    synced = sync_remote_dataset_catalog(
                        remote_catalog_url.strip(),
                        destination_dir=destination_dir,
                        pack_ids=[pack_options[label] for label in selected_pack_labels]
                        if selected_pack_labels
                        else None,
                        force=sync_force,
                        timeout_seconds=float(remote_catalog_timeout),
                        allow_untrusted=allow_untrusted,
                    )
                    st.success(f"Synced {len(synced)} pack(s) into {destination_dir}.")
                except Exception as exc:
                    st.error(f"Failed to sync packs: {exc}")
            st.json(remote_catalog)

    dataset_source = st.radio(
        "Dataset Source",
        [
            "Built-in Benchmark",
            "Preset Profiles",
            "Local Registry Packs",
            "Upload Custom Pack",
        ],
        label_visibility="collapsed",
    )

    dataset_label = "Built-in Benchmark"
    dataset_path = default_dataset_path
    dataset_tests = []
    dataset_issues = []
    dataset_metadata = {
        "name": dataset_label,
        "version": "builtin",
        "description": "",
        "source": dataset_path,
    }

    if dataset_source == "Built-in Benchmark":
        try:
            dataset_bundle = load_dataset_bundle(dataset_path)
            dataset_tests = dataset_bundle.tests
            dataset_metadata = dataset_bundle.metadata
            dataset_label = dataset_metadata.get("name", dataset_label)
            st.caption(f"{len(dataset_tests)} tests loaded from built-in pack")
        except Exception as exc:
            dataset_issues = [str(exc)]
            st.error(f"Failed to load built-in benchmark: {exc}")
    elif dataset_source == "Preset Profiles":
        presets = list_dataset_presets()
        preset_options = {
            f"{preset['name']} · {preset_id}": preset_id
            for preset_id, preset in presets.items()
        }
        selected_preset_label = st.selectbox(
            "Preset Profile",
            list(preset_options.keys()),
            key="dataset_preset_profile_select",
        )
        selected_preset_id = preset_options[selected_preset_label]
        selected_preset = presets[selected_preset_id]
        st.caption(selected_preset.get("description", ""))
        st.caption("Includes: " + ", ".join(selected_preset.get("pack_ids", [])))
        try:
            registry = build_local_dataset_registry()
            dataset_bundle = compose_dataset_preset(
                selected_preset_id, registry=registry
            )
            dataset_path = dataset_bundle.path
            dataset_tests = dataset_bundle.tests
            dataset_metadata = dataset_bundle.metadata
            dataset_label = dataset_metadata.get("name", selected_preset["name"])
            st.success(
                f"Loaded preset profile with {len(dataset_tests)} tests "
                f"from {len(dataset_metadata.get('preset_pack_ids', []))} packs"
            )
        except Exception as exc:
            dataset_tests = []
            dataset_issues = [str(exc)]
            st.error(f"Failed to load preset profile: {exc}")
    elif dataset_source == "Local Registry Packs":
        registry = build_local_dataset_registry()
        packs = registry.get("packs", [])
        filtered_packs = _render_local_registry_pack_picker(packs)
        if filtered_packs:
            pack_options = {_registry_pack_label(pack): pack for pack in filtered_packs}
            composition_mode = st.checkbox(
                "Compose multiple packs",
                value=False,
                key="dataset_registry_compose_mode",
                help="Merge several registry packs into one benchmark dataset with ID de-duplication.",
            )
            try:
                if composition_mode:
                    selected_pack_labels = st.multiselect(
                        "Registry Packs",
                        list(pack_options.keys()),
                        default=list(pack_options.keys())[: min(3, len(pack_options))],
                        key="dataset_registry_pack_multiselect",
                    )
                    if selected_pack_labels:
                        bundles = [
                            load_dataset_bundle(pack_options[label]["path"])
                            for label in selected_pack_labels
                        ]
                        dataset_bundle = compose_dataset_bundles(bundles)
                        dataset_path = dataset_bundle.path
                        dataset_tests = dataset_bundle.tests
                        dataset_metadata = dataset_bundle.metadata
                        dataset_label = dataset_metadata.get(
                            "name", "Composed Registry Pack"
                        )
                        st.success(
                            f"Composed {len(bundles)} packs into {len(dataset_tests)} tests "
                            f"from {dataset_metadata.get('source_pack_count', len(bundles))} source packs"
                        )
                    else:
                        st.info("Select one or more registry packs to compose.")
                else:
                    selected_pack_label = st.selectbox(
                        "Registry Pack",
                        list(pack_options.keys()),
                        key="dataset_registry_pack_select",
                    )
                    selected_pack = pack_options[selected_pack_label]
                    selected_pack_path = selected_pack["path"]
                    dataset_bundle = load_dataset_bundle(selected_pack_path)
                    dataset_path = selected_pack_path
                    dataset_tests = dataset_bundle.tests
                    dataset_metadata = dataset_bundle.metadata
                    dataset_label = dataset_metadata.get(
                        "name", Path(selected_pack_path).stem
                    )
                    st.success(f"Loaded {len(dataset_tests)} tests from registry pack")
            except Exception as exc:
                dataset_tests = []
                dataset_issues = [str(exc)]
                st.error(f"Failed to load registry pack: {exc}")
        else:
            st.info("No registry packs match the current filters.")
    else:
        uploaded_dataset = st.file_uploader(
            "Upload dataset",
            type=["json", "jsonl", "csv"],
            help="Upload a custom benchmark pack in JSON, JSONL, or CSV format.",
        )
        if uploaded_dataset is not None:
            dataset_label = uploaded_dataset.name
            try:
                dataset_tests, dataset_issues, dataset_metadata = load_uploaded_dataset(
                    uploaded_dataset
                )
                dataset_label = dataset_metadata.get("name", dataset_label)
                if dataset_issues:
                    st.warning(
                        f"Loaded {len(dataset_tests)} rows with validation issues"
                    )
                else:
                    st.success(f"Loaded {len(dataset_tests)} custom tests")
            except Exception as exc:
                dataset_tests = []
                dataset_issues = [str(exc)]
                st.error(f"Failed to parse uploaded dataset: {exc}")
        else:
            st.info("Upload a custom pack to benchmark your own attack set.")

    if dataset_tests:
        category_count = len(
            {test.get("universal_category", "unknown") for test in dataset_tests}
        )
        st.caption(
            f"{category_count} categories · "
            f"{dataset_metadata.get('name', dataset_label)} v{dataset_metadata.get('version', '1.0')}"
        )
        if dataset_metadata.get("description"):
            st.caption(dataset_metadata["description"])
        with st.expander("Dataset Preview"):
            preview_rows = pd.DataFrame(dataset_tests[:10])[
                ["id", "category", "universal_category", "should_refuse"]
            ]
            st.dataframe(preview_rows, use_container_width=True)
            st.json(dataset_metadata)
        if dataset_issues:
            with st.expander("Dataset Validation Issues"):
                for issue in dataset_issues:
                    st.write(f"- {issue}")

    return {
        "dataset_source": dataset_source,
        "dataset_label": dataset_label,
        "dataset_path": dataset_path,
        "dataset_tests": dataset_tests,
        "dataset_issues": dataset_issues,
        "dataset_metadata": dataset_metadata,
    }


def _registry_pack_label(pack: dict) -> str:
    metadata = pack.get("metadata", {})
    return (
        f"{metadata.get('name', Path(pack['path']).stem)} · "
        f"{metadata.get('attack_family', 'unknown')} · "
        f"{metadata.get('difficulty', 'mixed')} · "
        f"{pack.get('num_tests', 0)} tests"
    )


def _render_local_registry_pack_picker(packs: list[dict]) -> list[dict]:
    with st.expander("Registry Pack Filters", expanded=True):
        families = sorted(
            {
                str(pack.get("metadata", {}).get("attack_family", "")).strip()
                for pack in packs
                if str(pack.get("metadata", {}).get("attack_family", "")).strip()
            }
        )
        difficulties = sorted(
            {
                str(pack.get("metadata", {}).get("difficulty", "")).strip()
                for pack in packs
                if str(pack.get("metadata", {}).get("difficulty", "")).strip()
            }
        )
        modalities = sorted(
            {
                str(pack.get("metadata", {}).get("modality", "")).strip()
                for pack in packs
                if str(pack.get("metadata", {}).get("modality", "")).strip()
            }
        )
        threat_models = sorted(
            {
                str(pack.get("metadata", {}).get("threat_model", "")).strip()
                for pack in packs
                if str(pack.get("metadata", {}).get("threat_model", "")).strip()
            }
        )
        languages = sorted(
            {
                str(pack.get("metadata", {}).get("language", "")).strip()
                for pack in packs
                if str(pack.get("metadata", {}).get("language", "")).strip()
            }
        )

        cols = st.columns([1.4, 1, 1, 1, 1])
        with cols[0]:
            search_query = st.text_input(
                "Search Packs",
                key="dataset_registry_search",
                placeholder="name, tag, id, path",
            )
        with cols[1]:
            selected_families = st.multiselect(
                "Attack Family", families, key="dataset_registry_families"
            )
        with cols[2]:
            selected_difficulties = st.multiselect(
                "Difficulty", difficulties, key="dataset_registry_difficulties"
            )
        with cols[3]:
            selected_modalities = st.multiselect(
                "Modality", modalities, key="dataset_registry_modalities"
            )
        with cols[4]:
            selected_languages = st.multiselect(
                "Language", languages, key="dataset_registry_languages"
            )
        selected_threat_models = st.multiselect(
            "Threat Model", threat_models, key="dataset_registry_threat_models"
        )

        filtered = _filter_registry_packs(
            packs,
            search_query=search_query,
            attack_families=selected_families,
            difficulties=selected_difficulties,
            modalities=selected_modalities,
            threat_models=selected_threat_models,
            languages=selected_languages,
        )
        if filtered:
            preview = pd.DataFrame(
                [
                    {
                        "name": pack.get("metadata", {}).get(
                            "name", Path(pack["path"]).stem
                        ),
                        "attack_family": pack.get("metadata", {}).get(
                            "attack_family", ""
                        ),
                        "difficulty": pack.get("metadata", {}).get("difficulty", ""),
                        "modality": pack.get("metadata", {}).get("modality", ""),
                        "threat_model": pack.get("metadata", {}).get(
                            "threat_model", ""
                        ),
                        "language": pack.get("metadata", {}).get("language", ""),
                        "num_tests": pack.get("num_tests", 0),
                        "path": pack.get("path", ""),
                    }
                    for pack in filtered
                ]
            )
            st.dataframe(preview, use_container_width=True, hide_index=True)
        return filtered


def _filter_registry_packs(
    packs: list[dict],
    search_query: str = "",
    attack_families: list[str] | None = None,
    difficulties: list[str] | None = None,
    modalities: list[str] | None = None,
    threat_models: list[str] | None = None,
    languages: list[str] | None = None,
) -> list[dict]:
    attack_families = attack_families or []
    difficulties = difficulties or []
    modalities = modalities or []
    threat_models = threat_models or []
    languages = languages or []
    query = (search_query or "").strip().lower()
    filtered = []
    for pack in packs:
        metadata = pack.get("metadata", {})
        if attack_families and metadata.get("attack_family") not in attack_families:
            continue
        if difficulties and metadata.get("difficulty") not in difficulties:
            continue
        if modalities and metadata.get("modality") not in modalities:
            continue
        if threat_models and metadata.get("threat_model") not in threat_models:
            continue
        if languages and metadata.get("language") not in languages:
            continue
        if query:
            haystack = " ".join(
                [
                    str(metadata.get("name", "")),
                    str(metadata.get("registry_id", "")),
                    str(metadata.get("attack_family", "")),
                    str(metadata.get("threat_model", "")),
                    str(metadata.get("language", "")),
                    str(metadata.get("difficulty", "")),
                    str(" ".join(metadata.get("tags", []))),
                    str(pack.get("path", "")),
                ]
            ).lower()
            if query not in haystack:
                continue
        filtered.append(pack)
    return sorted(
        filtered,
        key=lambda pack: (
            str(pack.get("metadata", {}).get("attack_family", "")),
            str(pack.get("metadata", {}).get("name", "")),
        ),
    )


def render_build_pack_view(
    dataset_tests: list[dict], dataset_metadata: dict, dataset_label: str
) -> dict | None:
    st.subheader("Custom Pack Builder")
    if not dataset_tests:
        st.info("Load a dataset first to generate a custom pack.")
        return None

    editable_tests = _render_dataset_row_editor(dataset_tests)

    selected_transforms = st.multiselect(
        "Transforms",
        available_transforms(),
        help="Generate additional attack variants from the currently loaded dataset.",
        key="builder_transforms",
    )
    chain_transforms = st.checkbox(
        "Chain transforms",
        value=False,
        help="If enabled, each transform is applied to the output of the previous one.",
        key="builder_chain",
    )
    builder_cols = st.columns(2)
    with builder_cols[0]:
        start_id = st.number_input(
            "Generated Start ID",
            min_value=0,
            value=int(max((test["id"] for test in editable_tests), default=0) + 1),
            step=1,
            key="builder_start_id",
        )
    with builder_cols[1]:
        id_step = st.number_input(
            "Generated ID Step",
            min_value=1,
            value=1,
            step=1,
            key="builder_id_step",
        )

    generated_tests = list(editable_tests)
    if selected_transforms:
        generated_tests = apply_transforms(
            editable_tests,
            selected_transforms,
            start_id=int(start_id),
            id_step=int(id_step),
            chain=chain_transforms,
        )

    success_criteria = _render_success_criteria_editor(editable_tests, generated_tests)
    if success_criteria:
        generated_tests = _apply_success_criteria(
            editable_tests, generated_tests, success_criteria
        )

    if not selected_transforms and not success_criteria:
        st.info(
            "Select transforms or define success criteria to generate a custom pack."
        )
        return None

    generated_metadata = {
        **dataset_metadata,
        "name": _generated_pack_name(
            dataset_metadata, dataset_label, selected_transforms, success_criteria
        ),
        "transforms": selected_transforms,
        "transform_mode": "chain" if chain_transforms else "parallel",
        "num_tests": len(generated_tests),
    }
    if success_criteria:
        generated_metadata["success_criteria_template"] = success_criteria
    generated_label = generated_metadata["name"]
    generated_count = len(generated_tests) - len(editable_tests)
    if generated_count > 0:
        st.caption(f"Builder would add {generated_count} generated tests")
    elif success_criteria:
        st.caption(
            "Builder would keep the current test count and attach success criteria."
        )
    preview_rows = pd.DataFrame(generated_tests[-min(10, max(generated_count, 1)) :])[
        ["id", "category", "universal_category", "should_refuse"]
    ]
    st.dataframe(preview_rows, use_container_width=True)

    export_payload = json.dumps(
        {"metadata": generated_metadata, "tests": generated_tests},
        indent=2,
        ensure_ascii=False,
    )
    st.download_button(
        " Download Generated Pack (JSON)",
        export_payload,
        file_name=f"{generated_label.lower().replace(' ', '_')}.json",
        mime="application/json",
        key="download_generated_pack",
    )
    save_name = st.text_input(
        "Save generated pack as",
        value=f"{generated_label.lower().replace(' ', '_')}.json",
        key="generated_pack_filename",
    )
    if st.button(" Save Generated Pack To Workspace", key="save_generated_pack"):
        save_path = Path("datasets/generated") / save_name
        export_dataset(
            generated_tests,
            str(save_path),
            metadata=generated_metadata,
        )
        st.success(f"Saved generated pack to {save_path}")

    generated_pack = {
        "label": generated_label,
        "metadata": generated_metadata,
        "tests": generated_tests,
    }
    if st.button("Use Generated Pack For This Run", key="use_generated_pack"):
        return generated_pack
    return None


def _render_success_criteria_editor(
    dataset_tests: list[dict], generated_tests: list[dict]
) -> dict | None:
    with st.expander("Success Criteria", expanded=False):
        enabled = st.checkbox(
            "Attach custom success criteria",
            value=False,
            key="builder_success_criteria_enabled",
            help="Define assertions that a response must satisfy in addition to model-judge scoring.",
        )
        if not enabled:
            return None

        scope_options = [
            "All Tests",
            "Generated Tests Only",
            "By Category",
            "By Test IDs",
        ]
        scope = st.selectbox("Apply To", scope_options, key="builder_success_scope")
        category_value = None
        selected_test_ids = []
        if scope == "By Category":
            categories = sorted(
                {test.get("universal_category", "unknown") for test in generated_tests}
            )
            category_value = st.selectbox(
                "Category", categories, key="builder_success_scope_category"
            )
        elif scope == "By Test IDs":
            selected_test_ids = _parse_test_id_selection(
                st.text_input(
                    "Test IDs",
                    key="builder_success_scope_ids",
                    placeholder="1,2,5-8,100",
                    help="Comma-separated IDs and ranges for bulk editing existing rows.",
                )
            )

        clear_existing = st.checkbox(
            "Clear existing success criteria in target scope before applying",
            value=False,
            key="builder_success_clear_existing",
        )

        operator = st.radio(
            "Assertion Operator",
            ["all", "any"],
            horizontal=True,
            key="builder_success_operator",
            help="`all` means every assertion must pass. `any` means at least one assertion must pass.",
        )
        assertion_count = st.number_input(
            "Assertions",
            min_value=1,
            max_value=5,
            value=1,
            step=1,
            key="builder_success_count",
        )

        assertions = []
        for index in range(int(assertion_count)):
            st.markdown(f"**Assertion {index + 1}**")
            cols = st.columns([1.3, 1.7, 1])
            with cols[0]:
                assertion_type = st.selectbox(
                    "Type",
                    [
                        "contains",
                        "not_contains",
                        "regex",
                        "not_regex",
                        "equals",
                        "not_equals",
                    ],
                    key=f"builder_assertion_type_{index}",
                )
            with cols[1]:
                value = st.text_input(
                    "Value",
                    key=f"builder_assertion_value_{index}",
                    placeholder="SAFE",
                )
            with cols[2]:
                field = st.selectbox(
                    "Field",
                    ["response", "input", "category", "expected_behavior"],
                    key=f"builder_assertion_field_{index}",
                )
            flags = ""
            if "regex" in assertion_type:
                flags = st.text_input(
                    "Regex Flags",
                    key=f"builder_assertion_flags_{index}",
                    placeholder="i",
                )
            description = st.text_input(
                "Description",
                key=f"builder_assertion_description_{index}",
                placeholder="Response must include SAFE",
            )
            if value:
                assertions.append(
                    {
                        "type": assertion_type,
                        "value": value,
                        "field": field,
                        "flags": flags,
                        "description": description,
                    }
                )

        if not assertions:
            st.warning("Add at least one assertion value to attach success criteria.")
            return None
        return {
            "operator": operator,
            "assertions": assertions,
            "_scope": scope,
            "_category": category_value,
            "_test_ids": selected_test_ids,
            "_clear_existing": clear_existing,
            "_source_count": len(dataset_tests),
        }


def _apply_success_criteria(
    dataset_tests: list[dict], generated_tests: list[dict], criteria: dict
) -> list[dict]:
    applied = []
    source_count = criteria.get("_source_count", len(dataset_tests))
    scope = criteria.get("_scope", "All Tests")
    category_value = criteria.get("_category")
    selected_test_ids = set(criteria.get("_test_ids", []))
    clear_existing = bool(criteria.get("_clear_existing"))
    criteria_payload = {
        "operator": criteria["operator"],
        "assertions": criteria["assertions"],
    }

    for index, test in enumerate(generated_tests):
        test_copy = deepcopy(test)
        should_apply = False
        if scope == "All Tests":
            should_apply = True
        elif scope == "Generated Tests Only":
            should_apply = index >= source_count
        elif scope == "By Category":
            should_apply = test_copy.get("universal_category") == category_value
        elif scope == "By Test IDs":
            should_apply = int(test_copy.get("id", -1)) in selected_test_ids
        if should_apply and clear_existing:
            test_copy.pop("success_criteria", None)
        if should_apply:
            test_copy["success_criteria"] = criteria_payload
        applied.append(test_copy)
    return applied


def _generated_pack_name(
    dataset_metadata: dict,
    dataset_label: str,
    transforms: list[str],
    success_criteria: dict | None,
) -> str:
    base_name = dataset_metadata.get("name", dataset_label)
    suffixes = []
    if transforms:
        suffixes.append("transforms")
    if success_criteria:
        suffixes.append("criteria")
    if not suffixes:
        return base_name
    return f"{base_name} + {' + '.join(suffixes)}"


def _parse_test_id_selection(raw: str | None) -> list[int]:
    if not raw:
        return []
    selected = set()
    for part in raw.split(","):
        item = part.strip()
        if not item:
            continue
        if "-" in item:
            start, end = item.split("-", 1)
            start_id = int(start.strip())
            end_id = int(end.strip())
            for value in range(min(start_id, end_id), max(start_id, end_id) + 1):
                selected.add(value)
        else:
            selected.add(int(item))
    return sorted(selected)


def _render_dataset_row_editor(dataset_tests: list[dict]) -> list[dict]:
    with st.expander("Dataset Row Editor", expanded=False):
        st.caption(
            "Edit existing dataset rows, filter them, and apply bulk mutations before transforms or success criteria."
        )
        state_key = "dataset_row_editor_rows"
        signature_key = "dataset_row_editor_signature"
        current_signature = tuple(
            sorted(int(row.get("id", -1)) for row in dataset_tests)
        )
        if st.session_state.get(signature_key) != current_signature:
            st.session_state[signature_key] = current_signature
            st.session_state[state_key] = _table_rows_from_tests(dataset_tests)

        table_rows = list(
            st.session_state.get(state_key, _table_rows_from_tests(dataset_tests))
        )
        filtered_rows = _render_row_editor_filters(table_rows)
        table_rows = _render_row_editor_bulk_actions(table_rows, filtered_rows)

        filtered_ids = {int(row["id"]) for row in filtered_rows}
        filtered_view = [row for row in table_rows if int(row["id"]) in filtered_ids]
        edited_df = st.data_editor(
            pd.DataFrame(filtered_view),
            use_container_width=True,
            num_rows="dynamic",
            key="dataset_row_editor",
            column_config={
                "id": st.column_config.NumberColumn("ID", min_value=0, step=1),
                "should_refuse": st.column_config.CheckboxColumn("Should Refuse"),
                "input": st.column_config.TextColumn("Input", width="large"),
                "success_criteria_json": st.column_config.TextColumn(
                    "Success Criteria (JSON)", width="large"
                ),
            },
        )
        try:
            table_rows = _merge_partial_editor_rows(
                table_rows, edited_df.to_dict("records")
            )
            st.session_state[state_key] = table_rows
            merged = _merge_editor_rows(dataset_tests, table_rows)
            st.caption(
                f"{len(merged)} editable rows active for this run/build. "
                f"Filtered rows: {len(filtered_view)} / {len(table_rows)}."
            )
            return merged
        except Exception as exc:
            st.error(f"Invalid row edits: {exc}")
            return list(dataset_tests)


def _table_rows_from_tests(dataset_tests: list[dict]) -> list[dict]:
    table_rows = []
    for row in dataset_tests:
        criteria = row.get("success_criteria")
        table_rows.append(
            {
                "id": int(row.get("id", 0)),
                "category": row.get("category", ""),
                "universal_category": row.get("universal_category", ""),
                "should_refuse": bool(row.get("should_refuse", False)),
                "expected_behavior": row.get("expected_behavior", ""),
                "input": row.get("input", ""),
                "success_criteria_json": json.dumps(criteria, ensure_ascii=False)
                if criteria
                else "",
            }
        )
    return table_rows


def _render_row_editor_filters(table_rows: list[dict]) -> list[dict]:
    categories = sorted(
        {
            str(row.get("category", "")).strip()
            for row in table_rows
            if str(row.get("category", "")).strip()
        }
    )
    universal_categories = sorted(
        {
            str(row.get("universal_category", "")).strip()
            for row in table_rows
            if str(row.get("universal_category", "")).strip()
        }
    )
    filter_cols = st.columns([1.4, 1.2, 1.2, 1, 1])
    with filter_cols[0]:
        search_query = st.text_input(
            "Search", key="dataset_editor_search", placeholder="input/category/behavior"
        )
    with filter_cols[1]:
        selected_categories = st.multiselect(
            "Category", categories, key="dataset_editor_filter_category"
        )
    with filter_cols[2]:
        selected_universal = st.multiselect(
            "Universal", universal_categories, key="dataset_editor_filter_universal"
        )
    with filter_cols[3]:
        refuse_filter = st.selectbox(
            "Should Refuse",
            ["All", "True", "False"],
            key="dataset_editor_filter_refuse",
        )
    with filter_cols[4]:
        sort_by = st.selectbox(
            "Sort By",
            ["id", "category", "universal_category", "expected_behavior"],
            key="dataset_editor_sort_by",
        )
    descending = st.checkbox("Descending", value=False, key="dataset_editor_sort_desc")
    return _filter_sort_editor_rows(
        table_rows,
        search_query=search_query,
        categories=selected_categories,
        universal_categories=selected_universal,
        refuse_filter=refuse_filter,
        sort_by=sort_by,
        descending=descending,
    )


def _render_row_editor_bulk_actions(
    table_rows: list[dict], filtered_rows: list[dict]
) -> list[dict]:
    with st.expander("Bulk Actions", expanded=False):
        st.caption(
            f"Apply changes to the current filtered slice: {len(filtered_rows)} rows."
        )
        cols = st.columns(4)
        with cols[0]:
            bulk_refuse = st.selectbox(
                "Set Should Refuse",
                ["No Change", "True", "False"],
                key="dataset_editor_bulk_refuse",
            )
        with cols[1]:
            bulk_expected_behavior = st.text_input(
                "Expected Behavior",
                key="dataset_editor_bulk_behavior",
                placeholder="answer/refuse",
            )
        with cols[2]:
            bulk_category = st.text_input(
                "Category",
                key="dataset_editor_bulk_category",
                placeholder="optional overwrite",
            )
        with cols[3]:
            bulk_universal_category = st.text_input(
                "Universal Category",
                key="dataset_editor_bulk_universal_category",
                placeholder="optional overwrite",
            )
        bulk_clear_criteria = st.checkbox(
            "Clear success criteria",
            value=False,
            key="dataset_editor_bulk_clear_criteria",
        )
        bulk_criteria_json = st.text_area(
            "Success Criteria JSON",
            key="dataset_editor_bulk_criteria_json",
            placeholder='{"operator":"all","assertions":[{"type":"contains","field":"response","value":"SAFE"}]}',
            height=120,
        )
        if st.button(
            "Apply Bulk Actions To Filtered Rows", key="dataset_editor_apply_bulk"
        ):
            try:
                updated_rows = _apply_bulk_actions(
                    table_rows,
                    filtered_ids={int(row["id"]) for row in filtered_rows},
                    bulk_refuse=bulk_refuse,
                    bulk_expected_behavior=bulk_expected_behavior,
                    bulk_category=bulk_category,
                    bulk_universal_category=bulk_universal_category,
                    bulk_clear_criteria=bulk_clear_criteria,
                    bulk_criteria_json=bulk_criteria_json,
                )
                st.session_state["dataset_row_editor_rows"] = updated_rows
                st.success(f"Applied bulk changes to {len(filtered_rows)} rows.")
                return updated_rows
            except Exception as exc:
                st.error(f"Bulk action failed: {exc}")
    return table_rows


def _filter_sort_editor_rows(
    table_rows: list[dict],
    search_query: str = "",
    categories: list[str] | None = None,
    universal_categories: list[str] | None = None,
    refuse_filter: str = "All",
    sort_by: str = "id",
    descending: bool = False,
) -> list[dict]:
    filtered = []
    categories = categories or []
    universal_categories = universal_categories or []
    search_query = (search_query or "").strip().lower()
    for row in table_rows:
        if categories and row.get("category") not in categories:
            continue
        if (
            universal_categories
            and row.get("universal_category") not in universal_categories
        ):
            continue
        if refuse_filter == "True" and not bool(row.get("should_refuse")):
            continue
        if refuse_filter == "False" and bool(row.get("should_refuse")):
            continue
        if search_query:
            haystack = " ".join(
                [
                    str(row.get("category", "")),
                    str(row.get("universal_category", "")),
                    str(row.get("expected_behavior", "")),
                    str(row.get("input", "")),
                ]
            ).lower()
            if search_query not in haystack:
                continue
        filtered.append(row)

    def _sort_key(item: dict) -> tuple:
        val = item.get(sort_by)
        if val is None:
            return (1, 0, "")
        if isinstance(val, (int, float)):
            return (0, val, "")
        return (0, 0, str(val))

    return sorted(filtered, key=_sort_key, reverse=descending)


def _apply_bulk_actions(
    table_rows: list[dict],
    filtered_ids: set[int],
    bulk_refuse: str,
    bulk_expected_behavior: str,
    bulk_category: str,
    bulk_universal_category: str,
    bulk_clear_criteria: bool,
    bulk_criteria_json: str,
) -> list[dict]:
    parsed_criteria = (
        normalize_success_criteria(bulk_criteria_json)
        if bulk_criteria_json.strip()
        else None
    )
    updated_rows = []
    for row in table_rows:
        row_copy = deepcopy(row)
        if int(row_copy.get("id", -1)) in filtered_ids:
            if bulk_refuse == "True":
                row_copy["should_refuse"] = True
            elif bulk_refuse == "False":
                row_copy["should_refuse"] = False
            if bulk_expected_behavior.strip():
                row_copy["expected_behavior"] = bulk_expected_behavior.strip()
            elif bulk_refuse in {"True", "False"}:
                row_copy["expected_behavior"] = (
                    "refuse" if row_copy["should_refuse"] else "answer"
                )
            if bulk_category.strip():
                row_copy["category"] = bulk_category.strip()
            if bulk_universal_category.strip():
                row_copy["universal_category"] = bulk_universal_category.strip()
            if bulk_clear_criteria:
                row_copy["success_criteria_json"] = ""
            if parsed_criteria:
                row_copy["success_criteria_json"] = json.dumps(
                    parsed_criteria, ensure_ascii=False
                )
        updated_rows.append(row_copy)
    return updated_rows


def _merge_partial_editor_rows(
    original_rows: list[dict], edited_rows: list[dict]
) -> list[dict]:
    edited_by_id = {int(row.get("id", -1)): row for row in edited_rows}
    merged_rows = []
    for row in original_rows:
        row_id = int(row.get("id", -1))
        if row_id in edited_by_id:
            merged_rows.append({**row, **edited_by_id[row_id]})
        else:
            merged_rows.append(row)
    for row_id, row in edited_by_id.items():
        if row_id not in {int(existing.get("id", -1)) for existing in original_rows}:
            merged_rows.append(row)
    return merged_rows


def _merge_editor_rows(
    original_rows: list[dict], edited_rows: list[dict]
) -> list[dict]:
    merged = []
    original_by_id = {int(row.get("id", -1)): row for row in original_rows}
    for index, edited in enumerate(edited_rows):
        row_id = int(edited.get("id", index))
        base = deepcopy(original_by_id.get(row_id, {}))
        base["id"] = row_id
        base["category"] = str(edited.get("category", "")).strip()
        base["universal_category"] = str(
            edited.get("universal_category", "")
        ).strip() or base.get("universal_category", "")
        base["should_refuse"] = bool(edited.get("should_refuse", False))
        base["expected_behavior"] = str(
            edited.get("expected_behavior", "")
        ).strip() or ("refuse" if base["should_refuse"] else "answer")
        base["input"] = str(edited.get("input", ""))
        criteria_raw = str(edited.get("success_criteria_json", "") or "").strip()
        if criteria_raw:
            base["success_criteria"] = normalize_success_criteria(criteria_raw)
        else:
            base.pop("success_criteria", None)
        merged.append(base)
    return merged
