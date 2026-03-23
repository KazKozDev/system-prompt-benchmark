"""Administrative preset and dataset management views."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.datasets import (
    apply_transforms,
    available_transforms,
    build_local_dataset_registry,
    compose_dataset_preset,
    evaluate_remote_catalog_status,
    export_dataset,
    fetch_remote_dataset_catalog,
    list_dataset_presets,
    load_dataset_bundle,
    save_local_dataset_registry,
    sync_remote_dataset_catalog,
    validate_dataset_file,
)
from src.ui.datasets import load_uploaded_dataset


def render_presets_datasets_tab() -> None:
    """Render presets and dataset management tools."""
    st.write(
        "Preset composition, dataset validation, conversion, and registry "
        "tools."
    )

    tabs = st.tabs(
        ["Presets", "Validate", "Convert", "Registry", "Remote Catalog"]
    )
    with tabs[0]:
        _render_presets_view()
    with tabs[1]:
        _render_validate_view()
    with tabs[2]:
        _render_convert_view()
    with tabs[3]:
        _render_registry_view()
    with tabs[4]:
        _render_remote_catalog_view()


def _render_presets_view() -> None:
    presets = list_dataset_presets()
    if not presets:
        st.info("No dataset presets are registered.")
        return

    preset_options = list(presets.keys())
    selected_preset = st.selectbox(
        "Preset",
        options=preset_options,
        format_func=lambda key: f"{presets[key]['name']} · {key}",
        key="admin_preset_selected",
    )
    preset = presets[selected_preset]
    st.caption(preset.get("description", ""))
    st.caption("Includes: " + ", ".join(preset.get("pack_ids", [])))

    try:
        bundle = compose_dataset_preset(selected_preset)
    except Exception as exc:
        st.error(f"Failed to compose preset: {exc}")
        return

    metric_cols = st.columns(3)
    with metric_cols[0]:
        st.metric("Tests", len(bundle.tests))
    with metric_cols[1]:
        st.metric("Version", bundle.metadata.get("version", "1.0"))
    with metric_cols[2]:
        st.metric("Packs", len(bundle.metadata.get("preset_pack_ids", [])))

    preview_columns = "id category universal_category should_refuse".split()
    st.dataframe(
        pd.DataFrame(bundle.tests[:10])[preview_columns],
        width="stretch",
        hide_index=True,
    )
    with st.expander("Preset Metadata", expanded=False):
        st.json(bundle.metadata)

    export_cols = st.columns([1, 1, 2])
    with export_cols[0]:
        output_format = st.selectbox(
            "Format",
            options=["json", "jsonl", "csv"],
            key="admin_preset_export_format",
        )
    with export_cols[1]:
        default_name = f"{selected_preset}.{output_format}"
        output_name = st.text_input(
            "Filename",
            value=default_name,
            key="admin_preset_output_name",
        )
    with export_cols[2]:
        output_path = Path("datasets/generated") / output_name
        st.caption(f"Save path: {output_path}")

    if st.button("Save Preset Dataset", key="admin_preset_save"):
        saved_path = export_dataset(
            bundle.tests,
            str(output_path),
            file_format=output_format,
            metadata=bundle.metadata,
        )
        st.success(f"Preset exported to {saved_path}")

    download_payload = json.dumps(
        {"metadata": bundle.metadata, "tests": bundle.tests},
        indent=2,
        ensure_ascii=False,
    )
    st.download_button(
        "Download Preset JSON",
        data=download_payload,
        file_name=f"{selected_preset}.json",
        mime="application/json",
        key="admin_preset_download_json",
    )


def _render_validate_view() -> None:
    source = st.radio(
        "Validation Source",
        options=["Path", "Upload"],
        horizontal=True,
        key="admin_dataset_validate_source",
    )
    if source == "Path":
        dataset_path = st.text_input(
            "Dataset Path",
            value="tests/safeprompt-benchmark-v2.json",
            key="admin_dataset_validate_path",
        ).strip()
        dataset_format = st.selectbox(
            "Format Override",
            options=["auto", "json", "jsonl", "csv"],
            key="admin_dataset_validate_format",
        )
        if st.button(
            "Validate Dataset Path",
            key="admin_dataset_validate_button",
        ):
            try:
                rows, issues = validate_dataset_file(
                    dataset_path,
                    file_format=(
                        None if dataset_format == "auto" else dataset_format
                    ),
                )
                _render_validation_result(rows, issues, dataset_path)
            except Exception as exc:
                st.error(f"Dataset validation failed: {exc}")
    else:
        uploaded_file = st.file_uploader(
            "Upload Dataset",
            type=["json", "jsonl", "csv"],
            key="admin_dataset_upload_validate",
        )
        if uploaded_file is not None:
            try:
                rows, issues, metadata = load_uploaded_dataset(uploaded_file)
                _render_validation_result(rows, issues, uploaded_file.name)
                with st.expander("Uploaded Metadata", expanded=False):
                    st.json(metadata)
            except Exception as exc:
                st.error(f"Uploaded dataset is invalid: {exc}")


def _render_convert_view() -> None:
    input_path = st.text_input(
        "Input Dataset",
        value="tests/safeprompt-benchmark-v2.json",
        key="admin_dataset_convert_input",
    ).strip()
    transform_names = st.multiselect(
        "Transforms",
        options=available_transforms(),
        key="admin_dataset_convert_transforms",
    )
    chain_transforms = st.checkbox(
        "Chain transforms",
        value=False,
        key="admin_dataset_convert_chain",
    )
    config_cols = st.columns(3)
    with config_cols[0]:
        output_format = st.selectbox(
            "Output Format",
            options=["json", "jsonl", "csv"],
            key="admin_dataset_convert_output_format",
        )
    with config_cols[1]:
        start_id = st.number_input(
            "Start ID",
            min_value=0,
            value=100000,
            step=1,
            key="admin_dataset_convert_start_id",
        )
    with config_cols[2]:
        id_step = st.number_input(
            "ID Step",
            min_value=1,
            value=1,
            step=1,
            key="admin_dataset_convert_id_step",
        )
    output_name = st.text_input(
        "Output Filename",
        value=f"converted.{output_format}",
        key="admin_dataset_convert_output_name",
    )

    if st.button("Convert Dataset", key="admin_dataset_convert_button"):
        try:
            bundle = load_dataset_bundle(input_path)
            transformed_rows = apply_transforms(
                bundle.tests,
                transform_names,
                start_id=int(start_id),
                id_step=int(id_step),
                chain=chain_transforms,
            )
            metadata = {
                **bundle.metadata,
                "transforms": transform_names,
                "transform_mode": "chain" if chain_transforms else "parallel",
                "num_tests": len(transformed_rows),
            }
            output_path = Path("datasets/generated") / output_name
            saved_path = export_dataset(
                transformed_rows,
                str(output_path),
                file_format=output_format,
                metadata=metadata,
            )
            st.success(
                f"Converted dataset saved to {saved_path} "
                f"({len(transformed_rows)} rows)."
            )
            preview_columns = "id category transform source_id".split()
            st.dataframe(
                pd.DataFrame(transformed_rows[:10])[preview_columns],
                width="stretch",
                hide_index=True,
            )
        except Exception as exc:
            st.error(f"Dataset conversion failed: {exc}")


def _render_registry_view() -> None:
    roots_text = st.text_input(
        "Registry Roots",
        value="datasets,tests",
        key="admin_dataset_registry_roots",
        help="Comma-separated root directories to scan.",
    )
    include_tests_dir = st.checkbox(
        "Include tests/ directory",
        value=True,
        key="admin_dataset_registry_include_tests",
    )
    roots = [item.strip() for item in roots_text.split(",") if item.strip()]

    if st.button("Build Registry", key="admin_dataset_registry_build"):
        registry = build_local_dataset_registry(
            roots=roots or None,
            include_tests_dir=include_tests_dir,
        )
        st.session_state["admin_dataset_registry_payload"] = registry

    registry = st.session_state.get("admin_dataset_registry_payload")
    if not registry:
        return

    packs = registry.get("packs", [])
    if packs:
        rows = []
        for pack in packs:
            metadata = pack.get("metadata", {})
            rows.append(
                {
                    "id": pack.get("id"),
                    "name": metadata.get("name", pack.get("id")),
                    "tests": pack.get("num_tests", 0),
                    "format": pack.get("format"),
                    "path": pack.get("path"),
                    "version": metadata.get("version", "1.0"),
                    "attack_family": metadata.get("attack_family", ""),
                }
            )
        st.dataframe(
            pd.DataFrame(rows),
            width="stretch",
            hide_index=True,
        )
    else:
        st.info("Registry is empty.")

    output_path = st.text_input(
        "Registry Output Path",
        value="datasets/registry.json",
        key="admin_dataset_registry_output_path",
    )
    if st.button("Save Registry JSON", key="admin_dataset_registry_save"):
        saved_path = save_local_dataset_registry(
            output_path,
            roots=roots or None,
            include_tests_dir=include_tests_dir,
        )
        st.success(f"Registry saved to {saved_path}")

    with st.expander("Raw Registry JSON", expanded=False):
        st.json(registry)


def _render_remote_catalog_view() -> None:
    url_cols = st.columns([2.5, 1])
    with url_cols[0]:
        catalog_url = st.text_input(
            "Catalog URL",
            placeholder="https://example.com/catalog.json",
            key="admin_dataset_catalog_url",
        ).strip()
    with url_cols[1]:
        timeout_seconds = st.number_input(
            "Timeout",
            min_value=1.0,
            value=20.0,
            step=1.0,
            key="admin_dataset_catalog_timeout",
        )

    if st.button("Fetch Catalog", key="admin_dataset_catalog_fetch"):
        if not catalog_url:
            st.warning("Enter a remote catalog URL first.")
        else:
            try:
                catalog = fetch_remote_dataset_catalog(
                    catalog_url,
                    timeout_seconds=float(timeout_seconds),
                )
                st.session_state["admin_dataset_catalog_payload"] = catalog
            except Exception as exc:
                st.error(f"Failed to fetch remote catalog: {exc}")

    catalog = st.session_state.get("admin_dataset_catalog_payload")
    if not catalog:
        return

    destination_dir = st.text_input(
        "Destination Directory",
        value="datasets/remote",
        key="admin_dataset_catalog_destination",
    )
    sync_force = st.checkbox(
        "Overwrite existing packs",
        value=False,
        key="admin_dataset_catalog_force",
    )
    allow_untrusted = st.checkbox(
        "Allow packs without sha256",
        value=False,
        key="admin_dataset_catalog_allow_untrusted",
    )

    statuses = evaluate_remote_catalog_status(
        catalog,
        destination_dir=destination_dir,
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
            width="stretch",
            hide_index=True,
        )

    pack_options = {
        f"{item['id']} · {item['name']}": item["id"]
        for item in statuses
    }
    selected_labels = st.multiselect(
        "Select Packs To Sync",
        options=list(pack_options.keys()),
        key="admin_dataset_catalog_selected_packs",
    )
    if st.button("Sync Selected Packs", key="admin_dataset_catalog_sync"):
        try:
            synced = sync_remote_dataset_catalog(
                catalog_url,
                destination_dir=destination_dir,
                pack_ids=[pack_options[label] for label in selected_labels]
                if selected_labels
                else None,
                force=sync_force,
                timeout_seconds=float(timeout_seconds),
                allow_untrusted=allow_untrusted,
            )
            st.success(f"Synced {len(synced)} pack(s) into {destination_dir}.")
        except Exception as exc:
            st.error(f"Remote pack sync failed: {exc}")

    with st.expander("Raw Remote Catalog", expanded=False):
        st.json(catalog)


def _render_validation_result(
    rows: list[dict],
    issues: list[str],
    label: str,
) -> None:
    st.write(f"Validated: **{label}**")
    if issues:
        st.warning(f"Found {len(issues)} validation issue(s).")
        for issue in issues:
            st.write(f"- {issue}")
    else:
        st.success("Dataset is valid.")
    if rows:
        preview_columns = [
            "id",
            "category",
            "universal_category",
            "should_refuse",
        ]
        st.dataframe(
            pd.DataFrame(rows[:10])[preview_columns],
            width="stretch",
            hide_index=True,
        )
