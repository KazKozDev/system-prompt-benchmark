"""Tests for API key auth helpers."""

from __future__ import annotations

import json
import os
from unittest.mock import patch

import pytest


def _set_keys(mapping: dict) -> str:
    return json.dumps(mapping)


def test_load_api_keys_empty_env():
    from src.platform import api_auth

    with patch.dict(os.environ, {"SPB_API_KEYS_JSON": ""}):
        keys = api_auth.load_api_keys()
    assert keys == {}


def test_load_api_keys_string_value_defaults_to_admin():
    from src.platform import api_auth

    payload = _set_keys({"mytoken": "alice"})
    with patch.dict(os.environ, {"SPB_API_KEYS_JSON": payload}):
        keys = api_auth.load_api_keys()
    assert "mytoken" in keys
    assert keys["mytoken"].principal == "alice"
    assert keys["mytoken"].role == "admin"


def test_load_api_keys_dict_value():
    from src.platform import api_auth

    payload = _set_keys({"tok1": {"principal": "bob", "role": "viewer"}})
    with patch.dict(os.environ, {"SPB_API_KEYS_JSON": payload}):
        keys = api_auth.load_api_keys()
    assert keys["tok1"].role == "viewer"
    assert keys["tok1"].principal == "bob"


def test_load_api_keys_invalid_json_raises():
    from src.platform import api_auth

    with patch.dict(os.environ, {"SPB_API_KEYS_JSON": "{not valid json"}):
        with pytest.raises(RuntimeError, match="Invalid SPB_API_KEYS_JSON"):
            api_auth.load_api_keys()


def test_load_api_keys_invalid_role_raises():
    from src.platform import api_auth

    payload = _set_keys({"tok": {"principal": "x", "role": "superuser"}})
    with patch.dict(os.environ, {"SPB_API_KEYS_JSON": payload}):
        with pytest.raises(RuntimeError, match="Unsupported API role"):
            api_auth.load_api_keys()


def test_load_api_keys_all_valid_roles():
    from src.platform import api_auth

    payload = _set_keys(
        {
            "tok1": {"principal": "a", "role": "viewer"},
            "tok2": {"principal": "b", "role": "runner"},
            "tok3": {"principal": "c", "role": "admin"},
        }
    )
    with patch.dict(os.environ, {"SPB_API_KEYS_JSON": payload}):
        keys = api_auth.load_api_keys()
    assert len(keys) == 3
