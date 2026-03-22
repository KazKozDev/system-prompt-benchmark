"""Simple API key authentication and role checks for the REST API."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

from fastapi import Header, HTTPException, status

ROLE_LEVELS = {
    "viewer": 10,
    "runner": 20,
    "admin": 30,
}


@dataclass
class AuthContext:
    principal: str
    role: str


def load_api_keys() -> dict[str, AuthContext]:
    """Parse API keys from the ``SPB_API_KEYS_JSON`` environment variable.

    Acceptable formats::

        # Simple: token → principal (role defaults to "admin")
        {"mytoken": "alice"}

        # Explicit role:
        {"mytoken": {"principal": "alice", "role": "runner"}}
    """
    raw = os.getenv("SPB_API_KEYS_JSON", "")
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Invalid SPB_API_KEYS_JSON payload") from exc

    resolved: dict[str, AuthContext] = {}
    for token, value in payload.items():
        if isinstance(value, str):
            principal = value
            role = "admin"
        else:
            principal = str(value.get("principal", "api-user"))
            role = str(value.get("role", "viewer")).lower()
        if role not in ROLE_LEVELS:
            raise RuntimeError(f"Unsupported API role: {role}")
        resolved[token] = AuthContext(principal=principal, role=role)
    return resolved


# ---------------------------------------------------------------------------
# Module-level cache — parsed exactly once when this module is first imported.
# All require_role() closures share this single dict for the process lifetime.
# Re-reading the env var on every request would be both wasteful and incorrect
# (env vars do not change after process start in production deployments).
# ---------------------------------------------------------------------------
_API_KEYS: dict[str, AuthContext] = load_api_keys()


def require_role(min_role: str):
    """Return a FastAPI dependency that enforces *min_role* or higher.

    When ``SPB_API_KEYS_JSON`` is not set the service runs in open mode:
    every request is accepted as ``anonymous / admin``.  This is intentional
    for local / development use and matches the previous behaviour.
    """
    minimum = ROLE_LEVELS[min_role]

    def _dependency(
        authorization: str | None = Header(default=None),
        x_api_key: str | None = Header(default=None),
    ) -> AuthContext:
        if not _API_KEYS:
            # No keys configured → open / dev mode.
            return AuthContext(principal="anonymous", role="admin")

        token = x_api_key
        if not token and authorization:
            scheme, _, credentials = authorization.partition(" ")
            if scheme.lower() == "bearer" and credentials:
                token = credentials.strip()

        if not token or token not in _API_KEYS:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing or invalid API key",
            )

        context = _API_KEYS[token]
        if ROLE_LEVELS[context.role] < minimum:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient API role",
            )
        return context

    return _dependency
