"""The one place dev-mode values cross the contract models.

In dev mode (``StoreConfig.dev_mode``: plain ``http://localhost``) a few contract fields cannot hold
their real value, because the contract requires ``https://`` URLs and a DNS hostname:
``UploadGrant.url`` and ``ContentGrant.url`` (``^https://``), and ``WebAuthnRelyingParty`` inside
``StoreHealth``/``StoreStatus`` (``rp_id`` must pass ``validate_store_hostname``; origins must be
``https://<rp_id>``). ``respond(Model, payload, config)`` validates the payload with the dev host
swapped for a placeholder that passes those validators (so every *other* rule is still checked),
then emits the real dev values. Outside dev mode it is a plain validate-and-return.

Clients running against a dev Store read these fields as data (for example the Process Store
client's grant URLs) and must not re-validate them with the contract model's ``https://`` rule.
"""

from __future__ import annotations

from typing import Any, Type

from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from starlette.responses import JSONResponse

from .config import DEV_HOSTNAME, StoreConfig

_PLACEHOLDER_HOST = "call1-dev.invalid"


def _swap(value: Any, pairs) -> Any:
    if isinstance(value, str):
        for old, new in pairs:
            if value == old or value.startswith(old + "/") or value.startswith(old + ":") or value.startswith(old + "?"):
                return new + value[len(old):]
        return value
    if isinstance(value, list):
        return [_swap(v, pairs) for v in value]
    if isinstance(value, dict):
        return {k: _swap(v, pairs) for k, v in value.items()}
    return value


def _pairs(config: StoreConfig):
    to_placeholder = [("http://" + DEV_HOSTNAME, "https://" + _PLACEHOLDER_HOST), (DEV_HOSTNAME, _PLACEHOLDER_HOST)]
    back = [("https://" + _PLACEHOLDER_HOST, "http://" + DEV_HOSTNAME), (_PLACEHOLDER_HOST, DEV_HOSTNAME)]
    return to_placeholder, back


def validate(model: Type[BaseModel], payload: Any, config: StoreConfig) -> dict:
    """The JSON body for ``model``: fully validated; dev host values restored in dev mode."""
    data = jsonable_encoder(payload)
    if not config.dev_mode:
        return model.model_validate(data).model_dump(mode="json")
    to_placeholder, back = _pairs(config)
    checked = model.model_validate(_swap(data, to_placeholder)).model_dump(mode="json")
    return _swap(checked, back)


def respond(model: Type[BaseModel], payload: Any, config: StoreConfig, *, status_code: int = 200) -> JSONResponse:
    return JSONResponse(validate(model, payload, config), status_code=status_code)
