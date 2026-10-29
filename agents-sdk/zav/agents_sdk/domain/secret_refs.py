from typing import Any, Dict, List, Optional, Union

from zav.encryption.pydantic import EncryptedStr, EncryptedStrRef
from zav.logging import logger
from zav.pydantic_compat import BaseModel, sequence_like

from zav.agents_sdk.domain.secret_store import SecretStore

SecretRefHolder = Union[EncryptedStrRef, BaseModel, Dict[str, Any], List[Any]]


def resolve_setup_secret_refs(
    value: SecretRefHolder, setup_value: Any, store: Optional[SecretStore]
) -> SecretRefHolder:
    """Walks a configuration like EncryptableBaseModel.decrypt. ``setup_value``
    is the stored agent setup at the same path: a ref resolves only where the
    setup names that same secret, so a request cannot pick one. Configuration
    is stored in clear, so it may name a secret but never hold one."""
    if isinstance(value, EncryptedStrRef):
        if value != setup_value:
            return value
        secret = store.get(value) if store is not None else None
        if secret is None:
            logger.warning(f"Secret '{value}' is not configured")
        return value.resolve(secret)
    elif isinstance(value, EncryptedStr):
        raise ValueError(
            "Configuration holds an EncryptedStr; name the secret with an "
            "EncryptedStrRef instead"
        )
    elif isinstance(value, BaseModel):
        setup_value = setup_value if isinstance(setup_value, dict) else {}
        update = {
            k: resolve_setup_secret_refs(v, setup_value.get(k), store) for k, v in value
        }
        if all(update[k] is v for k, v in value):
            return value
        return value.copy(update=update)
    elif isinstance(value, dict):
        setup_value = setup_value if isinstance(setup_value, dict) else {}
        update = {
            k: resolve_setup_secret_refs(v, setup_value.get(k), store)
            for k, v in value.items()
        }
        if all(update[k] is v for k, v in value.items()):
            return value
        return update
    elif sequence_like(value):
        if not (sequence_like(setup_value) and len(setup_value) == len(value)):
            setup_value = [None] * len(value)
        items = [
            resolve_setup_secret_refs(v, s, store) for v, s in zip(value, setup_value)
        ]
        if all(item is v for item, v in zip(items, value)):
            return value
        return items
    return value
