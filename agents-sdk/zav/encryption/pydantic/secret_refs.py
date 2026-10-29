from typing import Any, Callable, Dict, List, Mapping, Optional, Set, Union

from zav.pydantic_compat import BaseModel, sequence_like

from zav.encryption.pydantic.encrypted_str import EncryptedStr
from zav.encryption.pydantic.encrypted_str_ref import EncryptedStrRef

GetSecret = Callable[[str], Optional[EncryptedStr]]
SecretRefHolder = Union[EncryptedStrRef, BaseModel, Dict[str, Any], List[Any]]


def resolve_secret_refs(
    value: SecretRefHolder, get_secret: GetSecret
) -> SecretRefHolder:
    """Walks a value like EncryptableBaseModel.decrypt. Every EncryptedStrRef is
    resolved, and a field ``x`` left unset takes the secret its sibling
    ``x_ref`` names. Values without refs are returned as they are."""
    if isinstance(value, EncryptedStrRef):
        return value.resolve(get_secret(value))
    elif isinstance(value, BaseModel):
        update: Dict[str, Any] = {
            k: resolve_secret_refs(v, get_secret) for k, v in value
        }
        __fill_fields_from_ref_siblings(value, update, get_secret)
        if all(update[k] is v for k, v in value):
            return value
        return value.copy(update=update)
    elif isinstance(value, dict):
        items = {k: resolve_secret_refs(v, get_secret) for k, v in value.items()}
        if all(items[k] is v for k, v in value.items()):
            return value
        return items
    elif sequence_like(value):
        resolved = [resolve_secret_refs(v, get_secret) for v in value]
        if all(item is v for item, v in zip(resolved, value)):
            return value
        return resolved
    return value


def __fill_fields_from_ref_siblings(
    value: BaseModel, update: Dict[str, Any], get_secret: GetSecret
) -> None:
    # Compatibility with models that still declare ``x: Optional[EncryptedStr]``
    # next to ``x_ref``: code reading ``x`` keeps working while the configuration
    # names the secret instead of holding it. Delete together with the inline
    # EncryptedStr fields.
    for name, ref in value:
        field = name[: -len("_ref")]
        if (
            isinstance(ref, EncryptedStrRef)
            and name.endswith("_ref")
            and field in update
            and update[field] is None
        ):
            update[field] = get_secret(ref)


def secret_ref_names(value: SecretRefHolder) -> Set[str]:
    """The names every EncryptedStrRef inside ``value`` refers to."""
    if isinstance(value, EncryptedStrRef):
        return {str(value)}
    if isinstance(value, BaseModel):
        parts = [secret_ref_names(v) for _, v in value]
    elif isinstance(value, dict):
        parts = [secret_ref_names(v) for v in value.values()]
    elif sequence_like(value):
        parts = [secret_ref_names(v) for v in value]
    else:
        return set()
    return set().union(*parts)


def secret_or_ref(vals: Mapping[str, Any], field: str, required: bool = False) -> None:
    """For a model validator: ``field`` holds the value or ``<field>_ref`` names the
    secret, never both."""
    ref = f"{field}_ref"
    if vals.get(field) is not None and vals.get(ref) is not None:
        raise ValueError(f"Set either '{field}' or '{ref}', not both.")
    if required and vals.get(field) is None and vals.get(ref) is None:
        raise ValueError(f"Set '{field}' or '{ref}'.")
