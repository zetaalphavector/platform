import json
import os
from typing import Dict, List, Optional, Protocol

from zav.encryption.pydantic import EncryptedStr


class SecretStore(Protocol):
    """Named secrets, scoped to one tenant or one local project."""

    def get(self, name: str) -> Optional[EncryptedStr]: ...

    def names(self) -> List[str]: ...


class StaticSecretStore:
    def __init__(self, secrets: Dict[str, EncryptedStr]):
        self.__secrets = secrets

    def get(self, name: str) -> Optional[EncryptedStr]:
        return self.__secrets.get(name)

    def names(self) -> List[str]:
        return sorted(self.__secrets)


class SecretStoreFromFile:
    """``[{"name": ..., "value": ...}]``, a project's env/secrets.json."""

    def __init__(self, path: Optional[str]):
        secrets: Dict[str, EncryptedStr] = {}
        if path and os.path.isfile(path):
            with open(path, "r") as file:
                for secret in json.load(file):
                    name = secret.get("name")
                    if name and secret.get("value") is not None:
                        secrets[name] = EncryptedStr.validate(str(secret["value"]))
        self.__secrets = secrets

    def get(self, name: str) -> Optional[EncryptedStr]:
        return self.__secrets.get(name)

    def names(self) -> List[str]:
        return sorted(self.__secrets)
