from abc import ABC, abstractmethod
from typing import Callable, Dict, Optional, Type

try:
    from pydantic.v1 import BaseModel
except ImportError:
    from pydantic import BaseModel  # type: ignore

from zav.encryption.configuration import (
    AesConfiguration,
    FernetConfiguration,
    KmsConfiguration,
)
from zav.encryption.envelope import CipherWrapper


class EncrypterConfiguration(BaseModel):
    fernet: Optional[FernetConfiguration] = None
    aes: Optional[AesConfiguration] = None
    kms: Optional[KmsConfiguration] = None
    plain_text: Optional[str] = None


class AbstractEncrypter(ABC):
    method: str

    @abstractmethod
    def __init__(self, **config):
        self.__config = config

    @abstractmethod
    async def _encrypt(self, value: str) -> str:
        raise NotImplementedError

    @abstractmethod
    async def _decrypt(self, value: str) -> str:
        raise NotImplementedError

    async def encrypt(self, value: str) -> str:
        encrypted = await self._encrypt(value)
        return CipherWrapper.wrap(self.method, encrypted)

    async def decrypt(self, value: str) -> str:
        method, encrypted_value = CipherWrapper.unwrap(value)
        if method != self.method:
            other_encrypter = EncrypterFactory.create(
                method, EncrypterConfiguration.parse_obj(self.__config)
            )
            return await other_encrypter.decrypt(value)
        return await self._decrypt(encrypted_value)


class EncrypterFactory:
    registry: Dict[str, Type[AbstractEncrypter]] = {}

    @classmethod
    def register(cls, backend: str) -> Callable:
        def inner_wrapper(
            wrapped_class: Type[AbstractEncrypter],
        ) -> Type[AbstractEncrypter]:
            cls.registry[backend] = wrapped_class
            return wrapped_class

        return inner_wrapper

    @classmethod
    def create(cls, backend: str, config: EncrypterConfiguration) -> AbstractEncrypter:
        if backend not in cls.registry:
            raise ValueError(f"Unknown class name: {backend}")

        instance = cls.registry[backend](
            **{key: value for key, value in config if value is not None}
        )
        instance.method = backend
        return instance
