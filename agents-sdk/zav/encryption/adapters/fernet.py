from cryptography.fernet import Fernet

from zav.encryption.configuration.fernet import FernetConfiguration
from zav.encryption.encrypter import AbstractEncrypter, EncrypterFactory


@EncrypterFactory.register("fernet")
class FernetEncrypter(AbstractEncrypter):
    def __init__(self, fernet: FernetConfiguration, **_rest):
        super().__init__(fernet=fernet, **_rest)
        self.__engine = Fernet(fernet.key)

    async def _encrypt(self, value: str) -> str:
        return self.__engine.encrypt(value.encode("utf-8")).decode("utf-8")

    async def _decrypt(self, value: str) -> str:
        return self.__engine.decrypt(value.encode("utf-8")).decode("utf-8")
