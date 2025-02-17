from typing import Any

try:
    from pydantic.v1 import BaseModel
    from pydantic.v1.utils import sequence_like
except ImportError:
    from pydantic import BaseModel  # type: ignore
    from pydantic.utils import sequence_like  # type: ignore

from zav.encryption.encrypter import AbstractEncrypter
from zav.encryption.pydantic.encrypted_str import EncryptedStr


class EncryptableBaseModel(BaseModel):
    async def __encrypt_value(self, encrypter: AbstractEncrypter, value: Any):
        if isinstance(value, BaseModel):
            return value.copy(
                update={k: await self.__encrypt_value(encrypter, v) for k, v in value}
            )
        elif isinstance(value, dict):
            return {
                k: await self.__encrypt_value(encrypter, v) for k, v in value.items()
            }
        elif sequence_like(value):
            return [await self.__encrypt_value(encrypter, v) for v in value]
        elif isinstance(value, EncryptedStr):
            unencrypted_value = value.get_unencrypted_secret()
            if unencrypted_value is not None:
                return EncryptedStr.validate(await encrypter.encrypt(unencrypted_value))
            return value
        else:
            return value

    async def __decrypt_value(self, encrypter: AbstractEncrypter, value: Any):
        if isinstance(value, BaseModel):
            return value.copy(
                update={k: await self.__decrypt_value(encrypter, v) for k, v in value}
            )
        elif isinstance(value, dict):
            return {
                k: await self.__decrypt_value(encrypter, v) for k, v in value.items()
            }
        elif sequence_like(value):
            return [await self.__decrypt_value(encrypter, v) for v in value]
        elif isinstance(value, EncryptedStr):
            if value is not None:
                return EncryptedStr.validate(await encrypter.decrypt(value))
            return value
        else:
            return value

    async def encrypt(self, encrypter: AbstractEncrypter):
        for k, v in self:
            encrypted_value = await self.__encrypt_value(encrypter, v)
            setattr(self, k, encrypted_value)

    async def decrypt(self, encrypter: AbstractEncrypter):
        for k, v in self:
            decrypted_value = await self.__decrypt_value(encrypter, v)
            setattr(self, k, decrypted_value)
