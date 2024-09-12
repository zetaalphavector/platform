import asyncio
import base64
import functools
from concurrent.futures import ThreadPoolExecutor

import boto3

from zav.encryption.configuration.kms import KmsConfiguration
from zav.encryption.encrypter import AbstractEncrypter, EncrypterFactory


def force_async(fn):
    pool = ThreadPoolExecutor()

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        future = pool.submit(fn, *args, **kwargs)
        return asyncio.wrap_future(future)  # make it awaitable

    return wrapper


@EncrypterFactory.register("kms")
class KmsEncrypter(AbstractEncrypter):
    def __init__(self, kms: KmsConfiguration, **_rest):
        super().__init__(kms=kms, **_rest)
        self.__kms_client = boto3.client("kms", region_name=kms.region_name)
        self.__key_id = kms.key_id

    async def _encrypt(self, value: str) -> str:
        encrypted = await force_async(self.__kms_client.encrypt)(
            KeyId=self.__key_id, Plaintext=value
        )
        binary_encrypted = encrypted["CiphertextBlob"]
        return base64.b64encode(binary_encrypted).decode("utf-8")

    async def _decrypt(self, value: str) -> str:
        value_bytes = base64.b64decode(value)
        meta = await force_async(self.__kms_client.decrypt)(CiphertextBlob=value_bytes)
        plaintext = meta["Plaintext"]
        return plaintext.decode("utf-8")
