import base64
import os

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

from zav.encryption.configuration.aes import AesConfiguration
from zav.encryption.encrypter import AbstractEncrypter, EncrypterFactory


@EncrypterFactory.register("aes")
class AesEncrypter(AbstractEncrypter):
    def __init__(self, aes: AesConfiguration, **_rest):
        super().__init__(aes=aes, **_rest)
        self.__algorithm = algorithms.AES(aes.key.encode("utf-8"))
        self.__iv_bytes = aes.iv_bytes

    def __pad_data(self, data: bytes):
        padder = padding.PKCS7(self.__algorithm.block_size).padder()
        padded_data = padder.update(data) + padder.finalize()
        return padded_data

    def __unpad_data(self, data: bytes):
        unpadder = padding.PKCS7(self.__algorithm.block_size).unpadder()
        unpadded_data = unpadder.update(data) + unpadder.finalize()
        return unpadded_data

    async def _encrypt(self, value: str) -> str:
        value_bytes = value.encode("utf-8")
        backend = default_backend()
        iv = os.urandom(self.__iv_bytes)
        padded_value = self.__pad_data(value_bytes)
        cipher = Cipher(self.__algorithm, modes.CFB(iv), backend=backend)
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(padded_value) + encryptor.finalize()
        combined_data = iv + ciphertext
        return base64.b64encode(combined_data).decode("utf-8")

    async def _decrypt(self, value: str) -> str:
        value_bytes = value.encode("utf-8")
        backend = default_backend()
        combined_data = base64.b64decode(value_bytes)
        iv = combined_data[: self.__iv_bytes]
        ciphertext = combined_data[self.__iv_bytes :]
        cipher = Cipher(self.__algorithm, modes.CFB(iv), backend=backend)
        decryptor = cipher.decryptor()
        plaintext = decryptor.update(ciphertext) + decryptor.finalize()
        plaintext = self.__unpad_data(plaintext)
        return plaintext.decode("utf-8")
