from typing import Optional

from zav.encryption.encrypter import AbstractEncrypter, EncrypterFactory


def alert():
    for _ in range(5):
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! DO NOT USE THIS ENCRYPTER IN PRODUCTION")


@EncrypterFactory.register("plain_text")
class PlainTextEncrypter(AbstractEncrypter):
    def __init__(self, plain_text: Optional[str] = None, **_rest):
        super().__init__(plain_text=plain_text, **_rest)
        alert()

    async def _encrypt(self, value: str) -> str:
        alert()
        return value

    async def _decrypt(self, value: str) -> str:
        alert()
        return value
