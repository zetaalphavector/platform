from typing import Tuple


class CipherWrapper:
    @staticmethod
    def is_wrapped_cipher(value: str) -> bool:
        return value.startswith("encrypted$")

    @staticmethod
    def unwrap(value: str) -> Tuple[str, str]:
        _, method, encrypted_value = value.split("$", 2)
        return method, encrypted_value

    @staticmethod
    def wrap(method: str, encrypted_value: str) -> str:
        return f"encrypted${method}${encrypted_value}"
