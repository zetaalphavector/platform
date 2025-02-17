from typing import Any, Dict, Optional, Type

try:
    from pydantic.v1.utils import update_not_none
    from pydantic.v1.validators import constr_length_validator, str_validator
except ImportError:
    from pydantic.utils import update_not_none  # type: ignore
    from pydantic.validators import constr_length_validator, str_validator  # type: ignore # noqa

from zav.encryption.envelope import CipherWrapper


class EncryptedStr(str):
    min_length: Optional[int] = None
    max_length: Optional[int] = None

    @classmethod
    def __modify_schema__(cls, field_schema: Dict[str, Any]) -> None:
        update_not_none(
            field_schema,
            type="string",
            writeOnly=True,
            format="password",
            minLength=cls.min_length,
            maxLength=cls.max_length,
        )

    def __set_unencrypted_secret(self, unencrypted_value: Optional[str] = None):
        self.__unencrypted_value = unencrypted_value

    def get_unencrypted_secret(self) -> Optional[str]:
        return self.__unencrypted_value

    @classmethod
    def __get_validators__(cls):
        yield cls.validate
        yield constr_length_validator

    @classmethod
    def validate(cls, value: Any) -> "EncryptedStr":
        if isinstance(value, EncryptedStr):
            return value
        value = str_validator(value)
        # value is encrypted so we set it directly to underlying str
        if CipherWrapper.is_wrapped_cipher(value):
            instance = cls(value)
            instance.__set_unencrypted_secret()
        else:
            # value is not encrypted so we hide it under the unencrypted_value attribute
            instance = cls("")
            instance.__set_unencrypted_secret(value)
        return instance


def encrypted_str(
    *, min_length: Optional[int] = None, max_length: Optional[int] = None
) -> Type[EncryptedStr]:
    # use kwargs then define conf in a dict to aid with IDE type hinting
    namespace = dict(
        min_length=min_length,
        max_length=max_length,
    )
    return type("EncryptedStr", (EncryptedStr,), namespace)
