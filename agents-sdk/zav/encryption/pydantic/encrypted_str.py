from typing import Any, Dict, Optional, Type

from zav.pydantic_compat import (
    PYDANTIC_V2,
    GetCoreSchemaHandler,
    GetJsonSchemaHandler,
    JsonSchemaValue,
    constr_length_validator,
    core_schema,
    str_validator,
    update_not_none,
)

from zav.encryption.envelope import CipherWrapper


class EncryptedStr(str):
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    # Pydantic v1 uses __modify_schema__, v2 uses __get_pydantic_json_schema__
    if PYDANTIC_V2:

        @classmethod
        def __get_pydantic_json_schema__(
            cls, core_schema: core_schema.CoreSchema, handler: GetJsonSchemaHandler
        ) -> JsonSchemaValue:
            json_schema = handler(core_schema)
            json_schema = handler.resolve_ref_schema(json_schema)
            json_schema.update(
                {
                    "type": "string",
                    "writeOnly": True,
                    "format": "password",
                    **({"minLength": cls.min_length} if cls.min_length else {}),
                    **({"maxLength": cls.max_length} if cls.max_length else {}),
                }
            )
            return json_schema

    else:

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

    if PYDANTIC_V2:

        @staticmethod
        def _serialize(value: "EncryptedStr") -> str:
            return value

        @classmethod
        def __get_pydantic_core_schema__(
            cls, source: Type[Any], handler: GetCoreSchemaHandler
        ) -> core_schema.CoreSchema:
            return core_schema.no_info_plain_validator_function(
                cls.validate,
                json_schema_input_schema=core_schema.str_schema(
                    max_length=cls.max_length, min_length=cls.min_length
                ),
                serialization=core_schema.plain_serializer_function_ser_schema(
                    cls._serialize,
                    info_arg=False,
                    return_schema=core_schema.str_schema(),
                ),
            )

    else:

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
            instance = EncryptedStr(value)
            instance.__set_unencrypted_secret()
        else:
            # value is not encrypted so we hide it under the unencrypted_value attribute
            instance = EncryptedStr("")
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
