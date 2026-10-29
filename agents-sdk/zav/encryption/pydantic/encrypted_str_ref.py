from typing import Any, Dict, Optional, Type

from zav.pydantic_compat import (
    PYDANTIC_V2,
    GetCoreSchemaHandler,
    GetJsonSchemaHandler,
    JsonSchemaValue,
    core_schema,
    str_validator,
    update_not_none,
)

from zav.encryption.pydantic.encrypted_str import EncryptedStr


class EncryptedStrRef(str):
    """The name of a secret. The value is reachable only through
    get_unencrypted_secret, and only on the instance returned by resolve."""

    if PYDANTIC_V2:

        @classmethod
        def __get_pydantic_json_schema__(
            cls, core_schema: core_schema.CoreSchema, handler: GetJsonSchemaHandler
        ) -> JsonSchemaValue:
            json_schema = handler(core_schema)
            json_schema = handler.resolve_ref_schema(json_schema)
            json_schema.update({"type": "string", "format": "secret-ref"})
            return json_schema

    else:

        @classmethod
        def __modify_schema__(cls, field_schema: Dict[str, Any]) -> None:
            update_not_none(field_schema, type="string", format="secret-ref")

    def __set_unencrypted_secret(self, unencrypted_value: Optional[str] = None):
        self.__unencrypted_value = unencrypted_value

    def get_unencrypted_secret(self) -> Optional[str]:
        return getattr(self, "_EncryptedStrRef__unencrypted_value", None)

    def resolve(self, secret: Optional[EncryptedStr]) -> "EncryptedStrRef":
        resolved = EncryptedStrRef(self)
        resolved.__set_unencrypted_secret(
            secret.get_unencrypted_secret() if secret is not None else None
        )
        return resolved

    if PYDANTIC_V2:

        @staticmethod
        def _serialize(value: "EncryptedStrRef") -> str:
            return value

        @classmethod
        def __get_pydantic_core_schema__(
            cls, source: Type[Any], handler: GetCoreSchemaHandler
        ) -> core_schema.CoreSchema:
            return core_schema.union_schema(
                [
                    core_schema.is_instance_schema(cls),
                    core_schema.chain_schema(
                        [
                            core_schema.str_schema(min_length=1),
                            core_schema.no_info_plain_validator_function(cls.validate),
                        ]
                    ),
                ],
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

    @classmethod
    def validate(cls, value: Any) -> "EncryptedStrRef":
        if isinstance(value, EncryptedStrRef):
            return value
        value = str_validator(value)
        if not value:
            raise ValueError("A secret reference needs a name")
        return EncryptedStrRef(value)
