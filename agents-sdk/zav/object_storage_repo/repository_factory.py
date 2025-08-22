from typing import Callable, Dict, Optional, Type, TypedDict

from typing_extensions import Unpack

from zav.object_storage_repo.repository import ObjectRepository


class ObjectRepositoryConfig(TypedDict, total=False):

    azure_credential: Optional[str]
    aws_access_key_id: Optional[str]
    aws_secret_access_key: Optional[str]
    aws_region: Optional[str]
    aws_endpoint_url: Optional[str]


class ObjectRepositoryFactory:

    registry: Dict[str, Type[ObjectRepository]] = {}

    @classmethod
    def register(cls, backend: str) -> Callable:
        def inner_wrapper(
            wrapped_class: Type[ObjectRepository],
        ) -> Type[ObjectRepository]:
            cls.registry[backend] = wrapped_class
            return wrapped_class

        return inner_wrapper

    @classmethod
    def create(
        cls, backend: str, **config: Unpack[ObjectRepositoryConfig]
    ) -> ObjectRepository:
        if backend not in cls.registry:
            raise ValueError(f"Unknown class name: {backend}")

        return cls.registry[backend](**config)
