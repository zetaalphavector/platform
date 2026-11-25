from abc import ABC, abstractmethod
from typing import List, Optional

from zav.object_storage_repo.domain.object_storage_item import (
    ObjectStorageAttributes,
    ObjectStorageItem,
    ObjectStorageRange,
)


class ObjectRepositoryException(Exception):
    pass


class ObjectModifiedException(ObjectRepositoryException):
    """The object no longer matches the etag a read was conditioned on."""


def validate_byte_range(offset: int, size: int) -> None:
    if offset < 0 or size <= 0:
        raise ValueError(f"Invalid byte range: offset={offset}, size={size}")


def total_size_from_content_range(content_range: str) -> int:
    """Parses the total from a `bytes <start>-<end>/<total>` Content-Range."""
    return int(content_range.rsplit("/", 1)[1])


class ObjectRepository(ABC):
    @abstractmethod
    def __init__(self, **config):
        raise NotImplementedError

    @classmethod
    def from_configuration(cls, **config) -> "ObjectRepository":
        return cls(**config)

    @abstractmethod
    async def add(self, item: ObjectStorageItem) -> Optional[ObjectStorageItem]:
        raise NotImplementedError

    @abstractmethod
    async def get(self, url: str) -> Optional[ObjectStorageItem]:
        raise NotImplementedError

    async def get_by_range(
        self, url: str, offset: int, size: int, etag: Optional[str] = None
    ) -> Optional[ObjectStorageRange]:
        """Returns None when the object doesn't exist. Raises ObjectModifiedException
        when `etag` is given and the object was changed or deleted since.
        """
        raise NotImplementedError

    @abstractmethod
    async def get_delegated_get_url(
        self,
        url: str,
        expiration_seconds: Optional[int] = 1,
        content_type: Optional[str] = None,
        filename: Optional[str] = None,
    ) -> Optional[str]:
        raise NotImplementedError

    @abstractmethod
    async def get_delegated_add_url(
        self,
        url: str,
        expiration_seconds: Optional[int] = 1,
        content_type: Optional[str] = None,
    ) -> Optional[str]:
        raise NotImplementedError

    @abstractmethod
    async def delete(self, item: ObjectStorageItem) -> Optional[ObjectStorageItem]:
        raise NotImplementedError

    @abstractmethod
    async def get_object_attributes(
        self, url: str
    ) -> Optional[ObjectStorageAttributes]:
        raise NotImplementedError

    @abstractmethod
    async def filter_objects_attributes(
        self, url_prefix: str, recursive: bool = True
    ) -> List[ObjectStorageAttributes]:
        """With recursive=False, only the objects directly under url_prefix, a
        directory ending in `/`, and none of its subdirectories."""
        raise NotImplementedError
