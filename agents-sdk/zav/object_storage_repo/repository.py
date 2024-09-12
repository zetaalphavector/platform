from abc import ABC, abstractmethod
from typing import List, Optional

from zav.object_storage_repo.domain.object_storage_item import (
    ObjectStorageAttributes,
    ObjectStorageItem,
)


class ObjectRepositoryException(Exception):

    pass


class ObjectRepository(ABC):
    @abstractmethod
    def __init__(self, **config):
        raise NotImplementedError

    @abstractmethod
    async def add(self, item: ObjectStorageItem) -> Optional[ObjectStorageItem]:
        raise NotImplementedError

    @abstractmethod
    async def get(self, url: str) -> Optional[ObjectStorageItem]:
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
        self, url_prefix: str
    ) -> List[ObjectStorageAttributes]:
        raise NotImplementedError
