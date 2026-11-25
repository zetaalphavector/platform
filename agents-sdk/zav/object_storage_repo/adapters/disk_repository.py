import os
from os import DirEntry
from typing import AsyncGenerator, List, Optional, Tuple
from urllib.parse import urlparse

import aiofiles
from aiofiles.os import makedirs, remove, scandir, stat

from zav.object_storage_repo.domain.object_storage_item import (
    ObjectStorageAttributes,
    ObjectStorageItem,
    ObjectStorageRange,
)
from zav.object_storage_repo.repository import (
    ObjectModifiedException,
    ObjectRepository,
    ObjectRepositoryException,
    validate_byte_range,
)
from zav.object_storage_repo.repository_factory import ObjectRepositoryFactory


def _etag(stats: os.stat_result) -> str:
    return f'"{stats.st_mtime_ns}-{stats.st_size}"'


async def scantree(path: str, recursive: bool = True) -> AsyncGenerator[DirEntry, None]:
    """Yield the file entries of a directory, and of its subdirectories if
    recursive."""
    dir_iterator = await scandir(path)
    for entry in sorted(dir_iterator, key=lambda en: en.path):
        if entry.is_dir(follow_symlinks=False):
            if recursive:
                async for nested_entry in scantree(entry.path):
                    yield nested_entry
        else:
            yield entry


@ObjectRepositoryFactory.register("disk")
class DiskObjectRepository(ObjectRepository):
    def __init__(self, **kwargs):
        pass

    @classmethod
    def _parse_url(cls, url) -> Tuple[str, str]:
        parsed = urlparse(url, allow_fragments=False)
        host = parsed.netloc
        path = parsed.path
        return host, path

    async def add(self, item: ObjectStorageItem) -> Optional[ObjectStorageItem]:
        try:
            _, path = self._parse_url(item.url)
            parent = "/".join(path.split("/")[:-1])
            if not parent:
                parent = "/"
            await makedirs(parent, exist_ok=True)
            async with aiofiles.open(path, mode="wb") as f:
                await f.write(item.payload)
            return item
        except Exception:
            return None

    async def get(self, url: str) -> Optional[ObjectStorageItem]:
        """ "
        Args:
            url: Disk URL to download. This is of the form
            file:///path/to/file
        """
        try:
            _, path = self._parse_url(url)
            async with aiofiles.open(path, mode="rb") as f:
                content = await f.read()
            if isinstance(content, str):
                content = content.encode("utf-8")
            return ObjectStorageItem(url=url, payload=content)
        except Exception:
            return None

    async def get_by_range(
        self, url: str, offset: int, size: int, etag: Optional[str] = None
    ) -> Optional[ObjectStorageRange]:
        validate_byte_range(offset, size)
        try:
            _, path = self._parse_url(url)
            async with aiofiles.open(path, mode="rb") as f:
                stats = os.fstat(f.fileno())
                await f.seek(offset)
                content = await f.read(size)
                stats_after_read = os.fstat(f.fileno())
        except FileNotFoundError:
            if etag is not None:
                raise ObjectModifiedException(f"{url} no longer matches etag {etag}")
            return None
        except Exception as e:
            raise ObjectRepositoryException(e)
        current_etag = _etag(stats)
        if _etag(stats_after_read) != current_etag:
            raise ObjectModifiedException(f"{url} changed while it was read")
        if etag is not None and etag != current_etag:
            raise ObjectModifiedException(f"{url} no longer matches etag {etag}")
        if not content:
            raise ObjectRepositoryException(f"Offset {offset} is past the end of {url}")
        return ObjectStorageRange(
            content=content, total_size=stats.st_size, etag=current_etag
        )

    async def delete(self, item: ObjectStorageItem) -> Optional[ObjectStorageItem]:
        try:
            _, path = self._parse_url(item.url)
            await remove(path)
            return item
        except Exception:
            return None

    async def get_delegated_get_url(
        self,
        url: str,
        expiration_seconds: Optional[int] = 1,
        content_type: Optional[str] = None,
        filename: Optional[str] = None,
    ) -> Optional[str]:
        raise NotImplementedError

    async def get_delegated_add_url(
        self,
        url: str,
        expiration_seconds: Optional[int] = 1,
        content_type: Optional[str] = None,
    ) -> Optional[str]:
        raise NotImplementedError

    async def get_object_attributes(
        self, url: str
    ) -> Optional[ObjectStorageAttributes]:
        try:
            _, path = self._parse_url(url)
            stats = await stat(path)
            if stats.st_mtime and stats.st_size:
                return ObjectStorageAttributes(
                    url=url,
                    last_modified=stats.st_mtime,  # type: ignore
                    content_length=stats.st_size,
                )
            return None
        except Exception:
            return None

    async def filter_objects_attributes(
        self, url_prefix: str, recursive: bool = True
    ) -> List[ObjectStorageAttributes]:
        try:
            host, path = self._parse_url(url_prefix)
            objects: List[ObjectStorageAttributes] = []
            async for entry in scantree(path, recursive):
                stats = entry.stat()
                if stats.st_mtime and stats.st_size and entry.path:
                    objects.append(
                        ObjectStorageAttributes(
                            url=f"file://{host}/{entry.path.lstrip('/')}",
                            last_modified=stats.st_mtime,  # type: ignore
                            content_length=stats.st_size,
                        )
                    )
            return objects
        except Exception:
            return []
