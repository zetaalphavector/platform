import json
from typing import Any, List, Optional, Tuple
from urllib.parse import urlparse

from azure.core import MatchConditions
from azure.core.exceptions import (
    HttpResponseError,
    ResourceModifiedError,
    ResourceNotFoundError,
)
from azure.identity.aio import DefaultAzureCredential
from azure.storage.blob import BlobProperties
from azure.storage.blob.aio import BlobClient, ContainerClient

from zav.object_storage_repo.domain.object_storage_item import (
    ObjectStorageAttributes,
    ObjectStorageItem,
    ObjectStorageRange,
)
from zav.object_storage_repo.repository import (
    ObjectModifiedException,
    ObjectRepository,
    ObjectRepositoryException,
    total_size_from_content_range,
    validate_byte_range,
)
from zav.object_storage_repo.repository_factory import ObjectRepositoryFactory


@ObjectRepositoryFactory.register("azure")
class AzureObjectRepository(ObjectRepository):
    def __init__(
        self,
        azure_credential: Optional[str] = None,
        **kwargs,
    ):
        if azure_credential is None:
            self.__default_credential: Any = DefaultAzureCredential()
        else:
            try:
                self.__default_credential = json.loads(azure_credential)
            except Exception:
                self.__default_credential = azure_credential

    @classmethod
    def _parse_url(cls, url) -> Tuple[str, str, str]:
        parsed = urlparse(url, allow_fragments=False)
        container_name, *rest = parsed.path.lstrip("/").split("/")
        account_url = f"{parsed.scheme}://{parsed.netloc}"
        blob_name = "/".join(rest)
        return account_url, container_name, blob_name

    async def add(self, item: ObjectStorageItem) -> Optional[ObjectStorageItem]:
        try:
            account_url, container_name, blob_name = self._parse_url(item.url)
            async with BlobClient(
                account_url=account_url,
                container_name=container_name,
                blob_name=blob_name,
                credential=self.__default_credential,
            ) as blob:
                await blob.upload_blob(item.payload, overwrite=True)
            return item
        except Exception:
            return None

    async def get(self, url: str) -> Optional[ObjectStorageItem]:
        """ "
        Args:
            url: Azure URL to download. This is of the form
            http://<subdomain.customdomain>/<mycontainer>/<myblob>
        """
        try:
            account_url, container_name, blob_name = self._parse_url(url)
            async with BlobClient(
                account_url=account_url,
                container_name=container_name,
                blob_name=blob_name,
                credential=self.__default_credential,
            ) as blob:
                downloader = await blob.download_blob()
                content = await downloader.content_as_bytes()
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
            account_url, container_name, blob_name = self._parse_url(url)
            async with BlobClient(
                account_url=account_url,
                container_name=container_name,
                blob_name=blob_name,
                credential=self.__default_credential,
            ) as blob:
                downloader = await blob.download_blob(
                    offset=offset,
                    length=size,
                    **(
                        {"etag": etag, "match_condition": MatchConditions.IfNotModified}
                        if etag is not None
                        else {}
                    ),
                )
                content = await downloader.readall()
        except ResourceModifiedError:
            raise ObjectModifiedException(f"{url} no longer matches etag {etag}")
        except ResourceNotFoundError:
            if etag is not None:
                raise ObjectModifiedException(f"{url} no longer matches etag {etag}")
            return None
        except HttpResponseError as e:
            if e.status_code != 416:
                raise ObjectRepositoryException(e)
            content = b""
        except Exception as e:
            raise ObjectRepositoryException(e)
        if not content:
            raise ObjectRepositoryException(f"Offset {offset} is past the end of {url}")
        # `properties.size` is the size of the range, the total is in Content-Range.
        return ObjectStorageRange(
            content=content,
            total_size=total_size_from_content_range(
                downloader.properties.content_range
            ),
            etag=downloader.properties.etag,
        )

    async def delete(self, item: ObjectStorageItem) -> Optional[ObjectStorageItem]:
        try:
            account_url, container_name, blob_name = self._parse_url(item.url)
            async with BlobClient(
                account_url=account_url,
                container_name=container_name,
                blob_name=blob_name,
                credential=self.__default_credential,
            ) as blob:
                await blob.delete_blob()
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
            account_url, container_name, blob_name = self._parse_url(url)
            async with BlobClient(
                account_url=account_url,
                container_name=container_name,
                blob_name=blob_name,
                credential=self.__default_credential,
            ) as blob:
                properties = await blob.get_blob_properties()
            if properties.last_modified and properties.size:
                return ObjectStorageAttributes(
                    url=url,
                    last_modified=properties.last_modified,
                    content_length=properties.size,
                    content_encoding=properties.content_settings.content_encoding,
                    content_language=properties.content_settings.content_language,
                )
            return None
        except Exception:
            return None

    async def filter_objects_attributes(
        self, url_prefix: str, recursive: bool = True
    ) -> List[ObjectStorageAttributes]:
        try:
            account_url, container_name, blob_name_prefix = self._parse_url(url_prefix)
            objects: List[ObjectStorageAttributes] = []
            async with ContainerClient(
                account_url=account_url,
                container_name=container_name,
                credential=self.__default_credential,
            ) as container:
                blobs = (
                    container.list_blobs(name_starts_with=blob_name_prefix)
                    if recursive
                    else container.walk_blobs(name_starts_with=blob_name_prefix)
                )
                async for blob in blobs:
                    # walk_blobs also yields a BlobPrefix for each subdirectory.
                    if not isinstance(blob, BlobProperties):
                        continue
                    if blob.last_modified and blob.size and blob.name:
                        objects.append(
                            ObjectStorageAttributes(
                                url=(
                                    f"{account_url}/{container_name}/"
                                    f"{blob.name.lstrip('/')}"
                                ),
                                last_modified=blob.last_modified,
                                content_length=blob.size,
                                content_encoding=(
                                    blob.content_settings.content_encoding
                                ),
                                content_language=(
                                    blob.content_settings.content_language
                                ),
                            )
                        )
            return objects
        except Exception:
            return []
