import asyncio
import functools
import io
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple
from urllib.parse import urlparse

import boto3
from botocore.client import BaseClient
from botocore.config import Config
from botocore.exceptions import ClientError

from zav.object_storage_repo.domain.object_storage_item import (
    ObjectStorageItem,
    ObjectStorageRange,
)
from zav.object_storage_repo.repository import (
    ObjectModifiedException,
    ObjectRepository,
    ObjectRepositoryException,
    ObjectStorageAttributes,
    total_size_from_content_range,
    validate_byte_range,
)
from zav.object_storage_repo.repository_factory import ObjectRepositoryFactory

# A zeta object reads, writes and deletes all its representations at once on the one
# client its storage settings share: at botocore's default of 10, a drain deleting
# documents of 15 representations discarded about 6 connections per document.
S3_MAX_POOL_CONNECTIONS = 50


def force_async(fn):
    pool = ThreadPoolExecutor()

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        future = pool.submit(fn, *args, **kwargs)
        return asyncio.wrap_future(future)  # make it awaitable

    return wrapper


def create_s3_client(
    aws_access_key_id: Optional[str] = None,
    aws_secret_access_key: Optional[str] = None,
    aws_region: Optional[str] = None,
    aws_endpoint_url: Optional[str] = None,
    aws_max_pool_connections: int = S3_MAX_POOL_CONNECTIONS,
) -> BaseClient:
    client_kwargs = {}
    if aws_access_key_id and aws_secret_access_key:
        client_kwargs["aws_access_key_id"] = aws_access_key_id
        client_kwargs["aws_secret_access_key"] = aws_secret_access_key
    if aws_region:
        client_kwargs["region_name"] = aws_region
    if aws_endpoint_url:
        client_kwargs["endpoint_url"] = aws_endpoint_url
    return boto3.client(
        "s3",
        config=Config(max_pool_connections=aws_max_pool_connections),
        **client_kwargs,
    )


@ObjectRepositoryFactory.register("s3")
class S3ObjectRepository(ObjectRepository):
    def __init__(self, s3_client: BaseClient):
        self.__s3_client = s3_client

    @classmethod
    def from_configuration(  # type: ignore[override]
        cls,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_region: Optional[str] = None,
        aws_endpoint_url: Optional[str] = None,
        aws_max_pool_connections: int = S3_MAX_POOL_CONNECTIONS,
        **kwargs,
    ) -> "S3ObjectRepository":
        return cls(
            create_s3_client(
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                aws_region=aws_region,
                aws_endpoint_url=aws_endpoint_url,
                aws_max_pool_connections=aws_max_pool_connections,
            )
        )

    def __list_objects(
        self, bucket_name: str, key_prefix: str, recursive: bool
    ) -> List[ObjectStorageAttributes]:
        # One ListObjectsV2 page holds 1000 keys, so read every page.
        pages = self.__s3_client.get_paginator("list_objects_v2").paginate(
            Bucket=bucket_name,
            Prefix=key_prefix,
            **({} if recursive else {"Delimiter": "/"}),
        )
        return [
            ObjectStorageAttributes(
                url=f"s3://{bucket_name}/{content['Key'].lstrip('/')}",
                last_modified=content["LastModified"],
                content_length=content["Size"],
            )
            for page in pages
            for content in page.get("Contents", [])
        ]

    @classmethod
    def _parse_url(cls, url) -> Tuple[str, str]:
        parsed = urlparse(url, allow_fragments=False)
        key = parsed.path.lstrip("/")
        bucket_name = parsed.netloc
        return bucket_name, key

    async def add(self, item: ObjectStorageItem) -> Optional[ObjectStorageItem]:
        try:
            bucket_name, key = self._parse_url(item.url)
            await force_async(self.__s3_client.upload_fileobj)(
                io.BytesIO(item.payload), bucket_name, key
            )
            return item
        except Exception as e:
            raise ObjectRepositoryException(e)

    async def get(self, url: str) -> Optional[ObjectStorageItem]:
        """
        Args:
            url: S3 URL to download. This is of the form
            s3://<mybucket>/<mykey>
        """
        try:
            bucket_name, key = self._parse_url(url)
            memfile = io.BytesIO()
            await force_async(self.__s3_client.download_fileobj)(
                bucket_name, key, memfile
            )
            memfile.seek(0)
            content = memfile.read()
            return ObjectStorageItem(url=url, payload=content)
        except ClientError as e:
            # The managed download HEADs first (404), then GETs: a key deleted in
            # between fails the GET with NoSuchKey.
            if e.response.get("Error", {}).get("Code") in ("404", "NoSuchKey"):
                return None
            raise ObjectRepositoryException(e)
        except Exception as e:
            raise ObjectRepositoryException(e)

    async def get_by_range(
        self, url: str, offset: int, size: int, etag: Optional[str] = None
    ) -> Optional[ObjectStorageRange]:
        validate_byte_range(offset, size)
        try:
            bucket_name, key = self._parse_url(url)
            response = await force_async(self.__s3_client.get_object)(
                Bucket=bucket_name,
                Key=key,
                Range=f"bytes={offset}-{offset + size - 1}",
                **({"IfMatch": etag} if etag is not None else {}),
            )
            return ObjectStorageRange(
                content=await force_async(response["Body"].read)(),
                total_size=total_size_from_content_range(response["ContentRange"]),
                etag=response["ETag"],
            )
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code")
            if code == "PreconditionFailed" or (
                code == "NoSuchKey" and etag is not None
            ):
                raise ObjectModifiedException(f"{url} no longer matches etag {etag}")
            if code == "NoSuchKey":
                return None
            raise ObjectRepositoryException(e)
        except Exception as e:
            raise ObjectRepositoryException(e)

    async def delete(self, item: ObjectStorageItem) -> Optional[ObjectStorageItem]:
        try:
            bucket_name, key = self._parse_url(item.url)
            await force_async(self.__s3_client.delete_object)(
                Bucket=bucket_name, Key=key
            )
            return item
        except Exception as e:
            raise ObjectRepositoryException(e)

    async def get_delegated_get_url(
        self,
        url: str,
        expiration_seconds: Optional[int] = 1,
        content_type: Optional[str] = None,
        filename: Optional[str] = None,
    ) -> Optional[str]:
        try:
            bucket_name, key = self._parse_url(url)
            params = {
                "Bucket": bucket_name,
                "Key": key,
            }
            if content_type:
                params.update({"ResponseContentType": content_type})
            if filename:
                params.update(
                    {"ResponseContentDisposition": (f"inline; filename={filename}")}
                )
            delegated_url = await force_async(self.__s3_client.generate_presigned_url)(
                ClientMethod="get_object",
                Params=params,
                ExpiresIn=expiration_seconds,
            )
            return delegated_url
        except Exception as e:
            raise ObjectRepositoryException(e)

    async def get_delegated_add_url(
        self,
        url: str,
        expiration_seconds: Optional[int] = 1,
        content_type: Optional[str] = None,
    ) -> Optional[str]:
        try:
            bucket_name, key = self._parse_url(url)
            params = {
                "Bucket": bucket_name,
                "Key": key,
            }
            if content_type:
                params.update({"ContentType": content_type})
            delegated_url = await force_async(self.__s3_client.generate_presigned_url)(
                ClientMethod="put_object",
                Params=params,
                ExpiresIn=expiration_seconds,
            )
            return delegated_url
        except Exception as e:
            raise ObjectRepositoryException(e)

    async def get_object_attributes(
        self, url: str
    ) -> Optional[ObjectStorageAttributes]:
        try:
            bucket_name, key = self._parse_url(url)
            response = await force_async(self.__s3_client.head_object)(
                Bucket=bucket_name, Key=key
            )
            if response:
                obj_attributes = ObjectStorageAttributes(
                    url=url,
                    last_modified=response["LastModified"],
                    content_length=response["ContentLength"],
                    content_encoding=response.get("ContentEncoding"),
                    content_language=response.get("ContentLanguage"),
                )
                return obj_attributes
            return None
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") == "404":
                return None
            raise ObjectRepositoryException(e)
        except Exception as e:
            raise ObjectRepositoryException(e)

    async def filter_objects_attributes(
        self, url_prefix: str, recursive: bool = True
    ) -> List[ObjectStorageAttributes]:
        try:
            bucket_name, key_prefix = self._parse_url(url_prefix)
            return await force_async(self.__list_objects)(
                bucket_name, key_prefix, recursive
            )
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") == "404":
                return []
            raise ObjectRepositoryException(e)
        except Exception as e:
            raise ObjectRepositoryException(e)
