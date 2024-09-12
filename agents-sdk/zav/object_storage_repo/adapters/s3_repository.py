import asyncio
import functools
import io
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple
from urllib.parse import urlparse

import boto3
from botocore.exceptions import ClientError

from zav.object_storage_repo.domain.object_storage_item import ObjectStorageItem
from zav.object_storage_repo.repository import (
    ObjectRepository,
    ObjectRepositoryException,
    ObjectStorageAttributes,
)
from zav.object_storage_repo.repository_factory import ObjectRepositoryFactory


def force_async(fn):
    pool = ThreadPoolExecutor()

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        future = pool.submit(fn, *args, **kwargs)
        return asyncio.wrap_future(future)  # make it awaitable

    return wrapper


@ObjectRepositoryFactory.register("s3")
class S3ObjectRepository(ObjectRepository):
    def __init__(
        self,
        **kwargs,
    ):
        self.__s3_client = boto3.client("s3")

    @classmethod
    def _parse_url(cls, url) -> Tuple[str, str]:
        parsed = urlparse(url, allow_fragments=False)
        key = parsed.path.lstrip("/")
        bucket_name = parsed.netloc
        return bucket_name, key

    async def add(self, item: ObjectStorageItem) -> Optional[ObjectStorageItem]:
        try:
            bucket_name, key = self._parse_url(item.url)
            memfile = io.BytesIO()
            memfile.write(item.payload)
            memfile.seek(0)
            await force_async(self.__s3_client.upload_fileobj)(
                memfile, bucket_name, key
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
            if e.response.get("Error", {}).get("Code") == "404":
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
        self, url_prefix: str
    ) -> List[ObjectStorageAttributes]:
        try:
            bucket_name, key_prefix = self._parse_url(url_prefix)
            response = await force_async(self.__s3_client.list_objects_v2)(
                Bucket=bucket_name, Prefix=key_prefix
            )
            if response and (contents := response.get("Contents")):
                return [
                    ObjectStorageAttributes(
                        url=f"s3://{bucket_name}/{content['Key'].lstrip('/')}",
                        last_modified=content["LastModified"],
                        content_length=content["Size"],
                    )
                    for content in contents
                ]
            return []
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") == "404":
                return []
            raise ObjectRepositoryException(e)
        except Exception as e:
            raise ObjectRepositoryException(e)
