from typing import Any, Optional
from urllib.parse import quote

from fastapi.responses import Response


class DownloadResponse(Response):
    def __init__(
        self,
        content: Any,
        media_type: str,
        filename: str,
        status_code: int = 200,
        headers: Optional[dict] = None,
    ) -> None:
        super().__init__(
            content=content,
            headers=headers,  # type: ignore
            media_type=media_type,
            status_code=status_code,
        )
        content_disposition_filename = quote(filename)
        if content_disposition_filename != filename:
            content_disposition = "attachment; filename*=utf-8''{}".format(
                content_disposition_filename
            )
        else:
            content_disposition = f'attachment; filename="{filename}"'
        self.headers.setdefault("content-disposition", content_disposition)
