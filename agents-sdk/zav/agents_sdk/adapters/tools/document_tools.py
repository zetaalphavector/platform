import json
import re
from typing import Any, Dict, List, Optional

from zav.common.nested_models import getpathattr
from zav.logging import logger
from zav.pydantic_compat import BaseModel, Field

from zav.agents_sdk.adapters.retrievers.zav_retriever import ZAVRetriever
from zav.agents_sdk.adapters.tools.tools_source import ToolsSource
from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory
from zav.agents_sdk.domain.tools import Tool, hide, streamable

_CONTENT_TYPE_FIELD = "custom_metadata.document_content_type"

_DEFAULT_METADATA_FIELDS = [
    "custom_metadata.metadata.title",
    "custom_metadata.metadata.source",
    "custom_metadata.metadata.created",
]

_DEFAULT_DESCRIPTION_FIELD = "custom_metadata.metadata.abstract"


def extract_hit_metadata(
    hit: Dict[str, Any],
    metadata_fields: List[str],
    description_field: str,
) -> Dict[str, Any]:
    """Extract structured document data from a search hit.

    Returns a dict with ``content_type``, ``metadata`` (flat key-value from
    configured fields), and ``description`` (from the description field).
    """
    data: Dict[str, Any] = {}

    content_type = getpathattr(hit, _CONTENT_TYPE_FIELD)
    if content_type:
        data["content_type"] = content_type

    metadata: Dict[str, Any] = {}
    for field in metadata_fields:
        value = getpathattr(hit, field)
        if value is not None:
            key = field.rsplit(".", 1)[-1]
            metadata[key] = value
    if metadata:
        data["metadata"] = metadata

    description = getpathattr(hit, description_field)
    if description:
        data["description"] = str(description)

    return data


class DocumentToolsConfig(BaseModel):
    fe_base_url: str = "https://search.zeta-alpha.com"
    metadata_fields: List[str] = Field(
        default_factory=lambda: list(_DEFAULT_METADATA_FIELDS)
    )
    description_field: str = _DEFAULT_DESCRIPTION_FIELD
    page_size: int = 10


class DocumentTools(ToolsSource):
    """Provides tools for interacting with individual documents: reading,
    locating by fuzzy name, and retrieving structured metadata."""

    source_name = "document_tools"

    def __init__(
        self,
        retriever: ZAVRetriever,
        document_tools_config: DocumentToolsConfig,
    ):
        self.__retriever = retriever
        self.__config = document_tools_config

    async def get_tools(self) -> List[Tool]:
        return [
            Tool.from_callable(
                name="read_document",
                executable=self.read_document,
            ),
            Tool.from_callable(
                name="locate_document",
                executable=self.locate_document,
            ),
            Tool.from_callable(
                name="retrieve_metadata",
                executable=self.retrieve_metadata,
            ),
        ]

    @streamable(
        running_text="Reading document...",
        completed_text="Read document.",
        params_transform=hide,
        response_transform=hide,
    )
    async def read_document(self, document_id: str) -> Dict[str, Any]:
        """Read the full content of a document by its document ID.

        Args:
            document_id: The ID of the document to read.

        Returns:
            Dict[str, Any]: A dictionary containing the document content.
        """
        doc_id = document_id.split("_")[0]
        search_page = 1
        pages_data: Dict[int, Dict[str, Any]] = {}

        while True:
            doc_hits_response = await self.__retriever.search(
                doc_ids=[doc_id],
                retrieval_unit="chunk",
                page_size=10,
                page=search_page,
                collapse=None,
                filters={"exists": {"field_path": "pages"}},
            )
            for hit in doc_hits_response.get("hits", []):
                page_data = getpathattr(hit, "custom_metadata.pages")
                if not isinstance(page_data, dict):
                    continue
                page_number = page_data.get("page_number")
                if page_number is not None and page_number not in pages_data:
                    chunk_id = hit["id"]
                    doc_chunk_id = chunk_id.split("_")[0] + "_0"
                    if any(
                        r.get("resource_type") == "pdf_url"
                        for r in hit.get("custom_metadata", {}).get("resources", [])
                    ):
                        source_url: Optional[str] = (
                            f"/pdf/{doc_chunk_id}?chunkId={chunk_id}"
                        )
                    else:
                        uri_hash = hit.get("uri_hash") or hit.get("organize_doc_id")
                        source_url = f"/documents/{uri_hash}" if uri_hash else None
                    pages_data[page_number] = {**page_data, "source_url": source_url}

            search_page += 1
            if not doc_hits_response.get("next"):
                break

        if not pages_data:
            try:
                text = await self.__retriever.get_full_text(document_id=doc_id)
                return {"content": text}
            except Exception as e:
                logger.exception(f"Error during document retrieval: {e}")
                raise Exception(f"Could not retrieve document: {e}") from e

        document_header = (
            "# Document Content Guide\n\n"
            "This document is organized by pages. Each page contains:\n"
            "- **Source URL**: URL for citing this page.\n"
            "- **Bounding Boxes**: Coordinates and dimensions.\n"
            "- **Extracted Text**: Raw text from the page.\n"
            "- **Contextual Description**: Visual elements description.\n\n"
        )

        markdown_pages = [document_header]
        for page_num in sorted(pages_data.keys()):
            page_data = pages_data[page_num]
            parts = [f"\n{'=' * 80}\nPAGE {page_num}\n{'=' * 80}\n"]

            if source_url := page_data.get("source_url"):
                parts.append(f"\n**Source URL:** {source_url}\n")
            if bounding_boxes := page_data.get("bounding_boxes"):
                parts.append(
                    f"\n**Bounding Boxes:**\n```json\n"
                    f"{json.dumps(bounding_boxes, indent=2)}\n```\n"
                )
            if extracted_text := page_data.get("page_text", ""):
                parts.append(f"\n**Extracted Text:**\n{extracted_text}\n")
            if contextual_text := page_data.get("page_contextual_text", ""):
                parts.append(f"\n**Contextual Description:**\n{contextual_text}\n")

            markdown_pages.append("".join(parts))

        return {"content": "\n".join(markdown_pages)}

    @staticmethod
    def _normalize_locator(locator: str) -> str:
        locator = re.sub(
            r"\.(pdf|docx?|xlsx?|pptx?|txt|csv)$", "", locator, flags=re.IGNORECASE
        )
        locator = re.sub(r"[-_]+", " ", locator)
        locator = re.sub(r"\s+", " ", locator).strip()
        return locator

    @streamable(
        running_text="Searching for document: {locator}...",
        completed_text="Found documents.",
        params_transform=hide,
        response_transform=hide,
    )
    async def locate_document(self, locator: str) -> Dict[str, Any]:
        """Locate a document by a **fuzzy** title, URL, or file name. Use this
        when the user refers to a document by name, even inexactly.

        Args:
            locator: The title, URL, or file name as mentioned by the user.

        Returns:
            Dict with 'results' (list of document metadata dicts) and 'result_count'.
        """
        exact_results = await self.__search_documents(locator)
        normalized = self._normalize_locator(locator)

        if normalized != locator:
            normalized_results = await self.__search_documents(normalized)
            seen_doc_ids: set[str] = set()
            merged: List[Dict[str, Any]] = []
            for result in exact_results + normalized_results:
                doc_id = result.get("source_url")
                if doc_id and doc_id not in seen_doc_ids:
                    seen_doc_ids.add(doc_id)
                    merged.append(result)
            return {"results": merged, "result_count": len(merged)}

        return {"results": exact_results, "result_count": len(exact_results)}

    async def __search_documents(self, query: str) -> List[Dict[str, Any]]:
        try:
            response = await self.__retriever.search(
                search_engine="zeta_alpha",
                retrieval_unit="document",
                retrieval_method="keyword",
                document_types=["document"],
                query_string=query,
                page=1,
                page_size=self.__config.page_size,
            )
        except Exception as e:
            logger.exception(f"Error during document search: {e}")
            raise Exception(f"Could not search documents: {e}") from e

        results = []
        for hit in response.get("hits", []):
            data = extract_hit_metadata(
                hit, self.__config.metadata_fields, self.__config.description_field
            )
            data["document_id"] = hit.get("id", "")
            source_url = f"{self.__config.fe_base_url}{hit['document_url']}"
            data["source_url"] = source_url
            results.append(data)
        return results

    @streamable(
        running_text="Retrieving document metadata...",
        completed_text="Retrieved metadata.",
        params_transform=hide,
        response_transform=hide,
    )
    async def retrieve_metadata(
        self,
        document_ids: List[str],
    ) -> Dict[str, Any]:
        """Retrieve structured metadata for a list of documents by their IDs.

        Args:
            document_ids: List of document IDs to retrieve metadata for.

        Returns:
            Dict with 'documents' (list of metadata dicts keyed by document_id)
            and 'count'.
        """
        if not document_ids:
            return {"documents": [], "count": 0}

        try:
            response = await self.__retriever.list(
                retrieval_unit="document",
                property_name="id",
                property_values=document_ids,
                page_size=len(document_ids),
            )
        except Exception:
            logger.warning("Failed to retrieve document metadata")
            return {"documents": [], "count": 0}

        documents = []
        for hit in response.get("hits", []):
            data = extract_hit_metadata(
                hit, self.__config.metadata_fields, self.__config.description_field
            )
            data["document_id"] = hit.get("id", "")
            source_url = f"{self.__config.fe_base_url}{hit['document_url']}"
            data["source_url"] = source_url
            documents.append(data)

        return {"documents": documents, "count": len(documents)}

    async def retrieve_metadata_by_organize_doc_ids(
        self,
        organize_doc_ids: List[str],
    ) -> Dict[str, Dict[str, Any]]:
        """Retrieve metadata for documents using their organize_doc_id.

        Returns a dict mapping organize_doc_id to extracted metadata.
        Used internally by TagTools to enrich tagged documents.
        """
        if not organize_doc_ids:
            return {}

        try:
            response = await self.__retriever.list(
                retrieval_unit="document",
                property_name="organize_doc_id",
                property_values=organize_doc_ids,
                page_size=len(organize_doc_ids),
            )
        except Exception:
            logger.warning("Failed to retrieve document metadata by organize_doc_id")
            return {}

        result: Dict[str, Dict[str, Any]] = {}
        for hit in response.get("hits", []):
            organize_doc_id = hit.get("organize_doc_id")
            if organize_doc_id:
                data = extract_hit_metadata(
                    hit, self.__config.metadata_fields, self.__config.description_field
                )
                data["document_id"] = hit.get("id", "")
                source_url = f"{self.__config.fe_base_url}{hit['document_url']}"
                data["source_url"] = source_url
                result[organize_doc_id] = data

        return result


class DocumentToolsFactory(AgentDependencyFactory):
    @classmethod
    def create(
        cls,
        zav_retriever: ZAVRetriever,
        document_tools_config: DocumentToolsConfig = DocumentToolsConfig(),
    ) -> DocumentTools:
        return DocumentTools(
            retriever=zav_retriever,
            document_tools_config=document_tools_config,
        )
