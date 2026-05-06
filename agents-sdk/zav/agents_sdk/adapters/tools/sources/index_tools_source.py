import inspect
from typing import Any, Dict, List, Literal, Optional, Union

from zav.common.nested_models import getpathattr
from zav.logging import logger
from zav.pydantic_compat import BaseModel, Field

from zav.agents_sdk.adapters.agent_state.citation import (
    CitationConfiguration,
    CitationStore,
)
from zav.agents_sdk.adapters.retrievers.zav_retriever import ZAVRetriever
from zav.agents_sdk.adapters.tools.sources._filter_translator import FilterTranslator
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

_DEFAULT_SORT_FIELD_MAPPING = {
    "date": "date",
    "year": "year",
    "citations": "citations",
}


def extract_hit_metadata(
    hit: Dict[str, Any],
    metadata_fields: List[str],
    description_field: str,
) -> Dict[str, Any]:
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


class FilterHint(BaseModel):
    field_path: str
    description: str = ""
    values: List[str] = Field(default_factory=list)
    example_values: List[str] = Field(default_factory=list)


class IndexToolsSourceConfiguration(BaseModel):
    enabled: bool = False
    fe_base_url: str = "https://search.zeta-alpha.com"
    page_size: int = 10
    browse_page_size: int = 20
    default_retrieval_unit: Literal["document", "chunk"] = "chunk"
    default_retrieval_method: Literal["knn", "keyword", "mixed"] = "mixed"
    default_collapse_key: Optional[str] = None
    metadata_fields: List[str] = Field(
        default_factory=lambda: list(_DEFAULT_METADATA_FIELDS)
    )
    description_field: str = _DEFAULT_DESCRIPTION_FIELD
    description: str = ""
    known_filters: List[FilterHint] = Field(default_factory=list)
    sort_field_mapping: Dict[str, str] = Field(
        default_factory=lambda: dict(_DEFAULT_SORT_FIELD_MAPPING)
    )


class IndexToolsSource(ToolsSource):

    source_name = "index_tools"

    def __init__(
        self,
        retriever: ZAVRetriever,
        index_tools_source_configuration: IndexToolsSourceConfiguration,
        citation_config: CitationConfiguration,
        citation_store: CitationStore,
    ):
        self.__retriever = retriever
        self.__config = index_tools_source_configuration
        self.__citation_config = citation_config
        self.__citation_store = citation_store
        self.__filter_configs: Optional[List[Dict[str, Any]]] = None
        self.enabled = index_tools_source_configuration.enabled

    async def get_tools(self) -> List[Tool]:
        filter_block = self.__filter_description()
        sort_block = self.__sort_description()

        search_desc = inspect.getdoc(self.search) or ""
        if filter_block:
            search_desc += f"\n\nKnown filters:\n{filter_block}"

        browse_desc = inspect.getdoc(self.browse) or ""
        if filter_block:
            browse_desc += f"\n\nKnown filters:\n{filter_block}"
        if sort_block:
            browse_desc += f"\n\nAvailable sort_by values: {sort_block}"

        tools = [
            Tool.from_callable(
                name="search",
                executable=self.search,
                description=search_desc,
            ),
            Tool.from_callable(
                name="browse",
                executable=self.browse,
                description=browse_desc,
            ),
            Tool.from_callable(
                name="get_filter_options",
                executable=self.get_filter_options,
            ),
            Tool.from_callable(
                name="get_document_fields",
                executable=self.get_document_fields,
            ),
            Tool.from_callable(
                name="get_index_configuration",
                executable=self.get_index_configuration,
            ),
        ]
        return tools

    def __filter_description(self) -> str:
        if not self.__config.known_filters:
            return ""
        lines = []
        for f in self.__config.known_filters:
            line = f"- {f.field_path}"
            if f.description:
                line += f": {f.description}"
            if f.values:
                vals = ", ".join(f.values)
                line += f" [values: {vals}]"
            elif f.example_values:
                examples = ", ".join(f.example_values[:5])
                line += f" (e.g. {examples})"
            lines.append(line)
        return "\n".join(lines)

    def __sort_description(self) -> str:
        if not self.__config.sort_field_mapping:
            return ""
        return ", ".join(self.__config.sort_field_mapping.keys())

    async def __get_filter_configs(self) -> List[Dict[str, Any]]:
        if self.__filter_configs is not None:
            return self.__filter_configs
        try:
            tenant_settings = await self.__retriever.get_tenant_settings()
        except Exception:
            self.__filter_configs = []
            return self.__filter_configs
        configs: List[Dict[str, Any]] = []
        for idx in self.__internal_indexes(tenant_settings):
            for f in idx.get("search_filters_config", []):
                if not f.get("field_name"):
                    continue
                configs.append(f)
        self.__filter_configs = configs
        return self.__filter_configs

    @staticmethod
    def __internal_indexes(tenant_settings: Dict[str, Any]) -> List[Dict[str, Any]]:
        client_settings = tenant_settings.get("client_settings", {})
        return [
            idx
            for idx in client_settings.get("indexes", [])
            if idx.get("type", "").lower() == "internal"
        ]

    async def to_prompt(self) -> str:
        parts: List[str] = []
        if self.__config.description:
            parts.append(
                f"You have access to a knowledge base: {self.__config.description}"
            )
        if self.__config.known_filters:
            lines = ["Available filters for narrowing searches:"]
            for f in self.__config.known_filters:
                line = f"- `{f.field_path}`"
                if f.description:
                    line += f": {f.description}"
                if f.values:
                    vals = ", ".join(f.values)
                    line += f" [values: {vals}]"
                elif f.example_values:
                    examples = ", ".join(f.example_values[:5])
                    line += f" (e.g. {examples})"
                lines.append(line)
            parts.append("\n".join(lines))
        if self.__config.sort_field_mapping:
            options = ", ".join(self.__config.sort_field_mapping.keys())
            parts.append(
                f"Use `search` to find information by topic or question "
                f"(results ranked by relevance).\n"
                f"Use `browse` to list or explore documents by criteria "
                f"(sortable by: {options})."
            )
        if not parts:
            return ""
        return "\n\n".join(parts)

    @streamable(
        running_text="Searching: {{ query }}...",
        completed_text=(
            "Retrieved {{ results|length }}"
            " result{{ 's' if results|length != 1 else '' }} about {{ query }}."
        ),
        params_transform=hide,
        response_transform=hide,
    )
    async def search(
        self,
        query: str,
        filters: Optional[Dict[str, Union[str, List[str]]]] = None,
        tag_ids: Optional[List[str]] = None,
        page: int = 1,
        page_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Search the knowledge base for documents matching a query.

        Results are ranked by relevance (semantic + keyword matching).

        Args:
            query: Natural language search query. Supports:
                - Plain text: "neural architecture search"
                    (best for most queries)
                - Quoted phrases: '"attention is all you need"'
                    (exact match, keyword leg only)
                - Boolean: "transformers AND fine-tuning",
                    "BERT NOT GPT" (keyword leg only)
                The semantic (vector) component only sees plain text — advanced
                syntax is processed by the keyword component only.
            filters: Optional key-value filters to narrow results.
                Single values match exactly, lists match any value.
                Example: {"metadata.DCMI.source": "arXiv", "metadata.DCMI.date": "2024"}
                Use get_filter_options to discover available filter fields.
            tag_ids: Optional list of tag IDs to scope the search to documents
                belonging to any of these tags.
            page: Result page number (default: 1).
            page_size: Number of results per page.

        Returns:
            Dict with 'results' (list of document hits with metadata and
            source_url), 'total_hits', 'page', and 'facet_results'.
        """
        return await self.__execute_search(
            query=query,
            filters=filters,
            tag_ids=tag_ids,
            page=page,
            page_size=page_size,
        )

    async def search_with_tag_ids(
        self,
        query: str,
        tag_ids: List[str],
        page: int = 1,
        page_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        return await self.__execute_search(
            query=query, tag_ids=tag_ids, page=page, page_size=page_size
        )

    async def search_federated(
        self,
        query: str,
        index_id: str,
        page: int = 1,
        page_size: Optional[int] = None,
        urls: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        effective_query = query
        if urls:
            site_clauses = " OR ".join(f"site:{url}" for url in urls)
            effective_query = f"{query} {site_clauses}"
        return await self.__execute_search(
            query=effective_query,
            index_type="federated",
            index_id=index_id,
            page=page,
            page_size=page_size,
            retrieval_unit="chunk",
        )

    async def search_with_visibility(
        self,
        query: str,
        visibility: List[str],
        page: int = 1,
        page_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        return await self.__execute_search(
            query=query,
            visibility=visibility,
            page=page,
            page_size=page_size,
        )

    async def get_federated_indexes(self) -> List[Dict[str, Any]]:
        try:
            tenant_settings = await self.__retriever.get_tenant_settings()
        except Exception:
            return []
        client_settings = tenant_settings.get("client_settings", {})
        return [
            idx
            for idx in client_settings.get("indexes", [])
            if idx.get("type", "").lower() == "federated"
        ]

    async def search_documents_by_keyword(
        self,
        query: str,
        page_size: int = 10,
    ) -> List[Dict[str, Any]]:
        """Keyword search at document level returning formatted results.

        Uses keyword retrieval at the document level — intended for
        document-location use cases, not answer-generation.
        """
        try:
            response = await self.__retriever.search(
                search_engine="zeta_alpha",
                retrieval_unit="document",
                retrieval_method="keyword",
                document_types=["document"],
                query_string=query,
                page=1,
                page_size=page_size,
            )
        except Exception as e:
            logger.exception(f"Error during document keyword search: {e}")
            raise Exception(f"Could not search documents: {e}") from e

        return self.__format_hits(
            response.get("hits", []),
            index_type=response.get("index_type"),
            index_id=response.get("index_id"),
        )

    async def list_documents(
        self,
        property_name: str,
        property_values: List[str],
        page_size: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """List documents by a property (e.g. ``id`` or ``organize_doc_id``).

        Returns formatted results with metadata and source URLs.
        """
        if not property_values:
            return []

        effective_page_size = page_size or len(property_values)
        try:
            response = await self.__retriever.list(
                retrieval_unit="document",
                property_name=property_name,
                property_values=property_values,
                page_size=effective_page_size,
            )
        except Exception:
            logger.warning(
                f"Failed to list documents by {property_name}",
            )
            return []

        return self.__format_hits(response.get("hits", []))

    async def retrieve_metadata_by_organize_doc_ids(
        self,
        organize_doc_ids: List[str],
    ) -> Dict[str, Dict[str, Any]]:
        """Retrieve metadata for documents using their organize_doc_id.

        Returns a dict mapping organize_doc_id to extracted metadata.
        Used by TagTools to enrich tagged documents.
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
                    hit,
                    self.__config.metadata_fields,
                    self.__config.description_field,
                )
                data["document_id"] = hit.get("id", "")
                data["source_url"] = f"{self.__config.fe_base_url}{hit['document_url']}"
                result[organize_doc_id] = data

        return result

    def __compute_short_id(self, doc_id: str) -> Optional[str]:
        if self.__citation_config.strategy != "short_id":
            return None
        hash_part = doc_id.rsplit("_", 1)[0] if "_" in doc_id else doc_id
        return hash_part[-8:]

    def __format_hits(
        self,
        hits: List[Dict[str, Any]],
        index_type: Optional[str] = None,
        index_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        results = []
        for hit in hits:
            data = extract_hit_metadata(
                hit,
                self.__config.metadata_fields,
                self.__config.description_field,
            )
            data["document_id"] = hit.get("id", "")
            source_url = f"{self.__config.fe_base_url}{hit['document_url']}"
            data["source_url"] = source_url

            short_id = self.__compute_short_id(hit.get("id", ""))
            if short_id:
                data["short_id"] = short_id

            self.__citation_store.register_hit(
                source_url=source_url,
                context_hit=data,
                search_hit=hit,
                short_id=short_id,
                index_type=index_type,
                index_id=index_id,
            )

            results.append(data)
        return results

    async def get_document_content(self, document_id: str) -> Dict[str, Any]:
        doc_id = document_id.split("_")[0]
        search_page = 1
        pages_data: Dict[int, Dict[str, Any]] = {}

        try:
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
                        source_path: Optional[str]
                        cm = hit.get("custom_metadata", {})
                        has_pdf = any(
                            r.get("resource_type") == "pdf_url"
                            for r in cm.get("resources", [])
                        )
                        if has_pdf:
                            source_path = f"/pdf/{doc_chunk_id}?chunkId={chunk_id}"
                        else:
                            uid = hit.get("uri_hash") or hit.get("organize_doc_id")
                            source_path = f"/documents/{uid}" if uid else None

                        source_url = (
                            f"{self.__config.fe_base_url}{source_path}"
                            if source_path
                            else None
                        )

                        if source_url:
                            short_id = self.__compute_short_id(chunk_id)
                            self.__citation_store.register_hit(
                                source_url=source_url,
                                context_hit={
                                    "document_id": doc_id,
                                    "page": page_number,
                                },
                                search_hit=hit,
                                short_id=short_id,
                                index_type=doc_hits_response.get("index_type"),
                                index_id=doc_hits_response.get("index_id"),
                            )

                        pages_data[page_number] = {
                            **page_data,
                            "source_url": source_url,
                        }

                search_page += 1
                if not doc_hits_response.get("next"):
                    break
        except Exception:
            logger.warning(
                f"Page-level retrieval failed for {doc_id}, falling back to full text"
            )

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
            "- **Extracted Text**: Raw text from the page.\n"
            "- **Contextual Description**: Visual elements description.\n\n"
        )

        markdown_pages = [document_header]
        for page_num in sorted(pages_data.keys()):
            page_data = pages_data[page_num]
            parts = [f"\n{'=' * 80}\nPAGE {page_num}\n{'=' * 80}\n"]

            if source_url := page_data.get("source_url"):
                parts.append(f"\n**Source URL:** {source_url}\n")
            if extracted_text := page_data.get("page_text", ""):
                parts.append(f"\n**Extracted Text:**\n{extracted_text}\n")
            if contextual_text := page_data.get("page_contextual_text", ""):
                parts.append(f"\n**Contextual Description:**\n{contextual_text}\n")

            markdown_pages.append("".join(parts))

        return {"content": "\n".join(markdown_pages)}

    async def __execute_search(
        self,
        query: str,
        filters: Optional[Dict[str, Union[str, List[str]]]] = None,
        page: int = 1,
        page_size: Optional[int] = None,
        tag_ids: Optional[List[str]] = None,
        index_type: Literal["internal", "federated"] = "internal",
        index_id: Optional[str] = None,
        visibility: Optional[List[str]] = None,
        retrieval_unit: Optional[Literal["document", "chunk"]] = None,
    ) -> Dict[str, Any]:
        merged_filters = (
            FilterTranslator.to_filters_configuration(filters) if filters else None
        )

        sorting = None

        effective_page_size = page_size or self.__config.page_size
        effective_unit = retrieval_unit or self.__config.default_retrieval_unit

        try:
            response = await self.__retriever.search(
                query_string=query,
                retrieval_unit=effective_unit,
                retrieval_method=self.__config.default_retrieval_method,
                filters=merged_filters,
                page=page,
                page_size=effective_page_size,
                sorting=sorting,
                document_types=["document"],
                tag_ids=tag_ids,
                index_type=index_type,
                index_id=index_id,
                visibility=visibility,
                collapse=self.__config.default_collapse_key,
            )
        except Exception as e:
            logger.exception(f"Error during search: {e}")
            raise Exception(f"Could not perform search: {e}") from e

        results = []
        for hit in response.get("hits", []):
            data = extract_hit_metadata(
                hit,
                self.__config.metadata_fields,
                self.__config.description_field,
            )
            data["document_id"] = hit.get("id", "")
            if index_type == "federated":
                source_url = hit.get("uri", "")
            else:
                source_url = f"{self.__config.fe_base_url}{hit['document_url']}"
            data["source_url"] = source_url

            short_id = self.__compute_short_id(hit.get("id", ""))
            if short_id:
                data["short_id"] = short_id

            self.__citation_store.register_hit(
                source_url=source_url,
                context_hit=data,
                search_hit=hit,
                short_id=short_id,
                index_type=index_type,
                index_id=index_id,
            )

            results.append(data)

        facet_results = response.get("facet_results", [])

        return {
            "results": results,
            "total_hits": response.get("total_hits", 0),
            "page": page,
            "facet_results": facet_results,
        }

    @streamable(
        running_text=(
            "Browsing documents" "{{ ' about ' + query if query else '' }}..."
        ),
        completed_text=(
            "Found {{ results|length }}"
            " document{{ 's' if results|length != 1 else '' }} about {{ query }}."
        ),
        params_transform=hide,
        response_transform=hide,
    )
    async def browse(
        self,
        filters: Optional[Dict[str, Union[str, List[str]]]] = None,
        sort_by: str = "date",
        sort_order: str = "desc",
        query: Optional[str] = None,
        page: int = 1,
        page_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Browse and list documents by criteria with sorting.

        Use this to explore, list, or filter documents — e.g. "show me the
        latest papers", "list documents from Confluence", "what was added
        this month". Results are sorted by the chosen field, not by relevance.

        Use `search` instead when you need to find documents by topic or
        answer a question (relevance-ranked).

        Args:
            filters: Key-value filters to select documents.
                Single values match exactly, lists match any value.
                Example: {"metadata.DCMI.source": "arXiv"}
            sort_by: Sort field name (see system prompt for available options).
                Default: "date".
            sort_order: "desc" (newest/highest first) or "asc"
                (oldest/lowest first). Default: "desc".
            query: Optional keyword query to narrow results. Keep it
                simple — a few terms or a quoted phrase. Do NOT use
                boolean operators (AND, OR, NOT) or parenthesised
                expressions; use filters instead to restrict results.
            page: Result page number (default: 1).
            page_size: Number of results per page.

        Returns:
            Dict with 'results' (list of document metadata with source_url),
            'total_hits', and 'page'.
        """
        merged_filters = (
            FilterTranslator.to_filters_configuration(filters) if filters else None
        )

        mapping = self.__config.sort_field_mapping
        if sort_by not in mapping:
            available = ", ".join(mapping.keys())
            raise ValueError(f"Unknown sort field '{sort_by}'. Available: {available}")
        mapped_field = mapping[sort_by]

        effective_page_size = page_size or self.__config.browse_page_size

        try:
            response = await self.__retriever.search(
                query_string=query or "",
                retrieval_unit="document",
                retrieval_method="keyword",
                filters=merged_filters,
                page=page,
                page_size=effective_page_size,
                sort={mapped_field: sort_order},
                sort_order=[mapped_field],
                collapse="uid",
                document_types=["document"],
            )
        except Exception as e:
            logger.exception(f"Error during browse: {e}")
            raise Exception(f"Could not browse documents: {e}") from e

        results = []
        for hit in response.get("hits", []):
            data = extract_hit_metadata(
                hit,
                self.__config.metadata_fields,
                self.__config.description_field,
            )
            data["document_id"] = hit.get("id", "")
            source_url = f"{self.__config.fe_base_url}{hit['document_url']}"
            data["source_url"] = source_url

            short_id = self.__compute_short_id(hit.get("id", ""))
            if short_id:
                data["short_id"] = short_id

            self.__citation_store.register_hit(
                source_url=source_url,
                context_hit=data,
                search_hit=hit,
                short_id=short_id,
                index_type=response.get("index_type"),
                index_id=response.get("index_id"),
            )

            results.append(data)

        return {
            "results": results,
            "total_hits": response.get("total_hits", 0),
            "page": page,
        }

    @streamable(
        running_text="Discovering filter options...",
        completed_text="Filter options loaded.",
        response_transform=hide,
    )
    async def get_filter_options(
        self,
        query: Optional[str] = None,
        field_paths: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Discover available filter fields and their values for the current index.

        Args:
            query: Optional query to scope facet values to relevant documents.
                If omitted, returns global facet values.
            field_paths: Specific field paths to get facets for.
                If omitted, returns all configured filter fields.

        Returns:
            Dict with 'filter_options' containing a list of available fields
            and their top values with counts.
        """
        if self.__config.known_filters:
            return await self.__filter_options_from_known_filters(query, field_paths)

        filter_configs = await self.__get_filter_configs()
        if not filter_configs:
            return {"filter_options": []}

        requested = set(field_paths) if field_paths else None
        filter_options: List[Dict[str, Any]] = []
        faceted_configs: List[Dict[str, Any]] = []

        for f in filter_configs:
            field_name = f["field_name"]
            if requested and field_name not in requested:
                continue

            filter_type = f.get("filter_type")
            settings = f.get("filter_type_settings") or {}

            if filter_type == "faceted_checkbox" and f.get("faceted"):
                faceted_configs.append(f)
            elif filter_type == "checkbox":
                values = self.__extract_checkbox_values(settings)
                filter_options.append(self.__make_filter_entry(f, values))
            elif filter_type == "autocomplete":
                values = self.__extract_autocomplete_values(settings)
                filter_options.append(self.__make_filter_entry(f, values))
            elif filter_type == "nested_checkbox":
                values = self.__extract_nested_checkbox_values(settings)
                filter_options.append(self.__make_filter_entry(f, values))
            else:
                entry = await self.__make_legacy_filter_entry(f)
                filter_options.append(entry)

        if faceted_configs:
            facet_results = await self.__fetch_facets(faceted_configs, query)
            for f in faceted_configs:
                field_name = f["field_name"]
                values = facet_results.get(field_name, [])
                filter_options.append(self.__make_filter_entry(f, values))

        return {"filter_options": filter_options}

    async def __filter_options_from_known_filters(
        self,
        query: Optional[str],
        field_paths: Optional[List[str]],
    ) -> Dict[str, Any]:
        hints_by_path = {f.field_path: f for f in self.__config.known_filters}
        requested = set(field_paths) if field_paths else set(hints_by_path.keys())

        static_entries: List[Dict[str, Any]] = []
        facet_paths: List[str] = []

        for fp in requested:
            hint = hints_by_path.get(fp)
            if not hint:
                continue
            if hint.values:
                entry: Dict[str, Any] = {
                    "field_path": fp,
                    "values": [{"value": v} for v in hint.values],
                }
                if hint.description:
                    entry["description"] = hint.description
                static_entries.append(entry)
            else:
                facet_paths.append(fp)

        facet_results: List[Dict[str, Any]] = []
        if facet_paths:
            facets = [{"field_path": fp} for fp in facet_paths]
            try:
                response = await self.__retriever.search(
                    query_string=query or "",
                    retrieval_unit="document",
                    retrieval_method="keyword",
                    facets=facets,
                    page=1,
                    page_size=1,
                    document_types=["document"],
                )
                facet_results = response.get("facet_results", [])
            except Exception as e:
                logger.exception(f"Error during facet discovery: {e}")

        seen_paths: set = set()
        filter_options = list(static_entries)

        for facet in facet_results:
            field_path = facet.get("field_path", "")
            seen_paths.add(field_path)
            values = [
                {"value": fvc.get("field_value"), "count": fvc.get("count", 0)}
                for fvc in facet.get("field_value_counts", [])
            ]
            entry = {"field_path": field_path, "values": values}
            hint = hints_by_path.get(field_path)
            if hint and hint.description:
                entry["description"] = hint.description
            filter_options.append(entry)

        for fp in facet_paths:
            if fp not in seen_paths:
                hint = hints_by_path.get(fp)
                if not hint:
                    continue
                entry = {"field_path": fp, "values": []}
                if hint.description:
                    entry["description"] = hint.description
                if hint.example_values:
                    entry["example_values"] = hint.example_values
                filter_options.append(entry)
                filter_options.append(entry)

        return {"filter_options": filter_options}

    async def __fetch_facets(
        self,
        faceted_configs: List[Dict[str, Any]],
        query: Optional[str],
    ) -> Dict[str, List[Dict[str, Any]]]:
        facets = []
        for f in faceted_configs:
            fc_settings = (f.get("filter_type_settings") or {}).get(
                "faceted_checkbox", {}
            )
            facet: Dict[str, Any] = {"field_path": f["field_name"]}
            if fc_settings.get("max_terms"):
                facet["max_facet_terms"] = fc_settings["max_terms"]
            if fc_settings.get("sort"):
                order: Dict[str, str] = {"by": fc_settings["sort"]}
                if fc_settings.get("sort_order"):
                    order["direction"] = fc_settings["sort_order"]
                facet["facet_terms_order"] = order
            facets.append(facet)

        try:
            response = await self.__retriever.search(
                query_string=query or "",
                retrieval_unit="document",
                retrieval_method="keyword",
                facets=facets,
                page=1,
                page_size=1,
                document_types=["document"],
            )
        except Exception as e:
            logger.exception(f"Error during facet fetch: {e}")
            return {}

        result: Dict[str, List[Dict[str, Any]]] = {}
        for facet_result in response.get("facet_results", []):
            field_path = facet_result.get("field_path", "")
            result[field_path] = [
                {"value": fvc.get("field_value"), "count": fvc.get("count", 0)}
                for fvc in facet_result.get("field_value_counts", [])
            ]
        return result

    __LEGACY_STATIC_VALUES: Dict[str, List[Dict[str, Any]]] = {
        "document_type": [
            {"value": "document"},
            {"value": "note"},
            {"value": "citation"},
        ],
        "allow_access_rights": [
            {"value": "public"},
            {"value": "private"},
        ],
        "metadata.github_metrics": [
            {"value": "true", "label": "Has code"},
            {"value": "false", "label": "No code"},
        ],
    }

    __LEGACY_FREETEXT_HINTS: Dict[str, str] = {
        "metadata.DCMI.date": "Date range filter. Use date values like '2024-01-01'.",
        "metadata.DCMI.creator.location.country": "Freetext country name.",
        "metadata.DCMI.creator.organization.name": "Freetext organization name.",
    }

    async def __make_legacy_filter_entry(self, f: Dict[str, Any]) -> Dict[str, Any]:
        field_name = f["field_name"]

        if field_name == "metadata.DCMI.source":
            sources = await self.__retriever.get_sources()
            values = [{"value": s.get("title")} for s in sources if s.get("title")]
            return self.__make_filter_entry(f, values)

        if field_name in self.__LEGACY_STATIC_VALUES:
            return self.__make_filter_entry(f, self.__LEGACY_STATIC_VALUES[field_name])

        entry = self.__make_filter_entry(f, [])
        hint = self.__LEGACY_FREETEXT_HINTS.get(field_name)
        if hint:
            entry["hint"] = hint
        return entry

    @staticmethod
    def __make_filter_entry(
        f: Dict[str, Any], values: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        entry: Dict[str, Any] = {
            "field_path": f["field_name"],
            "values": values,
        }
        if f.get("display_name"):
            entry["description"] = f["display_name"]
        if f.get("filter_type"):
            entry["filter_type"] = f["filter_type"]
        return entry

    @staticmethod
    def __extract_checkbox_values(
        settings: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        checkbox = settings.get("checkbox", {}) or {}
        return [
            {"value": v.get("value"), "label": v.get("label")}
            for v in (checkbox.get("values") or [])
        ]

    @staticmethod
    def __extract_autocomplete_values(
        settings: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        autocomplete = settings.get("autocomplete", {}) or {}
        return [{"value": v} for v in (autocomplete.get("values") or [])]

    @staticmethod
    def __extract_nested_checkbox_values(
        settings: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        nested = settings.get("nested_checkbox", {}) or {}
        return [{"value": v} for v in (nested.get("values") or [])]

    @streamable(
        running_text="Inspecting document fields...",
        completed_text="Document fields loaded.",
        response_transform=hide,
    )
    async def get_document_fields(
        self,
        query: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Discover metadata fields available on documents by inspecting a sample.

        Fetches a small number of documents and extracts all field paths
        found in their metadata. Use this to understand what metadata is
        available before deciding which fields to filter on or extract.

        Args:
            query: Optional query to scope the sample. If omitted, returns
                fields from a broad sample of documents.

        Returns:
            Dict with 'fields' (list of discovered field paths with type and
            sample value) and 'configured_fields' (the fields currently
            returned by search results).
        """
        try:
            response = await self.__retriever.search(
                query_string=query or "",
                retrieval_unit="document",
                retrieval_method="keyword",
                page=1,
                page_size=3,
                document_types=["document"],
            )
        except Exception as e:
            logger.exception(f"Error during document field discovery: {e}")
            raise Exception(f"Could not discover document fields: {e}") from e

        hits = response.get("hits", [])
        if not hits:
            return {
                "fields": [],
                "configured_fields": self.__config.metadata_fields,
            }

        all_paths: Dict[str, Dict[str, Any]] = {}
        for hit in hits:
            for path, value in _walk_fields(hit, ""):
                if path not in all_paths:
                    sample = str(value)[:200] if value is not None else None
                    all_paths[path] = {
                        "field_path": path,
                        "type": type(value).__name__,
                        "sample": sample,
                    }

        return {
            "fields": sorted(all_paths.values(), key=lambda f: f["field_path"]),
            "configured_fields": self.__config.metadata_fields,
        }

    @streamable(
        running_text="Loading index configuration...",
        completed_text="Index configuration loaded.",
        response_transform=hide,
    )
    async def get_index_configuration(self) -> Dict[str, Any]:
        """Retrieve the index configuration from the tenant settings.

        Returns the configured filter fields (with display names and types),
        sort options, available document sources, and index metadata.
        Use this to understand what filters, sort fields, and sources are
        available before searching or browsing.

        Returns:
            Dict with 'filters' (configured filter fields with field_name,
            display_name, filter_type), 'sort_options' (available sort fields),
            'sources' (document source taxonomy), and 'indexes' (available
            indexes with titles).
        """
        try:
            tenant_settings = await self.__retriever.get_tenant_settings()
        except Exception as e:
            logger.exception(f"Error fetching tenant settings: {e}")
            raise Exception(f"Could not fetch tenant settings: {e}") from e

        try:
            sources = await self.__retriever.get_sources()
        except Exception as e:
            logger.exception(f"Error fetching sources: {e}")
            sources = []

        indexes = []
        filters = []
        sort_options = []

        for idx in self.__internal_indexes(tenant_settings):
            indexes.append(
                {
                    "index_id": idx.get("index_id"),
                    "title": idx.get("title"),
                }
            )
            for f in idx.get("search_filters_config", []):
                entry: Dict[str, Any] = {
                    "field_name": f.get("field_name"),
                    "display_name": f.get("display_name"),
                }
                if f.get("filter_type"):
                    entry["filter_type"] = f["filter_type"]
                if f.get("faceted"):
                    entry["faceted"] = True
                if f.get("default_values"):
                    entry["default_values"] = f["default_values"]
                filters.append(entry)
            for s in idx.get("search_sorting_config", []):
                sort_options.append(
                    {
                        "field_name": s.get("field_name"),
                        "display_name": s.get("display_name"),
                    }
                )

        return {
            "filters": filters,
            "sort_options": sort_options,
            "sources": sources,
            "indexes": indexes,
        }


def _walk_fields(obj: Dict[str, Any], prefix: str) -> List[tuple]:
    results: List[tuple] = []
    for key, value in obj.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            results.extend(_walk_fields(value, path))
        else:
            results.append((path, value))
    return results


class IndexToolsSourceFactory(AgentDependencyFactory):
    @classmethod
    def create(
        cls,
        zav_retriever: ZAVRetriever,
        citation_store: CitationStore,
        index_tools_source_configuration: IndexToolsSourceConfiguration = (
            IndexToolsSourceConfiguration()
        ),
        citation_configuration: CitationConfiguration = CitationConfiguration(),
    ) -> IndexToolsSource:
        return IndexToolsSource(
            retriever=zav_retriever,
            index_tools_source_configuration=index_tools_source_configuration,
            citation_config=citation_configuration,
            citation_store=citation_store,
        )
