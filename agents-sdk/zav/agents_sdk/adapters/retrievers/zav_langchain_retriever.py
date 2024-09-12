import asyncio
from typing import Any, Dict, List, Literal, Optional

from langchain_core.documents.base import Document
from langchain_core.retrievers import BaseRetriever

from zav.agents_sdk.adapters.retrievers.zav_retriever import (
    ZAVRetriever,
    ZAVRetrieverFactory,
)
from zav.agents_sdk.domain.agent_dependency import AgentDependencyFactory
from zav.agents_sdk.domain.request_headers import RequestHeaders


class ZAVLangchainRetriever(BaseRetriever):
    document_retriever: ZAVRetriever
    index_id: Optional[str] = None
    retrieval_unit: Literal["document", "chunk"] = "document"
    retrieval_method: Optional[Literal["knn", "keyword", "mixed"]] = None
    filters: Optional[Dict] = None
    facets: Optional[List[Dict]] = None
    search_engine: Optional[
        Literal["zeta_alpha", "google_scholar", "bing", "google"]
    ] = None
    include_default_filters: Optional[bool] = None
    page: Optional[int] = None
    page_size: Optional[int] = None
    rerank: Optional[bool] = None
    rerank_top_n: Optional[int] = None
    sort: Optional[Dict] = None
    sort_order: Optional[List[str]] = None
    collapse: Optional[str] = "__NOT_SET"
    doc_ids: Optional[List[str]] = None
    visibility: Optional[List[str]] = None
    document_types: Optional[List[str]] = None

    def _get_relevant_documents(self, query: str, *, run_manager) -> List[Document]:
        return asyncio.run(
            self._aget_relevant_documents(query, run_manager=run_manager)
        )

    async def _aget_relevant_documents(
        self, query: str, *, run_manager
    ) -> List[Document]:
        search_response = await self.document_retriever.search(
            retrieval_unit=self.retrieval_unit,
            retrieval_method=self.retrieval_method,
            filters=self.filters,
            facets=self.facets,
            sort=self.sort,
            sort_order=self.sort_order,
            search_engine=self.search_engine,
            query_string=query,
            include_default_filters=self.include_default_filters,
            page=self.page,
            page_size=self.page_size,
            rerank=self.rerank,
            rerank_top_n=self.rerank_top_n,
            index_id=self.index_id,
            collapse=self.collapse,
            doc_ids=self.doc_ids,
            visibility=self.visibility,
            document_types=self.document_types,
        )
        hits = search_response.get("hits", [])
        docs: List[Document] = []
        for hit in hits:
            docs.append(
                Document(
                    page_content=hit.get("highlight", ""),
                    metadata=hit.get("custom_metadata", {}).get("metadata", {}),
                )
            )
        return docs


class ZAVLangchainStore:
    def __init__(self, document_retriever: ZAVRetriever):
        self.__document_retriever = document_retriever

    def as_retriever(
        self,
        index_id: Optional[str] = None,
        retrieval_unit: Literal["document", "chunk"] = "document",
        retrieval_method: Optional[Literal["knn", "keyword", "mixed"]] = None,
        filters: Optional[Dict] = None,
        facets: Optional[List[Dict]] = None,
        search_engine: Optional[
            Literal["zeta_alpha", "google_scholar", "bing", "google"]
        ] = None,
        include_default_filters: Optional[bool] = None,
        page: Optional[int] = None,
        page_size: Optional[int] = None,
        rerank: Optional[bool] = None,
        rerank_top_n: Optional[int] = None,
        sort: Optional[Dict] = None,
        sort_order: Optional[List[str]] = None,
        collapse: Optional[str] = "__NOT_SET",
        doc_ids: Optional[List[str]] = None,
        visibility: Optional[List[str]] = None,
        document_types: Optional[List[str]] = None,
        **kwargs: Any
    ) -> ZAVLangchainRetriever:
        return ZAVLangchainRetriever(
            document_retriever=self.__document_retriever,
            index_id=index_id,
            retrieval_unit=retrieval_unit,
            retrieval_method=retrieval_method,
            filters=filters,
            facets=facets,
            sort=sort,
            sort_order=sort_order,
            search_engine=search_engine,
            include_default_filters=include_default_filters,
            page=page,
            page_size=page_size,
            rerank=rerank,
            rerank_top_n=rerank_top_n,
            collapse=collapse,
            doc_ids=doc_ids,
            visibility=visibility,
            document_types=document_types,
            **kwargs
        )


class ZAVLangchainStoreFactory(AgentDependencyFactory):
    @classmethod
    def create(
        cls,
        request_headers: RequestHeaders,
        zav_retriever_host: str = "https://api.zeta-alpha.com/v0/service",
        tenant: str = "zetaalpha",
        index_id: Optional[str] = None,
        authorization: Optional[str] = None,
        x_auth: Optional[str] = None,
        retries: Optional[int] = None,
    ) -> ZAVLangchainStore:
        document_retriever = ZAVRetrieverFactory.create(
            request_headers=request_headers,
            zav_retriever_host=zav_retriever_host,
            tenant=tenant,
            index_id=index_id,
            authorization=authorization,
            x_auth=x_auth,
            retries=retries,
        )
        return ZAVLangchainStore(document_retriever=document_retriever)
