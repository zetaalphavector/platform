from dataclasses import dataclass, make_dataclass
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Type, Union, cast

from fastapi import APIRouter, Body, Depends, Response
from pydantic import BaseModel
from pydantic.version import VERSION as PYDANTIC_VERSION
from zav.logging import logger
from zav.message_bus import Command, MessageBus

from zav.api.dependencies import get_message_bus, pagination
from zav.api.errors import NotFoundException, UnknownException
from zav.api.response_models import PaginatedResponse

PYDANTIC_V2 = PYDANTIC_VERSION.startswith("2.")
CRUD_TYPE = Union[
    Literal["create"],
    Literal["retrieve"],
    Literal["filter"],
    Literal["update"],
    Literal["replace"],
    Literal["delete"],
]


def _to_camel(string: str) -> str:
    return "".join(word.capitalize() for word in string.split("_"))


def _create_operation_name(
    crud_type: CRUD_TYPE, resource_name: str, sub_resource_name: Optional[str] = None
):
    infix = f"{sub_resource_name}_in_" if sub_resource_name else ""
    return f"{crud_type}_{infix}{resource_name}"


# TODO: Refactor mixin to pass higher level params instead of path and query params.
# For example divide into dict used for finding one item, for configuring the filter
# specifications, and payload. The client of ControllersFactory should be responsible
# for shaping these params correctly via fastapi dependencies.
@dataclass
class CrudMixin:

    crud_type: CRUD_TYPE
    path_url: str
    response_model: Optional[Type[BaseModel]] = None
    form_model: Optional[Any] = None
    body_params: Optional[Any] = None
    path_params: Optional[Any] = None
    query_params: Optional[Any] = None
    dependencies: Optional[List[Any]] = None


class ControllersFactory:

    handled_commands: Dict[CRUD_TYPE, Type[Command]]
    domain_router: APIRouter

    def __init__(
        self,
        resource_name: str,
        crud_mixins: List[CrudMixin],
        router_tags: List[Union[str, Enum]],
    ) -> None:

        self.resource_name = resource_name
        self.router_tags = router_tags
        self.handled_commands = {}

        self.__create_router()
        for crud_mixin in crud_mixins:
            command_cls = self.__create_command_cls(crud_mixin=crud_mixin)
            if crud_mixin.crud_type == "create":
                self.__create_mixin(crud_mixin, command_cls)
            elif crud_mixin.crud_type == "retrieve":
                self.__retrieve_mixin(crud_mixin, command_cls)
            elif crud_mixin.crud_type == "filter":
                self.__filter_mixin(crud_mixin, command_cls)
            elif crud_mixin.crud_type == "update":
                self.__update_mixin(crud_mixin, command_cls)
            elif crud_mixin.crud_type == "replace":
                self.__replace_mixin(crud_mixin, command_cls)
            elif crud_mixin.crud_type == "delete":
                self.__delete_mixin(crud_mixin, command_cls)

    def __create_router(self):
        self.domain_router = APIRouter(tags=self.router_tags)

    def __create_mixin(self, crud_mixin: CrudMixin, command_cls: Type[Command]):
        form_model = crud_mixin.form_model
        if form_model is None:
            logger.warn("Create mixin must have a form_model.")
            return

        responses: Dict[str, Any] = {
            **(
                {"response_model": crud_mixin.response_model}
                if crud_mixin.response_model is not None
                else {}
            ),
            **(
                {"response_class": Response}
                if crud_mixin.response_model is None
                else {}
            ),
            **(
                {"dependencies": crud_mixin.dependencies}
                if crud_mixin.dependencies is not None
                else {}
            ),
        }

        body_default = (
            crud_mixin.body_params if crud_mixin.body_params is not None else Body(...)
        )

        @self.domain_router.post(
            crud_mixin.path_url,
            **responses,
            status_code=201,
            name=_create_operation_name(crud_mixin.crud_type, self.resource_name),
            operation_id=_create_operation_name(
                crud_mixin.crud_type, self.resource_name
            ),
        )
        async def func(
            body: form_model = body_default,  # type: ignore
            message_bus: MessageBus = Depends(get_message_bus),
            query_params=Depends(crud_mixin.query_params),
            path_params=Depends(crud_mixin.path_params),
        ):
            result = await self.__call_message_bus(
                message_bus=message_bus,
                command_cls=command_cls,
                crud_mixin=crud_mixin,
                body=body,
                path_params=path_params,
                query_params=query_params,
            )
            if not result:
                raise UnknownException(f"Could not create {self.resource_name}.")
            if "response_model" in responses:
                return (
                    responses["response_model"].model_validate(result)
                    if PYDANTIC_V2
                    else responses["response_model"].from_orm(result)
                )
            return None

    def __retrieve_mixin(self, crud_mixin: CrudMixin, command_cls: Type[Command]):

        responses: Dict[str, Any] = {
            **(
                {"response_model": crud_mixin.response_model}
                if crud_mixin.response_model is not None
                else {}
            ),
            **(
                {"response_class": Response}
                if crud_mixin.response_model is None
                else {}
            ),
            **(
                {"dependencies": crud_mixin.dependencies}
                if crud_mixin.dependencies is not None
                else {}
            ),
        }

        @self.domain_router.get(
            crud_mixin.path_url,
            **responses,
            name=_create_operation_name(crud_mixin.crud_type, self.resource_name),
            operation_id=_create_operation_name(
                crud_mixin.crud_type, self.resource_name
            ),
        )
        async def func(
            message_bus: MessageBus = Depends(get_message_bus),
            query_params=Depends(crud_mixin.query_params),
            path_params=Depends(crud_mixin.path_params),
        ):
            result = await self.__call_message_bus(
                message_bus=message_bus,
                command_cls=command_cls,
                crud_mixin=crud_mixin,
                path_params=path_params,
                query_params=query_params,
            )
            if not result:
                raise NotFoundException(f"{self.resource_name.title()} not found.")
            if "response_model" in responses:
                return (
                    responses["response_model"].model_validate(result)
                    if PYDANTIC_V2
                    else responses["response_model"].from_orm(result)
                )
            return None

    def __filter_mixin(self, crud_mixin: CrudMixin, command_cls: Type[Command]):

        response_model = crud_mixin.response_model
        responses: Dict[str, Any] = {
            **(
                {"response_model": response_model}
                if response_model is not None
                and response_model.__name__.startswith("PaginatedResponse")
                else (
                    {
                        "response_model": (
                            PaginatedResponse[response_model]  # type: ignore
                        )
                    }
                    if response_model is not None
                    else {}
                )
            ),
            **({"response_class": Response} if response_model is None else {}),
            **(
                {"dependencies": crud_mixin.dependencies}
                if crud_mixin.dependencies is not None
                else {}
            ),
        }

        @self.domain_router.get(
            crud_mixin.path_url,
            **responses,
            name=_create_operation_name(crud_mixin.crud_type, self.resource_name),
            operation_id=_create_operation_name(
                crud_mixin.crud_type, self.resource_name
            ),
        )
        async def func(
            pagination=Depends(pagination),
            message_bus: MessageBus = Depends(get_message_bus),
            query_params=Depends(crud_mixin.query_params),
            path_params=Depends(crud_mixin.path_params),
        ):
            total, results, page, page_size = await self.__call_message_bus(
                message_bus=message_bus,
                command_cls=command_cls,
                crud_mixin=crud_mixin,
                path_params=path_params,
                query_params=query_params,
                pagination=pagination,
            )
            if "response_model" in responses:
                return responses["response_model"](
                    count=total, results=results, page=page, page_size=page_size
                )
            return None

    def __update_mixin(self, crud_mixin: CrudMixin, command_cls: Type[Command]):
        form_model = crud_mixin.form_model
        if form_model is None:
            logger.warn("Update mixin must have a form_model.")
            return

        responses: Dict[str, Any] = {
            **(
                {"response_model": crud_mixin.response_model}
                if crud_mixin.response_model is not None
                else {}
            ),
            **(
                {"response_class": Response}
                if crud_mixin.response_model is None
                else {}
            ),
            **(
                {"dependencies": crud_mixin.dependencies}
                if crud_mixin.dependencies is not None
                else {}
            ),
        }

        body_default = (
            crud_mixin.body_params if crud_mixin.body_params is not None else Body(...)
        )

        @self.domain_router.patch(
            crud_mixin.path_url,
            status_code=204,
            **responses,
            name=_create_operation_name(crud_mixin.crud_type, self.resource_name),
            operation_id=_create_operation_name(
                crud_mixin.crud_type, self.resource_name
            ),
        )
        async def func(
            body: form_model = body_default,  # type: ignore
            message_bus: MessageBus = Depends(get_message_bus),
            query_params=Depends(crud_mixin.query_params),
            path_params=Depends(crud_mixin.path_params),
        ):
            result = await self.__call_message_bus(
                message_bus=message_bus,
                command_cls=command_cls,
                crud_mixin=crud_mixin,
                body=body,
                path_params=path_params,
                query_params=query_params,
            )
            if not result:
                raise NotFoundException(f"{self.resource_name.title()} not found.")
            if "response_model" in responses:
                return (
                    responses["response_model"].model_validate(result)
                    if PYDANTIC_V2
                    else responses["response_model"].from_orm(result)
                )
            return None

    def __replace_mixin(self, crud_mixin: CrudMixin, command_cls: Type[Command]):
        form_model = crud_mixin.form_model
        if form_model is None:
            logger.warn("Replace mixin must have a form_model.")
            return

        responses: Dict[str, Any] = {
            **(
                {"response_model": crud_mixin.response_model}
                if crud_mixin.response_model is not None
                else {}
            ),
            **(
                {"response_class": Response}
                if crud_mixin.response_model is None
                else {}
            ),
            **(
                {"dependencies": crud_mixin.dependencies}
                if crud_mixin.dependencies is not None
                else {}
            ),
        }

        body_default = (
            crud_mixin.body_params if crud_mixin.body_params is not None else Body(...)
        )

        @self.domain_router.put(
            crud_mixin.path_url,
            status_code=204,
            **responses,
            name=_create_operation_name(crud_mixin.crud_type, self.resource_name),
            operation_id=_create_operation_name(
                crud_mixin.crud_type, self.resource_name
            ),
        )
        async def func(
            body: form_model = body_default,  # type: ignore
            message_bus: MessageBus = Depends(get_message_bus),
            query_params=Depends(crud_mixin.query_params),
            path_params=Depends(crud_mixin.path_params),
        ):
            result = await self.__call_message_bus(
                message_bus=message_bus,
                command_cls=command_cls,
                crud_mixin=crud_mixin,
                body=body,
                path_params=path_params,
                query_params=query_params,
            )
            if not result:
                raise NotFoundException(f"{self.resource_name.title()} not found.")
            if "response_model" in responses:
                return (
                    responses["response_model"].model_validate(result)
                    if PYDANTIC_V2
                    else responses["response_model"].from_orm(result)
                )
            return None

    def __delete_mixin(self, crud_mixin: CrudMixin, command_cls: Type[Command]):

        responses: Dict[str, Any] = {
            **(
                {"response_model": crud_mixin.response_model}
                if crud_mixin.response_model is not None
                else {}
            ),
            **(
                {"response_class": Response}
                if crud_mixin.response_model is None
                else {}
            ),
            **(
                {"dependencies": crud_mixin.dependencies}
                if crud_mixin.dependencies is not None
                else {}
            ),
        }

        @self.domain_router.delete(
            crud_mixin.path_url,
            status_code=204,
            **responses,
            name=_create_operation_name(crud_mixin.crud_type, self.resource_name),
            operation_id=_create_operation_name(
                crud_mixin.crud_type, self.resource_name
            ),
        )
        async def func(
            message_bus: MessageBus = Depends(get_message_bus),
            query_params=Depends(crud_mixin.query_params),
            path_params=Depends(crud_mixin.path_params),
        ):
            result = await self.__call_message_bus(
                message_bus=message_bus,
                command_cls=command_cls,
                crud_mixin=crud_mixin,
                path_params=path_params,
                query_params=query_params,
            )
            if not result:
                raise NotFoundException(f"{self.resource_name.title()} not found.")
            if "response_model" in responses:
                return (
                    responses["response_model"].model_validate(result)
                    if PYDANTIC_V2
                    else responses["response_model"].from_orm(result)
                )
            return None

    async def __call_message_bus(
        self,
        message_bus: MessageBus,
        command_cls: Type[Command],
        crud_mixin: CrudMixin,
        body: Optional[Any] = None,
        path_params: Optional[Any] = None,
        query_params: Optional[Any] = None,
        pagination: Optional[dict] = None,
    ):
        results = await message_bus.handle(
            cast(type, command_cls)(
                **({"payload": body} if crud_mixin.form_model is not None else {}),
                **(
                    {"path_params": path_params}
                    if crud_mixin.path_params is not None
                    else {}
                ),
                **(
                    {"query_params": query_params}
                    if crud_mixin.query_params is not None
                    else {}
                ),
                **(
                    {"page": pagination["page"], "page_size": pagination["page_size"]}
                    if pagination is not None
                    else {}
                ),
            )
        )
        return results.pop(0)

    def __create_command_cls(self, crud_mixin: CrudMixin) -> Type[Command]:
        command_fields = [
            *(
                [("payload", crud_mixin.form_model)]
                if crud_mixin.form_model is not None
                else []
            ),
            *(
                [("path_params", crud_mixin.path_params)]
                if crud_mixin.path_params is not None
                else []
            ),
            *(
                [("query_params", crud_mixin.query_params)]
                if crud_mixin.query_params is not None
                else []
            ),
            *(
                [("page", int), ("page_size", int)]
                if crud_mixin.crud_type == "filter"
                else []
            ),
        ]
        command_cls = cast(
            Type[Command],
            make_dataclass(
                _to_camel(
                    _create_operation_name(crud_mixin.crud_type, self.resource_name)
                ),
                fields=command_fields,
                bases=(Command,),
            ),
        )
        self.handled_commands.update({crud_mixin.crud_type: command_cls})
        return command_cls
