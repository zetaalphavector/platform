from dataclasses import asdict, dataclass, make_dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Literal,
    Optional,
    Type,
    TypeVar,
    Union,
    cast,
)

from zav.logging import logger

from zav.message_bus.common import Command, Event, Message
from zav.message_bus.errors import HandlerException
from zav.message_bus.handler_registry import (
    CommandHandlerRegistry,
    EventHandlerRegistry,
)

FILTER_SPECS_ = TypeVar("FILTER_SPECS_")
DOMAIN_MODEL_ = TypeVar("DOMAIN_MODEL_")

CRUD_TYPE = Union[
    Literal["create"],
    Literal["retrieve"],
    Literal["filter"],
    Literal["update"],
    Literal["replace"],
    Literal["delete"],
]


@dataclass
class HandlerMixin:

    crud_type: CRUD_TYPE
    command_cls: Type[Command]
    publish_event: bool = False
    event_cls: Optional[Type[Event]] = None


class HandlersFactory(Generic[FILTER_SPECS_, DOMAIN_MODEL_]):

    command_handler_registry: Type[CommandHandlerRegistry]
    event_handler_registry: Type[EventHandlerRegistry]
    handled_events: Dict[CRUD_TYPE, Type[Event]]

    def __init__(
        self,
        handler_mixins: List[HandlerMixin],
        command_handler_registry: Type[CommandHandlerRegistry],
        event_handler_registry: Type[EventHandlerRegistry],
        domain_model_factory: Callable[[Any], DOMAIN_MODEL_],
        domain_model_annotations: Dict[str, Any],
        domain_filter_specifications: Type[FILTER_SPECS_],
        repo_name: str,
        resource_name: str,
    ) -> None:

        self.command_handler_registry = command_handler_registry
        self.event_handler_registry = event_handler_registry
        self.domain_model_factory = domain_model_factory
        self.domain_model_annotations = domain_model_annotations
        self.domain_filter_specifications = domain_filter_specifications
        self.repo_name = repo_name
        self.resource_name = resource_name
        self.handled_events = {}

        for handler_mixin in handler_mixins:
            event_cls = self.__create_event_cls(handler_mixin=handler_mixin)
            if event_cls is not None:
                self.__create_event_handler(event_cls)

            if handler_mixin.crud_type == "create":
                self.__create_mixin(handler_mixin, event_cls)
            elif handler_mixin.crud_type == "retrieve":
                self.__retrieve_mixin(handler_mixin, event_cls)
            elif handler_mixin.crud_type == "filter":
                self.__filter_mixin(handler_mixin, event_cls)
            elif handler_mixin.crud_type == "update":
                self.__update_mixin(handler_mixin, event_cls)
            elif handler_mixin.crud_type == "replace":
                self.__replace_mixin(handler_mixin, event_cls)
            elif handler_mixin.crud_type == "delete":
                self.__delete_mixin(handler_mixin, event_cls)

    def __create_mixin(
        self, handler_mixin: HandlerMixin, event_cls: Optional[type] = None
    ):
        @self.command_handler_registry.register(handler_mixin.command_cls)
        async def func(cmd, queue: List[Message], repos):
            payload = getattr(cmd, "payload", None)
            if payload is None:
                logger.warn("Create handler cmd must have payload.")
                return None
            repo = getattr(repos, self.repo_name, None)
            if repo is None:
                logger.warn(f"Create handler repos must contain {self.repo_name}.")
                return None

            domain_model = self.domain_model_factory(payload)
            created_domain_model = await repo.add(domain_model)
            if not created_domain_model:
                return None
            if handler_mixin.publish_event and event_cls is not None:
                queue.append(event_cls(**created_domain_model.dict()))
            return created_domain_model

    def __retrieve_mixin(
        self, handler_mixin: HandlerMixin, event_cls: Optional[type] = None
    ):
        @self.command_handler_registry.register(handler_mixin.command_cls)
        async def func(cmd, queue: List[Message], repos):
            query_params = getattr(cmd, "query_params", None)
            path_params = getattr(cmd, "path_params", None)
            if path_params is None:
                logger.warn("Retrieve handler cmd must have path_params.")
                return None
            repo = getattr(repos, self.repo_name, None)
            if repo is None:
                logger.warn(f"Retrieve handler repos must contain {self.repo_name}.")
                return None
            match_one = getattr(self.domain_filter_specifications, "one", None)
            if match_one is None:
                logger.warn("Filter specifications must contain 'one'.")
                return None
            try:
                get_specification = match_one(**asdict(path_params))
            except TypeError as e:
                logger.warn(
                    "Filter specifications 'one' does not work with "
                    f"given path_params: {e}."
                )
                return None

            filter_spec = None
            if query_params is not None:
                filter_specs = []
                for param_name, param_value in asdict(query_params).items():
                    if param_value is not None:
                        filter_spec_meth = getattr(
                            self.domain_filter_specifications, param_name, None
                        )
                        if filter_spec_meth is None:
                            logger.warn(
                                f"Filter specifications must contain '{param_name}'."
                            )
                            return None
                        try:
                            spec = filter_spec_meth(param_value)
                        except TypeError as e:
                            logger.warn(
                                f"Filter specifications '{param_name}' does not"
                                f" work with given query_param: {e}."
                            )
                            return None
                        else:
                            filter_specs.append(spec)
                if len(filter_specs) >= 1:
                    filter_spec = filter_specs.pop(0)
                    for spec in filter_specs:
                        filter_spec = filter_spec.__and__(spec)

            domain_model = await repo.get(get_specification, filter_spec)
            if not domain_model:
                return None

            return domain_model

    def __filter_mixin(
        self, handler_mixin: HandlerMixin, event_cls: Optional[type] = None
    ):
        @self.command_handler_registry.register(handler_mixin.command_cls)
        async def func(cmd, queue: List[Message], repos):
            start_position = cmd.page_size * (cmd.page - 1)
            stop_position = cmd.page_size * cmd.page

            query_params = getattr(cmd, "query_params", None)
            repo = getattr(repos, self.repo_name, None)
            if repo is None:
                logger.warn(f"Filter handler repos must contain {self.repo_name}.")
                return None
            match_all = getattr(self.domain_filter_specifications, "all", None)
            if match_all is None:
                logger.warn("Filter specifications must contain 'all'.")
                return None

            filter_spec = match_all()

            if query_params is not None:
                filter_specs = []
                for param_name, param_value in asdict(query_params).items():
                    if param_value is not None:
                        filter_spec_meth = getattr(
                            self.domain_filter_specifications, param_name, None
                        )
                        if filter_spec_meth is None:
                            logger.warn(
                                f"Filter specifications must contain '{param_name}'."
                            )
                            return None
                        try:
                            spec = filter_spec_meth(param_value)
                        except TypeError as e:
                            logger.warn(
                                f"Filter specifications '{param_name}' does not"
                                f" work with given query_param: {e}."
                            )
                            return None
                        else:
                            filter_specs.append(spec)
                if len(filter_specs) >= 1:
                    filter_spec = filter_specs.pop(0)
                    for spec in filter_specs:
                        filter_spec = filter_spec.__and__(spec)

            domain_model_seq = repo.filter(filter_spec)
            domain_models = await domain_model_seq.slice(start_position, stop_position)
            total = await domain_model_seq.len()

            return total, domain_models, cmd.page, cmd.page_size

    def __update_mixin(
        self, handler_mixin: HandlerMixin, event_cls: Optional[type] = None
    ):
        @self.command_handler_registry.register(handler_mixin.command_cls)
        async def func(cmd, queue: List[Message], repos):
            payload = getattr(cmd, "payload", None)
            if payload is None:
                logger.warn("Update handler cmd must have payload.")
                return None
            query_params = getattr(cmd, "query_params", None)
            path_params = getattr(cmd, "path_params", None)
            if path_params is None:
                logger.warn("Update handler cmd must have path_params.")
                return None
            repo = getattr(repos, self.repo_name, None)
            if repo is None:
                logger.warn(f"Update handler repos must contain {self.repo_name}.")
                return None
            match_one = getattr(self.domain_filter_specifications, "one", None)
            if match_one is None:
                logger.warn("Filter specifications must contain 'one'.")
                return None
            try:
                get_specification = match_one(**asdict(path_params))
            except TypeError as e:
                logger.warn(
                    "Filter specifications 'one' does not work with "
                    f"given path_params: {e}."
                )
                return None

            filter_spec = None
            if query_params is not None:
                filter_specs = []
                for param_name, param_value in asdict(query_params).items():
                    if param_value is not None:
                        filter_spec_meth = getattr(
                            self.domain_filter_specifications, param_name, None
                        )
                        if filter_spec_meth is None:
                            logger.warn(
                                f"Filter specifications must contain '{param_name}'."
                            )
                            return None
                        try:
                            spec = filter_spec_meth(param_value)
                        except TypeError as e:
                            logger.warn(
                                f"Filter specifications '{param_name}' does not"
                                f" work with given query_param: {e}."
                            )
                            return None
                        else:
                            filter_specs.append(spec)
                if len(filter_specs) >= 1:
                    filter_spec = filter_specs.pop(0)
                    for spec in filter_specs:
                        filter_spec = filter_spec.__and__(spec)

            existing_domain_model = await repo.get(get_specification, filter_spec)
            if not existing_domain_model:
                return None

            payload_data = payload.dict(exclude_unset=True)
            updated_existing_domain_model = existing_domain_model.copy(
                update=payload_data
            )

            # for key, value in payload_data.items():
            #     setattr(existing_domain_model, key, value)

            updated_domain_model = await repo.add(updated_existing_domain_model)
            if not updated_domain_model:
                raise HandlerException(f"Could not update {self.resource_name}.")
            if handler_mixin.publish_event and event_cls is not None:
                queue.append(event_cls(**updated_domain_model.dict()))
            return updated_domain_model

    def __replace_mixin(
        self, handler_mixin: HandlerMixin, event_cls: Optional[type] = None
    ):
        @self.command_handler_registry.register(handler_mixin.command_cls)
        async def func(cmd, queue: List[Message], repos):
            payload = getattr(cmd, "payload", None)
            if payload is None:
                logger.warn("Replace handler cmd must have payload.")
                return None
            query_params = getattr(cmd, "query_params", None)
            path_params = getattr(cmd, "path_params", None)
            if path_params is None:
                logger.warn("Replace handler cmd must have path_params.")
                return None
            repo = getattr(repos, self.repo_name, None)
            if repo is None:
                logger.warn(f"Replace handler repos must contain {self.repo_name}.")
                return None
            match_one = getattr(self.domain_filter_specifications, "one", None)
            if match_one is None:
                logger.warn("Filter specifications must contain 'one'.")
                return None
            try:
                get_specification = match_one(**asdict(path_params))
            except TypeError as e:
                logger.warn(
                    "Filter specifications 'one' does not work with "
                    f"given path_params: {e}."
                )
                return None

            filter_spec = None
            if query_params is not None:
                filter_specs = []
                for param_name, param_value in asdict(query_params).items():
                    if param_value is not None:
                        filter_spec_meth = getattr(
                            self.domain_filter_specifications, param_name, None
                        )
                        if filter_spec_meth is None:
                            logger.warn(
                                f"Filter specifications must contain '{param_name}'."
                            )
                            return None
                        try:
                            spec = filter_spec_meth(param_value)
                        except TypeError as e:
                            logger.warn(
                                f"Filter specifications '{param_name}' does not"
                                f" work with given query_param: {e}."
                            )
                            return None
                        else:
                            filter_specs.append(spec)
                if len(filter_specs) >= 1:
                    filter_spec = filter_specs.pop(0)
                    for spec in filter_specs:
                        filter_spec = filter_spec.__and__(spec)

            existing_domain_model = await repo.get(get_specification, filter_spec)
            if not existing_domain_model:
                return None

            payload_data = payload.dict(exclude_unset=True)
            for key, value in payload_data.items():
                setattr(existing_domain_model, key, value)

            replaced_domain_model = await repo.add(existing_domain_model)
            if not replaced_domain_model:
                raise HandlerException(f"Could not replace {self.resource_name}.")
            if handler_mixin.publish_event and event_cls is not None:
                queue.append(event_cls(**replaced_domain_model.dict()))
            return replaced_domain_model

    def __delete_mixin(
        self, handler_mixin: HandlerMixin, event_cls: Optional[type] = None
    ):
        @self.command_handler_registry.register(handler_mixin.command_cls)
        async def func(cmd, queue: List[Message], repos):
            query_params = getattr(cmd, "query_params", None)
            path_params = getattr(cmd, "path_params", None)
            if path_params is None:
                logger.warn("Delete handler cmd must have path_params.")
                return None
            repo = getattr(repos, self.repo_name, None)
            if repo is None:
                logger.warn(f"Delete handler repos must contain {self.repo_name}.")
                return None
            match_one = getattr(self.domain_filter_specifications, "one", None)
            if match_one is None:
                logger.warn("Filter specifications must contain 'one'.")
                return None
            try:
                get_specification = match_one(**asdict(path_params))
            except TypeError as e:
                logger.warn(
                    "Filter specifications 'one' does not work with "
                    f"given path_params: {e}."
                )
                return None

            filter_spec = None
            if query_params is not None:
                filter_specs = []
                for param_name, param_value in asdict(query_params).items():
                    if param_value is not None:
                        filter_spec_meth = getattr(
                            self.domain_filter_specifications, param_name, None
                        )
                        if filter_spec_meth is None:
                            logger.warn(
                                f"Filter specifications must contain '{param_name}'."
                            )
                            return None
                        try:
                            spec = filter_spec_meth(param_value)
                        except TypeError as e:
                            logger.warn(
                                f"Filter specifications '{param_name}' does not"
                                f" work with given query_param: {e}."
                            )
                            return None
                        else:
                            filter_specs.append(spec)
                if len(filter_specs) >= 1:
                    filter_spec = filter_specs.pop(0)
                    for spec in filter_specs:
                        filter_spec = filter_spec.__and__(spec)

            existing_domain_model = await repo.get(get_specification, filter_spec)
            if not existing_domain_model:
                return None

            deleted_domain_model = await repo.delete(existing_domain_model)
            if not deleted_domain_model:
                return None
            if handler_mixin.publish_event and event_cls is not None:
                queue.append(event_cls(**deleted_domain_model.dict()))
            return deleted_domain_model

    def __create_event_cls(self, handler_mixin: HandlerMixin) -> Optional[Type[Event]]:
        if not handler_mixin.publish_event:
            return None
        if handler_mixin.event_cls is not None:
            self.handled_events.update(
                {handler_mixin.crud_type: handler_mixin.event_cls}
            )
            return handler_mixin.event_cls
        else:
            command_cls_name = handler_mixin.command_cls.__name__
            verb = handler_mixin.crud_type.title()
            verb_past_tense = f"{verb}d"

            event_fields = [
                (field_name, field_type)
                for field_name, field_type in self.domain_model_annotations.items()
            ]
            event_cls = cast(
                Type[Event],
                make_dataclass(
                    command_cls_name.replace(verb, verb_past_tense),
                    fields=event_fields,
                    bases=(Event,),
                ),
            )
            self.handled_events.update({handler_mixin.crud_type: event_cls})
            return event_cls

    def __create_event_handler(self, event_cls: Type[Event]):
        @self.event_handler_registry.register(event_cls)
        async def func(event: Type[Event], queue: List[Message], repos):
            repo = getattr(repos, "events_publisher", None)
            if repo is None:
                logger.warn("Event handler repos must contain events_publisher.")
                return None
            await repo.enqueue(event)
