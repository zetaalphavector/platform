from contextlib import AbstractAsyncContextManager, AsyncExitStack, asynccontextmanager
from typing import Any, AsyncIterator, Callable, Dict, Generic, List, Mapping, TypeVar

from fastapi import FastAPI, Request

T = TypeVar("T")
ApiLifespan = Callable[[FastAPI], AbstractAsyncContextManager[Mapping[str, Any]]]


class AppResource(Generic[T]):
    """FastAPI dependency handle for a resource bound during app lifespan."""

    def __init__(self, name: str) -> None:

        self.__name = name
        self.__state_key = f"zav_api_resource_{name}"

    @property
    def state_key(self) -> str:

        return self.__state_key

    def bind(self, value: T) -> Dict[str, T]:

        return {self.__state_key: value}

    async def __call__(self, request: Request) -> T:

        if hasattr(request.app.state, self.__state_key):
            return getattr(request.app.state, self.__state_key)

        request_state = request.scope.get("state", {})
        if self.__state_key in request_state:
            return request_state[self.__state_key]

        raise RuntimeError(f"App resource is not configured: {self.__name}")


def compose_lifespans(lifespans: List[ApiLifespan]) -> ApiLifespan:
    """Compose stateful lifespans; startup runs in order, shutdown in reverse."""

    @asynccontextmanager
    async def composed_lifespan(app: FastAPI) -> AsyncIterator[Dict[str, Any]]:
        state: Dict[str, Any] = {}
        app_state_keys: List[str] = []
        try:
            async with AsyncExitStack() as stack:
                for lifespan in lifespans:
                    resource_state = await stack.enter_async_context(lifespan(app))
                    if not resource_state:
                        continue
                    for key, value in resource_state.items():
                        if key in state:
                            raise RuntimeError(
                                f"App resource is already configured: {key}"
                            )
                        setattr(app.state, key, value)
                        app_state_keys.append(key)
                        state[key] = value
                yield state
        finally:
            for key in reversed(app_state_keys):
                if hasattr(app.state, key):
                    delattr(app.state, key)

    return composed_lifespan
