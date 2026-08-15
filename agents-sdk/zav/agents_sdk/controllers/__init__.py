from fastapi import APIRouter

from zav.agents_sdk.controllers import v1


def is_public_router(router: APIRouter) -> bool:
    return bool(getattr(router, "public", False))


routers = [
    ("", router) for router in v1.__dict__.values() if isinstance(router, APIRouter)
]
public_routers = [
    (prefix, router) for prefix, router in routers if is_public_router(router)
]
secured_routers = [
    (prefix, router) for prefix, router in routers if not is_public_router(router)
]
routers = public_routers + secured_routers
