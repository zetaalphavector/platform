from fastapi import APIRouter

from zav.agents_sdk.controllers import v1

routers = [
    ("", router) for router in v1.__dict__.values() if isinstance(router, APIRouter)
]
