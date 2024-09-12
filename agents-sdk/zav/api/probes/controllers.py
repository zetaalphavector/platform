from fastapi import APIRouter, Depends
from zav.message_bus import MessageBus

from zav.api.dependencies.message_bus import get_message_bus
from zav.api.errors import UnknownException
from zav.api.probes.commands import CheckIfReady

probes_router = APIRouter(prefix="/probes")


@probes_router.get("/alive")
async def alive():

    return None


@probes_router.get("/ready")
async def ready(message_bus: MessageBus = Depends(get_message_bus)):

    try:
        results = await message_bus.handle(CheckIfReady())
        is_ready = results.pop(0)
    except Exception as e:
        raise UnknownException(e)
    if is_ready:
        return None
    else:
        raise UnknownException
