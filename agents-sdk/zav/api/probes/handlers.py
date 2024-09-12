from typing import List

from zav.message_bus import CommandHandlerRegistry, Message

from zav.api.probes.commands import CheckIfReady


@CommandHandlerRegistry.register(CheckIfReady)
async def check_if_ready(cmd: CheckIfReady, queue: List[Message]):
    return True
