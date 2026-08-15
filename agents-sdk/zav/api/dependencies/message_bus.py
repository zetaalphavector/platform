from zav.message_bus import MessageBus

from zav.api.app_resources import AppResource

get_message_bus = AppResource[MessageBus]("message_bus")
