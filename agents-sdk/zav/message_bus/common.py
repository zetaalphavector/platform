from typing import Union


class Event:

    pass


class Command:

    pass


Message = Union[Command, Event]
