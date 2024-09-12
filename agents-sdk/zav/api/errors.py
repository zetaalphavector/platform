from zav.message_bus import HandlerException


class NotFoundException(HandlerException):
    pass


class UnknownException(Exception):
    pass


class BadRequestException(HandlerException):
    pass


class ForbiddenException(HandlerException):
    pass
