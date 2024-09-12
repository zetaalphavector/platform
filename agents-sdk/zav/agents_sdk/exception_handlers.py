from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


def add_exception_handlers(app: FastAPI):
    @app.exception_handler(ValueError)
    async def value_error_request_handler(request: Request, exc: ValueError):
        return JSONResponse(status_code=400, content={"detail": str(exc)})
