"""Pieces shared by the two websites' servers (careers_app.py and main.py)."""
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.exc import SQLAlchemyError

from src.config import ROOT
from src.utils.logging_utils import get_logger

WEB_DIR = ROOT / "web"
log = get_logger("api")


def install_error_handlers(app: FastAPI):
    @app.exception_handler(SQLAlchemyError)
    async def db_error(request: Request, exc: SQLAlchemyError):
        log.exception("Database error")
        return JSONResponse(status_code=503, content={"detail": "Database error. Please try again."})

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        fields = [".".join(str(p) for p in e["loc"][1:]) for e in exc.errors()]
        return JSONResponse(status_code=422, content={"detail": f"Invalid or missing fields: {', '.join(fields)}"})

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception):
        log.exception("Unhandled error")
        return JSONResponse(status_code=500, content={"detail": "Internal server error. See server logs."})


def mount_website(app: FastAPI, folder: str):
    """Serve one website at '/'. Must be called AFTER the API routes so they take priority."""
    app.mount("/assets", StaticFiles(directory=WEB_DIR / "assets"), name="assets")
    app.mount("/", StaticFiles(directory=WEB_DIR / folder, html=True), name=folder)
