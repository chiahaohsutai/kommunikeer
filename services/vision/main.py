from contextlib import asynccontextmanager

from fastapi import FastAPI

from models.ocr import TextExtractor
from routes import router as vision_router

_prefix = "/api"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI application."""

    extractor = TextExtractor()
    yield
    extractor.model.close()


app = FastAPI(
    title="Kommunikeer - Vision Service",
    version="1.0.0",
    prefix=_prefix,
    docs_url=f"{_prefix}/docs",
    lifespan=lifespan,
)
app.include_router(vision_router)


@app.get("/health")
async def health_check():
    """Health check endpoint."""

    return {"status": "ok"}
