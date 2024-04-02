import logging

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from .logserver import configure_logger
from .routers import biom_router

configure_logger()
log = logging.getLogger(__name__)
log.setLevel("INFO")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

app.include_router(biom_router.router)


@app.get("/", include_in_schema=False)
async def root():
    """Endpoint for checking if api is alive."""
    return {"message": "Alive"}

