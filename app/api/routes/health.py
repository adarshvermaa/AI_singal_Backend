from fastapi import APIRouter, Request
from app.api.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request):
    """Health check endpoint."""
    factory = getattr(request.app.state, 'exchange_factory', None)
    connections = factory.connected_exchanges if factory else []
    models_loaded = hasattr(request.app.state, 'signal_engine')
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        exchange_connections=connections,
        models_loaded=models_loaded,
    )
