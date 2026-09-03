from fastapi import APIRouter
from app.schemas.model import ModelListResponse
from worker.ml.registry import model_registry

router = APIRouter(prefix="/api/v1/models", tags=["ML Model Registry"])


@router.get("", response_model=ModelListResponse, summary="List Registered ML Models")
def list_registered_models():
    """Retrieves all registered ML models, versions, required bands, and class labels"""
    models_metadata = model_registry.list_models()
    return {"models": models_metadata}
