import time
from fastapi import APIRouter
from app.config import settings

router = APIRouter(tags=["Health & Status"])
START_TIME = time.time()


@router.get("/health", summary="System Health Check")
def health_check():
    """Returns application health status and uptime"""
    uptime_seconds = round(time.time() - START_TIME, 2)
    return {
        "status": "healthy",
        "app_name": settings.APP_NAME,
        "gee_project_id": settings.GEE_PROJECT_ID,
        "uptime_seconds": uptime_seconds
    }


@router.get("/metrics", summary="Prometheus Metrics Endpoint")
def metrics():
    """Basic Prometheus style telemetry metrics"""
    return {
        "geoserve_uptime_seconds": round(time.time() - START_TIME, 2),
        "geoserve_status": 1
    }
