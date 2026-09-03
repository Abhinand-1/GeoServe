import json
import os
import uuid
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse
from sqlmodel import Session, select

from app.db.database import get_session
from app.db.models import InferenceJob, JobResult
from app.schemas.job import JobCreateRequest, JobResponse, JobResultResponse
from app.services.redis_queue import job_queue_service
from worker.ml.registry import model_registry

router = APIRouter(prefix="/api/v1/jobs", tags=["Inference Jobs"])


@router.post("", response_model=JobResponse, status_code=status.HTTP_202_ACCEPTED, summary="Create Satellite Inference Job")
def create_job(payload: JobCreateRequest, db: Session = Depends(get_session)):
    """
    Submits a new geospatial ML inference job over an Area of Interest (AOI) and date range.
    """
    # Verify target model is registered
    model = model_registry.get_model(payload.model_id)
    if not model:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model_id '{payload.model_id}'. Available models: {[m['model_id'] for m in model_registry.list_models()]}"
        )

    job_id = f"job-{uuid.uuid4().hex[:12]}"
    aoi_json_str = json.dumps(payload.aoi.model_dump())

    new_job = InferenceJob(
        job_id=job_id,
        status="PENDING",
        model_id=payload.model_id,
        start_date=payload.start_date,
        end_date=payload.end_date,
        cloud_cover_max=payload.cloud_cover_max,
        aoi_json=aoi_json_str,
        progress=0
    )

    db.add(new_job)
    db.commit()
    db.refresh(new_job)

    # Enqueue job task
    job_queue_service.enqueue_job(job_id)

    return JobResponse(
        job_id=new_job.job_id,
        status=new_job.status,
        model_id=new_job.model_id,
        start_date=new_job.start_date,
        end_date=new_job.end_date,
        cloud_cover_max=new_job.cloud_cover_max,
        progress=new_job.progress,
        created_at=new_job.created_at,
        updated_at=new_job.updated_at
    )


@router.get("", response_model=List[JobResponse], summary="List Jobs")
def list_jobs(status: Optional[str] = None, db: Session = Depends(get_session)):
    """Lists submitted inference jobs, optionally filtered by status"""
    query = select(InferenceJob)
    if status:
        query = query.where(InferenceJob.status == status.upper())
    
    jobs = db.exec(query.order_by(InferenceJob.id.desc())).all()
    return [
        JobResponse(
            job_id=j.job_id,
            status=j.status,
            model_id=j.model_id,
            start_date=j.start_date,
            end_date=j.end_date,
            cloud_cover_max=j.cloud_cover_max,
            progress=j.progress,
            result_tif_url=f"/api/v1/jobs/{j.job_id}/raster" if j.result_tif_path else None,
            error_message=j.error_message,
            created_at=j.created_at,
            updated_at=j.updated_at
        ) for j in jobs
    ]


@router.get("/{job_id}", response_model=JobResponse, summary="Get Job Status")
def get_job_status(job_id: str, db: Session = Depends(get_session)):
    """Retrieves status and metadata of a specific job"""
    statement = select(InferenceJob).where(InferenceJob.job_id == job_id)
    job = db.exec(statement).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return JobResponse(
        job_id=job.job_id,
        status=job.status,
        model_id=job.model_id,
        start_date=job.start_date,
        end_date=job.end_date,
        cloud_cover_max=job.cloud_cover_max,
        progress=job.progress,
        result_tif_url=f"/api/v1/jobs/{job.job_id}/raster" if job.result_tif_path else None,
        error_message=job.error_message,
        created_at=job.created_at,
        updated_at=job.updated_at
    )


@router.get("/{job_id}/result", response_model=JobResultResponse, summary="Get Inference Results")
def get_job_result(job_id: str, db: Session = Depends(get_session)):
    """Retrieves class area statistics ($km^2$) and vectorized GeoJSON features for completed job"""
    statement = select(InferenceJob).where(InferenceJob.job_id == job_id)
    job = db.exec(statement).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.status != "COMPLETED":
        raise HTTPException(status_code=400, detail=f"Job status is '{job.status}'. Results only available for COMPLETED jobs.")

    res_stmt = select(JobResult).where(JobResult.job_id == job_id)
    result = db.exec(res_stmt).first()
    if not result:
        raise HTTPException(status_code=404, detail="Result records not found")

    summary_stats = json.loads(result.summary_stats_json)
    geojson_features = json.loads(result.geojson_data) if result.geojson_data else None

    return JobResultResponse(
        job_id=job_id,
        summary_stats=summary_stats,
        vector_geojson=geojson_features,
        download_tif_url=f"/api/v1/jobs/{job_id}/raster"
    )


@router.get("/{job_id}/raster", summary="Download Prediction GeoTIFF")
def download_raster(job_id: str, db: Session = Depends(get_session)):
    """Downloads the generated GeoTIFF raster file for a completed job"""
    statement = select(InferenceJob).where(InferenceJob.job_id == job_id)
    job = db.exec(statement).first()
    if not job or not job.result_tif_path:
        raise HTTPException(status_code=404, detail="Raster file not found for this job")

    # Local filepath resolution
    path = job.result_tif_path
    if path.startswith("minio://"):
        parts = path.replace("minio://", "").split("/", 1)
        path = os.path.abspath(os.path.join("./storage_data", parts[0], parts[1]))

    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"File path '{path}' does not exist on server")

    return FileResponse(path=path, media_type="image/tiff", filename=f"{job_id}_prediction.tif")
