import json
import logging
import os
import time
from datetime import datetime
from sqlmodel import Session, select
from app.db.database import engine
from app.db.models import InferenceJob, JobResult
from app.services.storage import storage_service
from worker.ml.registry import model_registry
from worker.gee_fetcher import gee_fetcher
from worker.preprocessor import preprocessor
from worker.postprocessor import postprocessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("geoserve.worker")


def process_inference_job(job_id: str):
    """
    Main job processing pipeline executed by background Redis worker or fallback thread.
    """
    logger.info("Started processing inference job: %s", job_id)
    start_time = time.time()

    with Session(engine) as session:
        statement = select(InferenceJob).where(InferenceJob.job_id == job_id)
        job = session.exec(statement).first()

        if not job:
            logger.error("Job ID %s not found in database", job_id)
            return

        try:
            # Step 1: Update status to PROCESSING
            job.status = "PROCESSING"
            job.progress = 10
            job.updated_at = datetime.utcnow().isoformat()
            session.add(job)
            session.commit()

            # Step 2: Retrieve target model from registry
            model = model_registry.get_model(job.model_id)
            if not model:
                raise ValueError(f"Model '{job.model_id}' is not registered in ModelRegistry")

            # Step 3: Parse AOI GeoJSON geometry
            aoi_data = json.loads(job.aoi_json)

            # Step 4: Fetch satellite imagery from GEE
            logger.info("[%s] Fetching Sentinel-2 imagery from GEE...", job_id)
            data_array, profile, crs = gee_fetcher.fetch_imagery(
                aoi=aoi_data,
                start_date=job.start_date,
                end_date=job.end_date,
                bands=model.input_bands,
                cloud_cover_max=job.cloud_cover_max,
                scale_meters=model.resolution_meters
            )
            job.progress = 40
            session.add(job)
            session.commit()

            # Step 5: Preprocess raster tensor
            logger.info("[%s] Preprocessing spectral bands...", job_id)
            tensor = preprocessor.process_raster(data_array)
            job.progress = 60
            session.add(job)
            session.commit()

            # Step 6: Perform PyTorch Model Inference
            logger.info("[%s] Running PyTorch ML model inference (%s)...", job_id, model.model_id)
            predictions_tensor = model.predict(tensor)
            job.progress = 80
            session.add(job)
            session.commit()

            # Step 7: Postprocess predictions into spatial GeoTIFF and GeoJSON
            logger.info("[%s] Postprocessing predictions & vectorizing polygons...", job_id)
            local_tif_name = f"pred_{job_id}.tif"
            local_tif_path = os.path.join("./storage_data/predictions", local_tif_name)

            output_path, summary_stats, vector_geojson = postprocessor.process_predictions(
                predictions_tensor=predictions_tensor,
                profile=profile,
                class_labels=model.classes,
                output_filepath=local_tif_path
            )

            # Step 8: Save to storage service
            storage_uri = storage_service.save_file("predictions", local_tif_name, output_path)

            # Step 9: Save result to database
            result_record = JobResult(
                job_id=job_id,
                summary_stats_json=json.dumps(summary_stats),
                geojson_data=json.dumps(vector_geojson)
            )
            session.add(result_record)

            # Step 10: Complete job
            job.status = "COMPLETED"
            job.progress = 100
            job.result_tif_path = storage_uri
            job.updated_at = datetime.utcnow().isoformat()
            session.add(job)
            session.commit()

            elapsed = round(time.time() - start_time, 2)
            logger.info("Successfully completed inference job %s in %s seconds!", job_id, elapsed)

        except Exception as e:
            logger.exception("Error processing inference job %s: %s", job_id, e)
            job.status = "FAILED"
            job.error_message = str(e)
            job.updated_at = datetime.utcnow().isoformat()
            session.add(job)
            session.commit()


if __name__ == "__main__":
    import sys
    logger.info("GeoServe Redis RQ Worker starting up...")
    try:
        from rq import Worker, Queue, Connection
        from app.services.redis_queue import job_queue_service
        if job_queue_service.redis_conn:
            with Connection(job_queue_service.redis_conn):
                worker = Worker(["geoserve_jobs"])
                worker.work()
        else:
            logger.error("Redis connection unavailable. Cannot start RQ worker process.")
    except Exception as exc:
        logger.error("Worker launch error: %s", exc)
