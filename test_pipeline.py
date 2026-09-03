import time
from worker.worker import process_inference_job


def test_full_inference_pipeline(client):
    payload = {
        "model_id": "water_detector",
        "start_date": "2024-01-01",
        "end_date": "2024-04-01",
        "cloud_cover_max": 20.0,
        "aoi": {
            "type": "Polygon",
            "coordinates": [[
                [76.25, 9.90],
                [76.35, 9.90],
                [76.35, 10.00],
                [76.25, 10.00],
                [76.25, 9.90]
            ]]
        }
    }
    # 1. Create Job via API
    res = client.post("/api/v1/jobs", json=payload)
    assert res.status_code == 202
    job_id = res.json()["job_id"]

    # 2. Process Job via worker function
    process_inference_job(job_id)

    # 3. Check Job Status via API
    status_res = client.get(f"/api/v1/jobs/{job_id}")
    assert status_res.status_code == 200
    status_data = status_res.json()
    assert status_data["status"] == "COMPLETED"
    assert status_data["progress"] == 100

    # 4. Check Job Results via API
    result_res = client.get(f"/api/v1/jobs/{job_id}/result")
    assert result_res.status_code == 200
    result_data = result_res.json()
    assert "summary_stats" in result_data
    assert "Non-Water" in result_data["summary_stats"]
    assert "Water Body" in result_data["summary_stats"]
    assert result_data["vector_geojson"]["type"] == "FeatureCollection"
