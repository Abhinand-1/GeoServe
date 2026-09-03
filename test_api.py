def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "gee_project_id" in data


def test_list_models(client):
    response = client.get("/api/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert len(data["models"]) >= 2
    model_ids = [m["model_id"] for m in data["models"]]
    assert "landcover_unet" in model_ids
    assert "water_detector" in model_ids


def test_create_and_get_job(client):
    payload = {
        "model_id": "landcover_unet",
        "start_date": "2024-01-01",
        "end_date": "2024-03-01",
        "cloud_cover_max": 15.0,
        "aoi": {
            "type": "Polygon",
            "coordinates": [[
                [76.25, 9.90],
                [76.30, 9.90],
                [76.30, 9.95],
                [76.25, 9.95],
                [76.25, 9.90]
            ]]
        }
    }
    # Test job creation
    post_res = client.post("/api/v1/jobs", json=payload)
    assert post_res.status_code == 202
    job_data = post_res.json()
    assert "job_id" in job_data
    job_id = job_data["job_id"]

    # Test status check
    get_res = client.get(f"/api/v1/jobs/{job_id}")
    assert get_res.status_code == 200
    status_data = get_res.json()
    assert status_data["job_id"] == job_id
    assert status_data["model_id"] == "landcover_unet"


def test_invalid_model_job_creation(client):
    payload = {
        "model_id": "non_existent_model",
        "start_date": "2024-01-01",
        "end_date": "2024-03-01",
        "aoi": {
            "type": "Polygon",
            "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]
        }
    }
    res = client.post("/api/v1/jobs", json=payload)
    assert res.status_code == 400
