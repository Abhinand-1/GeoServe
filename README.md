# 🛰️ GeoServe — Self-Hosted Geospatial ML Inference Platform

**GeoServe** is a production-oriented backend and machine learning inference platform for geospatial workloads. It retrieves satellite imagery on demand from **Google Earth Engine (GEE)**, preprocesses multispectral data, and performs geospatial ML inference using **PyTorch** without requiring a large locally maintained satellite imagery dataset.

The system is designed around an asynchronous job-processing architecture, separating API requests from computationally intensive geospatial and ML workloads.

---

## 🏗️ System Architecture

```text
User / Client
(REST API / Dashboard)
       │
       ▼
 FastAPI Backend
 (REST API + Pydantic)
       │
       ├─────────────────────────┐
       ▼                         ▼
 PostgreSQL / PostGIS       Redis + RQ
 (Job State & Metadata)     (Task Queue)
                                 │
                                 ▼
                         ML Worker Node
                                 │
              ┌──────────────────┼──────────────────┐
              ▼                  ▼                  ▼
    Google Earth Engine   PyTorch ML Engine    MinIO / Storage
     (Sentinel-2 Data)   (Geospatial Models)   (GeoTIFF Outputs)
```

### Core Processing Flow

```text
AOI + Date Range + Model
            │
            ▼
       FastAPI API
            │
            ▼
       Create Job
        (PENDING)
            │
            ▼
      Redis / RQ Queue
            │
            ▼
       ML Worker
            │
            ├── Retrieve Sentinel-2 imagery
            ├── Cloud filtering / preprocessing
            ├── Multispectral tensor preparation
            ├── PyTorch inference
            ├── Raster post-processing
            ├── GeoTIFF generation
            └── GeoJSON + statistics
            │
            ▼
     Store Results
            │
            ▼
     COMPLETED / FAILED
```

---

## ✨ Features

### 🛰️ On-Demand Satellite Data

* Dynamically queries the **Sentinel-2 Surface Reflectance Harmonized** collection through Google Earth Engine.
* Supports AOI, date-range, and cloud-cover constraints.
* Retrieves only the imagery required for an inference job rather than maintaining a large local satellite archive.
* Performs multispectral preprocessing before model inference.

### ⚙️ Asynchronous ML Inference

* FastAPI handles lightweight API operations and job management.
* Redis + RQ decouples long-running geospatial and ML workloads from the API process.
* Workers independently execute imagery retrieval, preprocessing, inference, and post-processing.
* Job states are tracked through a persistent database.

### 🧠 Extensible PyTorch Model Registry

GeoServe uses a model registry architecture so additional geospatial ML models can be integrated without redesigning the inference pipeline.

Current models include:

* **`landcover_unet`**

  * U-Net based semantic segmentation architecture.
  * Four target classes:

    * Water
    * Vegetation
    * Built-up
    * Barren
  * Uses Sentinel-2 multispectral bands.

* **`water_detector`**

  * PyTorch-based water detection model.
  * Uses multispectral bands together with NDWI/MNDWI-derived features.
  * Produces binary water/non-water predictions.

### 🌍 Geospatial Processing

Inference outputs are converted into standard geospatial formats:

* **GeoTIFF** raster prediction outputs
* **GeoJSON** vectorized prediction polygons
* Per-class surface-area statistics
* Spatial reference and raster geotransform preservation

### 💾 Storage Architecture

GeoServe separates job metadata from larger ML output files:

* PostgreSQL/PostGIS-ready persistence for job and geospatial metadata
* MinIO-compatible object storage for generated raster artifacts
* Local filesystem fallback for development environments
* SQLite fallback for lightweight local development

### 🗺️ Interactive Dashboard

A built-in **Leaflet** dashboard provides a simple interface for:

* Selecting inference models
* Submitting AOIs and date ranges
* Monitoring job progress
* Viewing inference results
* Accessing generated GeoTIFF and GeoJSON outputs

### 🐳 Production-Oriented DevOps

The project includes:

* Docker-based API and worker containers
* Docker Compose orchestration
* Redis
* PostgreSQL/PostGIS
* MinIO object storage
* pytest test suite
* GitHub Actions CI

---

## 🔄 Job Lifecycle

Each inference request is processed as an asynchronous job:

```text
PENDING
   │
   ▼
PROCESSING
   │
   ├──────────────► FAILED
   │
   ▼
COMPLETED
```

The worker reports progress throughout the pipeline, allowing clients to monitor long-running inference jobs through the REST API.

---

## 📡 REST API Reference

| Method | Endpoint                       | Description                                                          |
| ------ | ------------------------------ | -------------------------------------------------------------------- |
| `GET`  | `/health`                      | System health and configuration status                               |
| `GET`  | `/api/v1/models`               | List registered ML models, versions, input bands, and target classes |
| `POST` | `/api/v1/jobs`                 | Submit a satellite ML inference job                                  |
| `GET`  | `/api/v1/jobs`                 | List submitted jobs with optional status filtering                   |
| `GET`  | `/api/v1/jobs/{job_id}`        | Retrieve job status and progress                                     |
| `GET`  | `/api/v1/jobs/{job_id}/result` | Retrieve inference statistics and GeoJSON                            |
| `GET`  | `/api/v1/jobs/{job_id}/raster` | Download the generated GeoTIFF                                       |

### Example Job Request

```json
{
  "model_id": "landcover_unet",
  "start_date": "2025-01-01",
  "end_date": "2025-01-31",
  "cloud_cover_max": 20,
  "aoi": {
    "type": "Polygon",
    "coordinates": [
      [
        [76.20, 9.90],
        [76.30, 9.90],
        [76.30, 10.00],
        [76.20, 10.00],
        [76.20, 9.90]
      ]
    ]
  }
}
```

---

## 🧪 Testing

The project includes automated tests covering:

* FastAPI endpoint behavior
* Pydantic request validation
* Job creation and status handling
* PyTorch model forward passes
* Model output shape and class validation
* Preprocessor tensor conversion
* NaN/Inf handling
* GEE imagery retrieval and fallback behavior
* End-to-end inference pipeline execution

Run the test suite with:

```bash
pytest -v
```

---

## 🐳 Running with Docker

```bash
docker compose up --build
```

The main services include:

```text
FastAPI API
Redis
RQ Worker
PostgreSQL/PostGIS
MinIO
```

API documentation is available through FastAPI's automatically generated Swagger interface.

---

## 📁 Project Structure

```text
GeoServe/
│
├── app/
│   ├── api/
│   ├── db/
│   ├── schemas/
│   ├── services/
│   ├── config.py
│   └── main.py
│
├── worker/
│   ├── ml/
│   ├── gee_fetcher.py
│   ├── preprocessor.py
│   ├── postprocessor.py
│   └── worker.py
│
├── tests/
│
├── .github/
│   └── workflows/
│
├── Dockerfile.api
├── Dockerfile.worker
├── docker-compose.yml
├── requirements.txt
├── pyproject.toml
├── .env.example
├── README.md
└── run_demo.py
```

---

## 🎯 Engineering Objectives

GeoServe demonstrates practical implementation of:

* Geospatial data ingestion
* Remote-sensing image processing
* PyTorch inference pipelines
* Semantic segmentation
* Spectral feature engineering
* Asynchronous job processing
* REST API design
* Model registry architecture
* Object storage
* Containerized deployment
* Automated testing
* CI/CD workflows

The architecture is intentionally designed so additional ML models, data sources, workers, and storage backends can be integrated without redesigning the core inference API.

---

## 🚧 Current Limitations & Future Improvements

GeoServe is a **production-oriented engineering project**, not a claim of production readiness.

Planned improvements include:

* Authentication and authorization
* Production secrets management
* PostgreSQL/PostGIS spatial data types
* Robust worker retries and job recovery
* Model checkpoint management and versioned artifacts
* Improved raster tiling/chunked processing for large AOIs
* More accurate geodesic/equal-area area calculations
* Prometheus/Grafana observability
* API rate limiting
* Improved frontend visualization
* GPU-enabled worker deployment
* Stronger production security and deployment configuration

---

## 📄 License

MIT License
