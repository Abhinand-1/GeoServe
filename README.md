# 🛰️ GeoServe — Self-Hosted Geospatial ML Inference Platform

**GeoServe** is a production-oriented backend and machine learning inference system designed to retrieve satellite imagery on demand from **Google Earth Engine (GEE)**, preprocess multi-spectral bands, and execute geospatial ML inference via **PyTorch** without maintaining a massive local satellite image dataset.

---

## 🏗️ System Architecture

```
User / Client (REST / Dashboard)
       │
       ▼
 FastAPI Backend (REST API, Pydantic, Auth)
       │
       ├─────────────────────────┐
       ▼                         ▼
 SQLModel / PostGIS        Redis Queue (RQ / Async Worker)
 (Job State & Geometries)        │
                                 ▼
                         ML Worker Node
                                 │
              ┌──────────────────┼──────────────────┐
              ▼                  ▼                  ▼
    Google Earth Engine   PyTorch ML Engine    MinIO / Storage
   (Sentinel-2 Fetcher)   (U-Net LandCover)    (GeoTIFF Rasters)
```

---

## ✨ Features

- **On-Demand Sentinel-2 Retrieval**: Queries `COPERNICUS/S2_SR_HARMONIZED` dynamically via Google Earth Engine API using GEE Project `ee-abhinandpsreenivasan1`.
- **Asynchronous Task Queue**: Uses Redis & Background workers to decouple API requests from heavy satellite download and model processing steps.
- **Extensible PyTorch Model Registry**:
  - `landcover_unet`: 4-class semantic land cover segmentation (Water, Vegetation, Built-up, Barren).
  - `water_detector`: Deep NDWI/MNDWI spectral water detector.
- **Geospatial Processing & Storage**:
  - Exports standard **GeoTIFF** rasters preserving affine geotransforms and spatial CRS (`EPSG:4326`).
  - Vectorizes class prediction masks to **GeoJSON** polygons.
  - Computes class surface area coverage ($km^2$).
  - Storage into **PostGIS** and **MinIO** object storage (with seamless local file & SQLite fallback).
- **Interactive Leaflet Dashboard**: Built-in visual map interface at `http://localhost:8000/`.
- **Production DevOps Setup**: Docker Compose, pytest suite, and GitHub Actions CI workflow.

---

## 🚀 Quickstart Guide

### 1. Requirements
- Python 3.10+
- (Optional) Docker & Docker Compose
- Google Earth Engine authenticated credentials or ADC setup.

### 2. Environment Setup

```bash
# Clone or navigate to directory
cd geoserve

# Create virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Running the Server & Dashboard

```bash
# Start FastAPI application
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
Open **`http://localhost:8000`** in your browser to view the interactive UI, or **`http://localhost:8000/docs`** for interactive OpenAPI documentation.

### 4. Running via Docker Compose

```bash
docker-compose up --build
```
This launches:
- `api` (FastAPI at `http://localhost:8000`)
- `worker` (Python Redis RQ worker)
- `redis` (Redis queue at `:6379`)
- `postgres` (PostGIS database at `:5432`)
- `minio` (MinIO Object Storage at `:9000`, Console at `:9001`)

---

## 📡 REST API Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | System health status & GEE project ID check |
| `GET` | `/api/v1/models` | List registered ML models, versions, input bands, and target classes |
| `POST` | `/api/v1/jobs` | Submit satellite ML inference job (AOI, date range, model) |
| `GET` | `/api/v1/jobs` | List submitted jobs with status filtering |
| `GET` | `/api/v1/jobs/{job_id}` | Poll job status, execution timeline, and progress |
| `GET` | `/api/v1/jobs/{job_id}/result` | Retrieve class surface area metrics ($km^2$) and vectorized GeoJSON |
| `GET` | `/api/v1/jobs/{job_id}/raster` | Download output spatial GeoTIFF raster file |

---

## 🧪 Testing

Execute the automated test suite with `pytest`:

```bash
pytest -v
```

Tests cover:
- FastAPI endpoint responses & Pydantic validation
- PyTorch model forward passes & shape consistency
- Preprocessor tensor conversions & NaN handling
- GEE fetching fallback mechanisms
- End-to-end inference pipeline execution

---

## 📄 License

MIT License. Developed for production geospatial ML inference systems.
