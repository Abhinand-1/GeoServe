import logging
import os
from typing import Dict, List, Tuple, Any
import numpy as np
from app.config import settings

logger = logging.getLogger("geoserve.gee_fetcher")


class GEEFetcher:
    def __init__(self, project_id: str = None):
        self.project_id = project_id or settings.GEE_PROJECT_ID
        self.initialized = False
        self._init_gee()

    def _init_gee(self):
        try:
            import ee
            ee.Initialize(project=self.project_id)
            self.initialized = True
            logger.info("Google Earth Engine initialized successfully with project: %s", self.project_id)
        except Exception as e:
            logger.warning("Google Earth Engine initialization warning (%s). Fallback mode ready.", e)
            self.initialized = False

    def fetch_imagery(
        self,
        aoi: Dict[str, Any],
        start_date: str,
        end_date: str,
        bands: List[str],
        cloud_cover_max: float = 20.0,
        scale_meters: float = 10.0
    ) -> Tuple[np.ndarray, Dict[str, Any], str]:
        """
        Retrieves Sentinel-2 imagery for requested AOI, dates, and bands.

        Returns:
            - data_array: np.ndarray of shape (Channels, Height, Width) normalized [0, 1]
            - profile: rasterio profile dict (transform, crs, height, width)
            - crs_epsg: CRS EPSG string (e.g. 'EPSG:4326')
        """
        if self.initialized:
            try:
                import ee
                ee_geometry = ee.Geometry(aoi)

                # Filter Sentinel-2 L2A collection
                collection = (
                    ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                    .filterBounds(ee_geometry)
                    .filterDate(start_date, end_date)
                    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_cover_max))
                )

                size = collection.size().getInfo()
                logger.info("Found %d Sentinel-2 scenes for requested AOI & date range", size)

                if size > 0:
                    # Cloud masking function using SCL (Scene Classification Layer)
                    def mask_clouds(img):
                        scl = img.select("SCL")
                        # 4=vegetation, 5=bare soil, 6=water, 7=unclassified
                        valid_mask = scl.gte(4).And(scl.lte(7))
                        return img.updateMask(valid_mask)

                    composite = collection.map(mask_clouds).median().select(bands).clip(ee_geometry)

                    # Compute bounding box coordinates
                    coords = aoi.get("coordinates", [[[]]])[0]
                    lons = [pt[0] for pt in coords]
                    lats = [pt[1] for pt in coords]
                    min_lon, max_lon = min(lons), max(lons)
                    min_lat, max_lat = min(lats), max(lats)

                    width_px = max(16, int((max_lon - min_lon) * 111320 / scale_meters))
                    height_px = max(16, int((max_lat - min_lat) * 111320 / scale_meters))
                    width_px = min(512, width_px)
                    height_px = min(512, height_px)

                    # Sample rectangle pixel array directly from GEE
                    rect_dict = composite.sampleRectangle(
                        region=ee_geometry,
                        defaultValue=0
                    ).getInfo()

                    properties = rect_dict.get("properties", {})
                    band_arrays = []
                    for b in bands:
                        if b in properties:
                            arr = np.array(properties[b], dtype=np.float32)
                            # Normalize Sentinel-2 surface reflectance (0..10000 -> 0..1)
                            arr = arr / 10000.0
                            arr = np.clip(arr, 0.0, 1.0)
                            band_arrays.append(arr)
                        else:
                            # Default band array if missing
                            band_arrays.append(np.ones((height_px, width_px), dtype=np.float32) * 0.1)

                    if len(band_arrays) == len(bands):
                        data = np.stack(band_arrays, axis=0) # (C, H, W)
                        h, w = data.shape[1], data.shape[2]

                        from rasterio.transform import from_bounds
                        transform = from_bounds(min_lon, min_lat, max_lon, max_lat, w, h)

                        profile = {
                            "driver": "GTiff",
                            "dtype": "float32",
                            "nodata": None,
                            "width": w,
                            "height": h,
                            "count": len(bands),
                            "crs": "EPSG:4326",
                            "transform": transform
                        }
                        logger.info("Successfully fetched real Sentinel-2 GEE pixels: shape %s", data.shape)
                        return data, profile, "EPSG:4326"
            except Exception as e:
                logger.error("GEE pixel sample error (%s). Generating fallback synthetic raster.", e)

        # Fallback synthetic satellite raster generation (for offline/test environments)
        return self._generate_synthetic_raster(aoi, bands)

    def _generate_synthetic_raster(
        self,
        aoi: Dict[str, Any],
        bands: List[str]
    ) -> Tuple[np.ndarray, Dict[str, Any], str]:
        """Generates realistic synthetic multi-spectral Sentinel-2 raster based on AOI geometry"""
        coords = aoi.get("coordinates", [[[76.25, 9.95], [76.35, 9.95], [76.35, 10.05], [76.25, 10.05], [76.25, 9.95]]])[0]
        lons = [pt[0] for pt in coords]
        lats = [pt[1] for pt in coords]
        min_lon, max_lon = min(lons), max(lons)
        min_lat, max_lat = min(lats), max(lats)

        h, w = 128, 128
        from rasterio.transform import from_bounds
        transform = from_bounds(min_lon, min_lat, max_lon, max_lat, w, h)

        # Generate synthetic spectral signatures (Water, Vegetation, Built-up, Barren)
        num_bands = len(bands)
        data = np.zeros((num_bands, h, w), dtype=np.float32)

        # Spatial grid
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h / 2, w / 2

        # Circular water body pattern in middle
        water_mask = ((x - center_x) ** 2 + (y - center_y) ** 2) < (h / 4) ** 2

        for i, band in enumerate(bands):
            if band == "B2":  # Blue
                data[i] = np.where(water_mask, 0.25, 0.08)
            elif band == "B3":  # Green
                data[i] = np.where(water_mask, 0.35, 0.15)
            elif band == "B4":  # Red
                data[i] = np.where(water_mask, 0.10, 0.12)
            elif band == "B8":  # NIR
                data[i] = np.where(water_mask, 0.05, 0.55)  # High NIR for vegetation
            elif band in ["B11", "B12"]:  # SWIR
                data[i] = np.where(water_mask, 0.02, 0.30)
            else:
                data[i] = np.random.uniform(0.1, 0.3, size=(h, w))

        # Add light noise for realism
        data += np.random.normal(0, 0.02, size=data.shape).astype(np.float32)
        data = np.clip(data, 0.0, 1.0)

        profile = {
            "driver": "GTiff",
            "dtype": "float32",
            "nodata": None,
            "width": w,
            "height": h,
            "count": num_bands,
            "crs": "EPSG:4326",
            "transform": transform
        }

        logger.info("Generated synthetic Sentinel-2 raster for AOI with shape %s", data.shape)
        return data, profile, "EPSG:4326"


gee_fetcher = GEEFetcher()
