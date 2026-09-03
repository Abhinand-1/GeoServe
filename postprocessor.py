import logging
import os
import json
from typing import Dict, Any, Tuple, List
import numpy as np
import torch

# Clear conflicting system PROJ environment variables if present (e.g. PostgreSQL/PostGIS PROJ conflicts)
os.environ.pop("PROJ_LIB", None)
os.environ.pop("GDAL_DATA", None)

import rasterio
from rasterio.features import shapes
from shapely.geometry import shape, mapping
from shapely.ops import transform as shapely_transform
import pyproj

logger = logging.getLogger("geoserve.postprocessor")


class Postprocessor:
    def process_predictions(
        self,
        predictions_tensor: torch.Tensor,
        profile: Dict[str, Any],
        class_labels: List[str],
        output_filepath: str
    ) -> Tuple[str, Dict[str, float], Dict[str, Any]]:
        """
        Postprocesses prediction tensor into spatial GeoTIFF and vectorized GeoJSON.

        Returns:
            - output_filepath: path to saved GeoTIFF raster
            - summary_stats: Dict of class name -> surface area in sq km
            - vector_geojson: GeoJSON FeatureCollection dict
        """
        # Convert tensor to numpy uint8 array (Height, Width)
        pred_arr = predictions_tensor.squeeze(0).cpu().numpy().astype(np.uint8)
        h, w = pred_arr.shape

        # Update profile for single-channel output GeoTIFF
        out_profile = profile.copy()
        out_profile.update({
            "count": 1,
            "dtype": "uint8",
            "driver": "GTiff",
            "nodata": 255
        })

        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
        with rasterio.open(output_filepath, "w", **out_profile) as dst:
            dst.write(pred_arr, 1)

        logger.info("Saved spatial output GeoTIFF to %s", output_filepath)

        # Compute Pixel Area & Class Statistics
        transform = profile.get("transform")
        # Calculate pixel resolution in degrees (approx ~111,320m per degree at equator)
        res_x = abs(transform[0])
        res_y = abs(transform[4])
        
        # Approximate area per pixel in sq km
        pixel_area_sqkm = (res_x * 111.32) * (res_y * 111.32)

        summary_stats = {}
        for class_idx, label in enumerate(class_labels):
            pixel_count = int(np.sum(pred_arr == class_idx))
            area_sqkm = round(pixel_count * pixel_area_sqkm, 4)
            summary_stats[label] = area_sqkm

        # Vectorize prediction raster to GeoJSON features
        features = []
        for geom_dict, val in shapes(pred_arr, transform=transform):
            val_int = int(val)
            if 0 <= val_int < len(class_labels):
                poly_shape = shape(geom_dict)
                if poly_shape.is_valid and not poly_shape.is_empty:
                    label = class_labels[val_int]
                    feature = {
                        "type": "Feature",
                        "geometry": mapping(poly_shape),
                        "properties": {
                            "class_id": val_int,
                            "class_label": label
                        }
                    }
                    features.append(feature)

        vector_geojson = {
            "type": "FeatureCollection",
            "features": features
        }

        logger.info("Generated %d vectorized GeoJSON features. Class stats: %s", len(features), summary_stats)
        return output_filepath, summary_stats, vector_geojson


postprocessor = Postprocessor()
