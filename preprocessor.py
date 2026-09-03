import logging
import torch
import numpy as np

logger = logging.getLogger("geoserve.preprocessor")


class Preprocessor:
    def process_raster(self, data_array: np.ndarray) -> torch.Tensor:
        """
        Preprocesses numpy raster data (Channels, Height, Width) into a PyTorch Tensor.

        Steps:
        1. Replace NaN/Inf values with 0.
        2. Normalize values to [0.0, 1.0].
        3. Convert to float32 PyTorch tensor.
        4. Add batch dimension -> (1, C, H, W).
        """
        # Clean invalid values
        data_clean = np.nan_to_num(data_array, nan=0.0, posinf=1.0, neginf=0.0)

        # Ensure float32
        data_clean = data_clean.astype(np.float32)

        # Convert to PyTorch Tensor
        tensor = torch.from_numpy(data_clean)

        # Add batch dimension
        tensor = tensor.unsqueeze(0)  # Shape: (1, C, H, W)

        logger.info("Preprocessed raster tensor shape: %s", tensor.shape)
        return tensor


preprocessor = Preprocessor()
