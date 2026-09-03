import numpy as np
import torch
from worker.preprocessor import preprocessor


def test_preprocessor_tensor_shape_and_type():
    # 4 channels (B2, B3, B4, B8), Height 64, Width 64
    raw_data = np.random.uniform(0, 10000, size=(4, 64, 64)).astype(np.float32)
    tensor = preprocessor.process_raster(raw_data)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 4, 64, 64)
    assert tensor.dtype == torch.float32


def test_preprocessor_nan_handling():
    raw_data = np.array([[[np.nan, 1.0], [2.0, np.inf]]], dtype=np.float32)
    tensor = preprocessor.process_raster(raw_data)

    assert not torch.isnan(tensor).any()
    assert not torch.isinf(tensor).any()
