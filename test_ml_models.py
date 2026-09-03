import torch
from worker.ml.registry import model_registry


def test_model_registry():
    models = model_registry.list_models()
    assert len(models) >= 2
    
    unet = model_registry.get_model("landcover_unet")
    assert unet is not None
    assert unet.input_bands == ["B2", "B3", "B4", "B8"]
    assert unet.classes == ["Water", "Vegetation", "Built-up", "Barren"]


def test_landcover_unet_forward_pass():
    unet = model_registry.get_model("landcover_unet")
    dummy_input = torch.randn(1, 4, 32, 32)
    output = unet.predict(dummy_input)

    assert output.shape == (1, 32, 32)
    assert output.dtype == torch.int64
    assert output.min() >= 0 and output.max() < 4


def test_water_detector_forward_pass():
    water_net = model_registry.get_model("water_detector")
    dummy_input = torch.randn(1, 3, 32, 32)
    output = water_net.predict(dummy_input)

    assert output.shape == (1, 32, 32)
    assert output.dtype == torch.int64
    assert output.min() >= 0 and output.max() < 2
