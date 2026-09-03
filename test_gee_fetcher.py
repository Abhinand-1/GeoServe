from worker.gee_fetcher import gee_fetcher


def test_gee_fetcher_synthetic_fallback():
    aoi = {
        "type": "Polygon",
        "coordinates": [[
            [76.25, 9.90],
            [76.30, 9.90],
            [76.30, 9.95],
            [76.25, 9.95],
            [76.25, 9.90]
        ]]
    }
    bands = ["B2", "B3", "B4", "B8"]
    data, profile, crs = gee_fetcher.fetch_imagery(
        aoi=aoi,
        start_date="2024-01-01",
        end_date="2024-03-01",
        bands=bands
    )

    assert data.ndim == 3
    assert data.shape[0] == 4
    assert profile["driver"] == "GTiff"
    assert crs == "EPSG:4326"
