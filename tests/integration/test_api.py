from __future__ import annotations


def test_health_returns_ok_with_model_loaded(api_client):
    r = api_client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_success(api_client):
    r = api_client.post(
        "/api/v1/predict",
        json={"store_nbr": 1, "family": "dairy", "date": "2017-08-20", "onpromotion": 2},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["family"] == "DAIRY"  # normalized to uppercase
    assert body["predicted_sales"] >= 0
    assert body["model_name"]
    assert body["model_alias"]


def test_predict_unknown_store_family_returns_404(api_client):
    r = api_client.post(
        "/api/v1/predict",
        json={"store_nbr": 424242, "family": "DAIRY", "date": "2017-08-20", "onpromotion": 0},
    )
    assert r.status_code == 404
    assert "detail" in r.json()


def test_predict_invalid_store_nbr_returns_422(api_client):
    r = api_client.post(
        "/api/v1/predict",
        json={"store_nbr": -1, "family": "DAIRY", "date": "2017-08-20", "onpromotion": 0},
    )
    assert r.status_code == 422


def test_predict_missing_required_field_returns_422(api_client):
    r = api_client.post("/api/v1/predict", json={"store_nbr": 1, "date": "2017-08-20"})
    assert r.status_code == 422


def test_predict_negative_onpromotion_returns_422(api_client):
    r = api_client.post(
        "/api/v1/predict",
        json={"store_nbr": 1, "family": "DAIRY", "date": "2017-08-20", "onpromotion": -5},
    )
    assert r.status_code == 422


def test_predict_batch_success(api_client):
    r = api_client.post(
        "/api/v1/predict/batch",
        json={
            "items": [
                {"store_nbr": 1, "family": "DAIRY", "date": "2017-08-20", "onpromotion": 0},
                {"store_nbr": 2, "family": "BEVERAGES", "date": "2017-08-21", "onpromotion": 1},
            ]
        },
    )
    assert r.status_code == 200
    predictions = r.json()["predictions"]
    assert len(predictions) == 2
    assert all(p["predicted_sales"] >= 0 for p in predictions)


def test_predict_batch_empty_list_rejected(api_client):
    r = api_client.post("/api/v1/predict/batch", json={"items": []})
    assert r.status_code == 422


def test_metrics_endpoint_exposes_prometheus_format(api_client):
    r = api_client.get("/metrics")
    assert r.status_code == 200
    assert "text/plain" in r.headers["content-type"]


def test_metrics_endpoint_includes_custom_ml_metrics(api_client):
    # trigger at least one prediction so the counters have a sample
    api_client.post(
        "/api/v1/predict",
        json={"store_nbr": 1, "family": "DAIRY", "date": "2017-08-20", "onpromotion": 0},
    )
    r = api_client.get("/metrics")
    body = r.text
    assert "demand_forecast_predictions_total" in body
    assert "demand_forecast_prediction_latency_seconds" in body
    assert "demand_forecast_model_info" in body


def test_docs_endpoint_available(api_client):
    r = api_client.get("/docs")
    assert r.status_code == 200


def test_openapi_schema_available(api_client):
    r = api_client.get("/openapi.json")
    assert r.status_code == 200
    schema = r.json()
    assert "/api/v1/predict" in schema["paths"]
