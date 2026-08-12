from __future__ import annotations


def test_health_returns_ok_with_model_loaded(api_client):
    r = api_client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_success(api_client):
    r = api_client.post("/api/v1/predict", json={"store_nbr": 1, "date": "2012-11-02"})
    assert r.status_code == 200
    body = r.json()
    assert body["store_nbr"] == 1
    assert body["predicted_sales"] >= 0
    assert body["model_name"]
    assert body["model_alias"]


def test_predict_auto_detects_holiday_week(api_client):
    # 2012-12-28 is a real Christmas-week Friday in the training data
    # (holiday_flag=1); the API must accept it without an explicit override.
    r = api_client.post("/api/v1/predict", json={"store_nbr": 1, "date": "2012-12-28"})
    assert r.status_code == 200
    assert r.json()["predicted_sales"] >= 0


def test_predict_accepts_explicit_overrides(api_client):
    r = api_client.post(
        "/api/v1/predict",
        json={
            "store_nbr": 1,
            "date": "2012-11-02",
            "holiday_flag": 1,
            "temperature": 45.0,
            "fuel_price": 3.1,
            "cpi": 210.5,
            "unemployment": 7.2,
        },
    )
    assert r.status_code == 200


def test_predict_store_outside_valid_range_returns_422(api_client):
    r = api_client.post("/api/v1/predict", json={"store_nbr": 46, "date": "2012-11-02"})
    assert r.status_code == 422


def test_predict_store_not_in_training_data_returns_404(api_client):
    # store 4 is within the valid 1-45 range but the test fixture only has
    # stores 1-3, so it's a legitimate "unknown to this model" 404, not a
    # pydantic-level validation error.
    r = api_client.post("/api/v1/predict", json={"store_nbr": 4, "date": "2012-11-02"})
    assert r.status_code == 404
    assert "detail" in r.json()


def test_predict_missing_required_field_returns_422(api_client):
    r = api_client.post("/api/v1/predict", json={"store_nbr": 1})
    assert r.status_code == 422


def test_predict_invalid_holiday_flag_returns_422(api_client):
    r = api_client.post(
        "/api/v1/predict", json={"store_nbr": 1, "date": "2012-11-02", "holiday_flag": 2}
    )
    assert r.status_code == 422


def test_predict_batch_success(api_client):
    r = api_client.post(
        "/api/v1/predict/batch",
        json={
            "items": [
                {"store_nbr": 1, "date": "2012-11-02"},
                {"store_nbr": 2, "date": "2012-11-09"},
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
    api_client.post("/api/v1/predict", json={"store_nbr": 1, "date": "2012-11-02"})
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
