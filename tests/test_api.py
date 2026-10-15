from fastapi.testclient import TestClient

from app.main import app

HIGH_RISK = {
    "gender": "Female", "SeniorCitizen": 1, "Partner": "No", "Dependents": "No",
    "tenure": 2, "PhoneService": "Yes", "MultipleLines": "Yes", "InternetService": "Fiber optic",
    "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
    "StreamingTV": "Yes", "StreamingMovies": "Yes", "Contract": "Month-to-month",
    "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
    "MonthlyCharges": 99.65, "TotalCharges": 199.3,
}
LOW_RISK = {
    **HIGH_RISK, "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "Yes", "tenure": 60,
    "InternetService": "DSL", "OnlineSecurity": "Yes", "OnlineBackup": "Yes",
    "DeviceProtection": "Yes", "TechSupport": "Yes", "StreamingTV": "No", "StreamingMovies": "No",
    "Contract": "Two year", "PaperlessBilling": "No", "PaymentMethod": "Bank transfer (automatic)",
    "MonthlyCharges": 74.1, "TotalCharges": 4446.0,
}


def client():
    return TestClient(app)


def test_health():
    with client() as c:
        assert c.get("/health").json() == {"status": "ok"}


def test_home_page_served():
    with client() as c:
        r = c.get("/")
        assert r.status_code == 200 and "Churn Predictor" in r.text


def test_model_info_has_metrics():
    with client() as c:
        info = c.get("/model-info").json()
        assert info["test_metrics"]["roc_auc"] > 0.8


def test_high_risk_customer_flagged():
    with client() as c:
        r = c.post("/predict", json=HIGH_RISK).json()
        assert r["will_churn"] is True
        assert r["risk_level"] == "High"
        assert r["top_drivers"], "high-risk prediction should come with drivers"
        assert {d["feature"] for d in r["top_drivers"]} & {"Contract", "tenure", "InternetService"}


def test_low_risk_customer_not_flagged():
    with client() as c:
        r = c.post("/predict", json=LOW_RISK).json()
        assert r["will_churn"] is False
        assert r["churn_probability"] < 0.2


def test_batch_matches_single():
    with client() as c:
        batch = c.post("/predict/batch", json={"customers": [HIGH_RISK, LOW_RISK]}).json()
        single = c.post("/predict", json=HIGH_RISK).json()
        assert len(batch["predictions"]) == 2
        assert batch["predictions"][0]["churn_probability"] == single["churn_probability"]


def test_invalid_input_rejected():
    with client() as c:
        assert c.post("/predict", json={**HIGH_RISK, "Contract": "Lifetime"}).status_code == 422
        assert c.post("/predict", json={**HIGH_RISK, "tenure": -5}).status_code == 422
        bad = {k: v for k, v in HIGH_RISK.items() if k != "tenure"}
        assert c.post("/predict", json=bad).status_code == 422
