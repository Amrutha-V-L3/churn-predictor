import pandas as pd

from src.features import add_features, clean


def test_clean_handles_blank_total_charges_and_target():
    df = pd.DataFrame(
        {"customerID": ["a", "b"], "TotalCharges": [" ", "10.5"], "Churn": ["Yes", "No"]}
    )
    out = clean(df)
    assert "customerID" not in out.columns
    assert out["TotalCharges"].tolist() == [0.0, 10.5]
    assert out["Churn"].tolist() == [1, 0]


def test_engineered_features():
    row = {
        "tenure": 0, "TotalCharges": 0.0, "InternetService": "Fiber optic",
        "PhoneService": "Yes", "MultipleLines": "No", "OnlineSecurity": "Yes",
        "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "Yes",
        "StreamingTV": "No", "StreamingMovies": "No",
    }
    out = add_features(pd.DataFrame([row]))
    assert out.loc[0, "avg_monthly_spend"] == 0.0  # no divide-by-zero for new customers
    assert out.loc[0, "num_services"] == 4  # phone + security + support + internet
