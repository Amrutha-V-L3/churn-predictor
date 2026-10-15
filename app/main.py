"""FastAPI service for customer churn prediction."""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.model import get_model
from app.schemas import BatchRequest, BatchResponse, Customer, Prediction

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("churn-api")
STATIC = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(_: FastAPI):
    model = get_model()  # load once at startup, fail fast if missing
    log.info("Loaded %s (threshold=%.2f)", model.metadata["model_name"], model.threshold)
    yield


app = FastAPI(
    title="Customer Churn Predictor",
    description="Predicts the probability that a telecom customer will churn, "
    "and explains which factors drive the risk.",
    version="1.0.0",
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.middleware("http")
async def timing(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    ms = (time.perf_counter() - start) * 1000
    response.headers["X-Process-Time-ms"] = f"{ms:.1f}"
    if request.url.path.startswith("/predict"):
        log.info("%s %s -> %s in %.1fms", request.method, request.url.path, response.status_code, ms)
    return response


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(STATIC / "index.html")


@app.get("/health", tags=["ops"])
def health():
    return {"status": "ok"}


@app.get("/model-info", tags=["ops"])
def model_info():
    m = get_model().metadata
    keys = ("model_name", "trained_at", "n_rows", "churn_rate", "threshold", "cv_roc_auc", "test_metrics")
    return {k: m[k] for k in keys}


@app.post("/predict", response_model=Prediction, tags=["prediction"])
def predict(customer: Customer):
    return get_model().predict([customer.model_dump()])[0]


@app.post("/predict/batch", response_model=BatchResponse, tags=["prediction"])
def predict_batch(req: BatchRequest):
    return {"predictions": get_model().predict([c.model_dump() for c in req.customers])}
