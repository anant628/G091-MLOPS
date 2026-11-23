import json
import logging
import sys
import time
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "model.joblib"


# ---------- JSON logging: one JSON line per prediction, to stdout ----------
class JsonFormatter(logging.Formatter):
    def format(self, record):
        entry = {
            "time": self.formatTime(record),
            "level": record.levelname,
            "message": record.getMessage(),
        }
        entry.update(getattr(record, "fields", {}))
        return json.dumps(entry)


logger = logging.getLogger("mp193")
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)


# ---------- request schema with the client's two rules ----------
class PredictRequest(BaseModel):
    patients_ahead: int
    doctors_on_duty: int
    arrival_hour: int
    avg_consult_min: float
    lab_tests_pending: int = Field(..., ge=0)    # rule 2: below 0 is a typing mistake
    distance_km: float = Field(..., le=70)       # rule 1: above 70 is impossible
    department: str
    wing: str


# ---------- model loading ----------
_model = None


def get_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


app = FastAPI(title="MP193 OPD Waiting Time")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    start = time.perf_counter()
    X = pd.DataFrame([req.model_dump()])
    prediction = float(get_model().predict(X)[0])
    logger.info("prediction", extra={"fields": {
        "event": "predict",
        "prediction": round(prediction, 2),
        "department": req.department,
        "wing": req.wing,
        "latency_ms": round((time.perf_counter() - start) * 1000, 2),
    }})
    return {"prediction": prediction}