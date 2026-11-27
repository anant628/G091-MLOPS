# G091 - MP193: OPD Waiting Time (Kochi)

Group ID: G091
Problem ID: MP193

## Members
- S24CSEU0628 - ANANTSINGH
- S24CSEU0438 - SachinKaushik
- S24CSEU2579 - AmanSingh
- S24CSEU0698 - ARNAVJAIN

## About
MLOps pipeline to predict patient waiting time (wait_min) at Kochi District Hospital OPD.

## Run the API with Docker

Install Docker with Linux container support, then run from the repository root:

```sh
docker build -t mp193-api:1.0 .
docker run --rm --name mp193-api -p 127.0.0.1:8000:8000 mp193-api:1.0
```

Open http://127.0.0.1:8000/docs for interactive requests, or
http://127.0.0.1:8000/health for `{"status":"ok"}`. The root `/` has no route.
Stop the foreground container with Ctrl+C.

The image includes the committed `models/model.joblib` and installs the existing
pinned requirements before copying the API. It runs as a non-root user and writes
prediction logs as JSON to stdout. Raw data, virtual environments, and MLflow
databases are excluded from the build context. Training is not needed to serve
the saved model.

### Test a prediction

In `/docs`, expand `POST /predict`, select **Try it out**, and use:

```json
{
  "patients_ahead": 50,
  "doctors_on_duty": 6,
  "arrival_hour": 11,
  "avg_consult_min": 10.5,
  "lab_tests_pending": 15,
  "distance_km": 20.0,
  "department": "General",
  "wing": "Old Wing"
}
```

A valid request returns HTTP 200 with a numeric `prediction`. Changing
`distance_km` to 71 or `lab_tests_pending` to -1 returns HTTP 422. Boundary values
70 and 0 are accepted. Missing or extra fields return HTTP 422. Each successful
prediction writes a JSON log line.

In another terminal, inspect container health and logs:

```sh
docker inspect --format '{{.State.Health.Status}}' mp193-api
docker logs mp193-api
```

The Docker health check uses `/health`; use a real `/predict` request to verify
model loading as well.

## Run the existing tests without raw CSVs

```sh
python -m pip install -r requirements.txt
python -m pytest -q
```

The API tests substitute a small dummy model and the cleaning tests construct
their own data. The tests do not require `train.csv` or `later.csv`.
