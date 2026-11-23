import hashlib

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.opd.cleaning import clean_train, load_params

NUM_COLS = ["patients_ahead", "doctors_on_duty", "arrival_hour",
            "avg_consult_min", "lab_tests_pending", "distance_km"]
CAT_COLS = ["department", "wing"]
FEATURES = NUM_COLS + CAT_COLS
TARGET = "wait_min"


def sha256_of(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def build_pipeline(cfg, seed):
    kind = cfg["model"]
    if kind == "ridge":
        reg = Ridge(alpha=cfg["alpha"])
    elif kind == "random_forest":
        reg = RandomForestRegressor(n_estimators=cfg["n_estimators"],
                                    max_depth=cfg["max_depth"],
                                    random_state=seed, n_jobs=-1)
    elif kind == "gradient_boosting":
        reg = GradientBoostingRegressor(n_estimators=cfg["n_estimators"],
                                        learning_rate=cfg["learning_rate"],
                                        max_depth=cfg["max_depth"],
                                        random_state=seed)
    else:
        raise ValueError(f"unknown model: {kind}")

    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), NUM_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
    ])
    return Pipeline([("pre", pre), ("reg", reg)])


def main():
    p = load_params()
    t, m = p["training"], p["mlflow"]

    train = clean_train(pd.read_csv(p["data"]["train"]), p)
    later = pd.read_csv(p["data"]["later"])
    X, y = train[FEATURES], train[TARGET]
    X_later, y_later = later[FEATURES], later[TARGET]
    sha = sha256_of(p["data"]["train"])

    mlflow.set_tracking_uri(m["tracking_uri"])
    mlflow.set_experiment(m["experiment_name"])

    results = []
    for i, cfg in enumerate(t["runs"]):
        pipe = build_pipeline(cfg, t["seed"])
        with mlflow.start_run(run_name=f"{cfg['model']}_{i}") as run:
            mlflow.log_param("data_sha256", sha)
            mlflow.log_params(cfg)

            mae_cv = -cross_val_score(pipe, X, y, cv=t["cv_folds"],
                                      scoring="neg_mean_absolute_error").mean()
            pipe.fit(X, y)
            mae_later = mean_absolute_error(y_later, pipe.predict(X_later))

            mlflow.log_metric("mae_cv", mae_cv)
            mlflow.log_metric("mae_later", mae_later)
            mlflow.sklearn.log_model(pipe, name="model",
                                     serialization_format="cloudpickle")
            results.append((mae_cv, run.info.run_id, pipe))
            print(f"{cfg['model']:18s} mae_cv={mae_cv:.2f} "
                  f"mae_later={mae_later:.2f} run_id={run.info.run_id}")

    # best model = lowest cross-validated MAE on the training data
    mae_cv, run_id, best = min(results, key=lambda r: r[0])
    mv = mlflow.register_model(f"runs:/{run_id}/model", m["registered_model_name"])
    MlflowClient().set_registered_model_alias(
        m["registered_model_name"], m["champion_alias"], mv.version)
    joblib.dump(best, "models/model.joblib")
    print(f"\nBEST run_id={run_id} mae_cv={mae_cv:.2f} -> "
          f"registered version {mv.version} as @{m['champion_alias']}")


if __name__ == "__main__":
    main()