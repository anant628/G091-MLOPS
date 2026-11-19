import numpy as np
import pandas as pd
import yaml


def load_params(path="params.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def clean_train(df, params=None):
    p = (params or load_params())["cleaning"]

    # 1. remove duplicate rows
    df = df.drop_duplicates().copy()

    # 2. sentinel -1 -> missing
    col = p["sentinel_column"]
    df[col] = df[col].replace(p["sentinel_value"], np.nan)

    # 3. stuck sensor readings are untrustworthy -> missing
    in_run = (df["date"] >= p["stuck_start"]) & (df["date"] <= p["stuck_end"])
    df.loc[in_run, p["stuck_column"]] = np.nan

    # 4. target typos: 10x too big -> divide by 10
    typo = df["wait_min"] > p["typo_threshold"]
    df.loc[typo, "wait_min"] = df.loc[typo, "wait_min"] / p["typo_factor"]

    return df.reset_index(drop=True)