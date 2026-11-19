import pandas as pd

TRAIN = "data/raw/train.csv"
HOLDOUT = "data/raw/holdout_features.csv"
LATER = "data/raw/later.csv"

NUM_COLS = ["patients_ahead", "doctors_on_duty", "arrival_hour",
            "avg_consult_min", "lab_tests_pending", "distance_km"]


def load(path=TRAIN):
    return pd.read_csv(path)


def check_leaking_column(df, holdout):
    print("\n== 1. LEAKING COLUMN ==")
    print("In train but not in holdout:", set(df.columns) - set(holdout.columns))
    print("Correlation with wait_min:")
    print(df[NUM_COLS].corrwith(df["wait_min"]))


def check_unit_mixup(df):
    print("\n== 2. UNIT MIXUP (per wing) ==")
    print(df.groupby("wing")[NUM_COLS].agg(["min", "mean", "max"]).T)


def check_sentinel(df):
    print("\n== 3. SENTINEL VALUES ==")
    for c in NUM_COLS:
        print(f"\n{c}:")
        print(df[c].value_counts().head(3))


def check_duplicates(df):
    print("\n== 4. DUPLICATES ==")
    print("Extra copies:", df.duplicated().sum())


def check_stuck_sensor(df):
    print("\n== 5. STUCK SENSOR ==")
    d = df.sort_values("date")
    for c in NUM_COLS:
        print(f"\n{c}: top repeated values")
        print(d[c].value_counts().head(2))


def find_stuck_runs(df):
    print("\n== 5b. LONGEST RUNS OF IDENTICAL VALUES ==")
    d = df.drop_duplicates().sort_values(["date", "record_id"]).reset_index(drop=True)
    for c in NUM_COLS:
        run_id = (d[c] != d[c].shift()).cumsum()
        runs = d.groupby(run_id).agg(
            value=(c, "first"), rows=(c, "size"),
            start=("date", "min"), end=("date", "max"))
        print(f"\n{c}:")
        print(runs.sort_values("rows", ascending=False).head(2))


def check_target_typos(df):
    print("\n== 6. TARGET TYPOS ==")
    print(df["wait_min"].sort_values().tail(25).values)


def check_spelling(df):
    print("\n== 7. SPELLING VARIANTS ==")
    for c in ["department", "wing"]:
        print(f"\n{c}:")
        print(df[c].value_counts())


def exact_counts(df):
    print("\n== EXACT COUNTS ==")
    print("Sentinel -1 rows:", (df["lab_tests_pending"] == -1).sum())
    print("Rows with wait_min > 400:", (df["wait_min"] > 400).sum())


if __name__ == "__main__":
    df = load()
    holdout = load(HOLDOUT)
    check_leaking_column(df, holdout)
    check_unit_mixup(df)
    check_sentinel(df)
    check_duplicates(df)
    check_stuck_sensor(df)
    find_stuck_runs(df)
    check_target_typos(df)
    check_spelling(df)
    exact_counts(df)