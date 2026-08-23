"""shared bucketing for the correction table"""
import pandas as pd

LEAD_BUCKET_WIDTH_MIN = 3
LEAD_BUCKET_MAX_MIN = 30


def add_lead_bucket(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    lead_min = df["lead_time_sec"] / 60.0
    capped = lead_min.clip(upper=LEAD_BUCKET_MAX_MIN)
    df["lead_bucket"] = (capped // LEAD_BUCKET_WIDTH_MIN * LEAD_BUCKET_WIDTH_MIN).astype(int)
    return df


def time_split(df: pd.DataFrame, train_frac: float = 0.75) -> tuple[pd.DataFrame, pd.DataFrame]:
    """split by time, not randomly. train on the earlier period"""
    df = df.sort_values("observed_at")
    cutoff = df["observed_at"].quantile(train_frac)
    train = df[df["observed_at"] <= cutoff]
    test = df[df["observed_at"] > cutoff]
    return train, test
