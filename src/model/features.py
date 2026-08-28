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


def apply_correction(data: pd.DataFrame, lookup: dict) -> tuple[pd.Series, int, int]:
    """apply a lookup, falling back stop -> route -> global"""
    idx_stop = pd.MultiIndex.from_arrays([data["route_id"], data["stop_id"], data["lead_bucket"]])
    idx_route = pd.MultiIndex.from_arrays([data["route_id"], data["lead_bucket"]])

    bias = lookup["by_stop"].reindex(idx_stop).to_numpy()
    route_bias = lookup["by_route"].reindex(idx_route).to_numpy()
    bucket_bias = lookup["by_bucket"].reindex(data["lead_bucket"]).to_numpy()

    missing_stop = pd.isna(bias)
    bias = pd.Series(bias).where(~missing_stop, route_bias).to_numpy()
    missing_route = pd.isna(bias)
    bias = pd.Series(bias).where(~missing_route, bucket_bias).to_numpy()
    bias = pd.Series(bias).fillna(0.0).to_numpy()

    return data["error_sec"] - bias, int(missing_stop.sum()), int(missing_route.sum())
