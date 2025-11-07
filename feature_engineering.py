import pandas as pd
from typing import List, Optional, Tuple


class FeatureEngineer:
    """
    Reads a raw CSV and produces a feature-engineered DataFrame (and optional CSV).
    - Creates lagged features for all columns except the target.
    - Keeps original features + lagged features + target.
    - Optionally binarizes the target (e.g. target > 0 -> 1 else 0).
    """
    def __init__(self, target_col: str = "target", lags: Tuple[int, ...] = (1, 2), binarize_target: bool = True):
        self.target_col = target_col
        self.lags = lags
        self.binarize_target = binarize_target

    def transform_df(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.target_col not in df.columns:
            raise ValueError(f"Target column '{self.target_col}' not found in dataframe")

        # features = all columns except target
        feature_cols = [c for c in df.columns if c != self.target_col]

        out = df.copy()

        # create lagged columns for each feature column
        for col in feature_cols:
            for lag in self.lags:
                out[f"{col}_prev{lag}"] = out[col].shift(lag)

        # Optionally binarize target (common for classification tasks)
        if self.binarize_target:
            out[self.target_col] = (out[self.target_col] > 0).astype(int)

        # Keep original features, lagged features and the target
        keep_cols = feature_cols + [f"{c}_prev{lag}" for c in feature_cols for lag in self.lags] + [self.target_col]
        # some columns may be duplicated or missing if input was odd — filter them
        keep_cols = [c for c in keep_cols if c in out.columns]

        engineered = out[keep_cols].dropna().reset_index(drop=True)
        return engineered

    def process_file(self, raw_csv_path: str, out_csv_path: Optional[str] = None) -> pd.DataFrame:
        """Read raw CSV, apply transform_df, optionally save to out_csv_path, return engineered DF."""
        df = pd.read_csv(raw_csv_path, index_col=0)
        engineered = self.transform_df(df)
        if out_csv_path:
            engineered.to_csv(out_csv_path)
        return engineered


def df_to_tensors(df: pd.DataFrame, target_col: str = "target"):
    """Convert engineered DataFrame to X (features) and y (target) torch tensors."""
    import torch
    features = [c for c in df.columns if c != target_col]
    X = torch.tensor(df[features].values.astype("float32"))
    y = torch.tensor(df[target_col].values.astype("float32")).unsqueeze(1)
    return X, y, features