"""Feature selection implemented as a single class.

This module provides one public symbol: FeatureSelector. The implementation
is intentionally compact and self-contained so tests can import and patch
instance methods easily.
"""

from typing import List, Tuple, Dict, Optional
import itertools
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


class _SimpleMLP(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, max(8, hidden // 2)),
            nn.ReLU(),
            nn.Linear(max(8, hidden // 2), 1),
        )

    def forward(self, x):
        return self.net(x)


class FeatureSelector:
    """Encapsulates feature selection logic.

    Main usage:
        fs = FeatureSelector(quick=True)
        selected, best_acc, history = fs.run_feature_selection(df, max_features=3)
    """

    def __init__(self, exhaustive_threshold: int = 4, cache: Optional[dict] = None, device: Optional[torch.device] = None):
        self.exhaustive_threshold = exhaustive_threshold
        self._cache = cache if cache is not None else {}
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    def list_candidate_tickers(self, df: pd.DataFrame, target_col: str = "target") -> List[str]:
        cols = [c for c in df.columns if c != target_col]
        tickers = set()
        for c in cols:
            base = c.split("_prev")[0]
            tickers.add(base)
        return sorted(tickers)

    def select_columns_for_tickers(self, df: pd.DataFrame, tickers: List[str], target_col: str = "target") -> pd.DataFrame:
        keep = []
        for t in tickers:
            for c in df.columns:
                if c == t or c.startswith(f"{t}_"):
                    keep.append(c)
        keep = [c for c in keep if c in df.columns]
        if target_col in df.columns:
            keep = keep + [target_col]
        return df.loc[:, keep].dropna()

    def evaluate_feature_set(
        self,
        df: pd.DataFrame,
        tickers: List[str],
        target_col: str = "target",
        test_size: float = 0.2,
        random_state: int = 0,
        epochs: int = 50,
        batch_size: int = 64,
        lr: float = 1e-3,
        device: Optional[torch.device] = None,
    ) -> float:
        """Train a tiny model on the selected columns and return accuracy.

        The function caches results per-instance using a tuple key built from
        the tickers and training hyperparameters.
        """
        cache_key = (tuple(tickers), target_col, test_size, random_state, epochs, batch_size, lr)
        if cache_key in self._cache:
            return self._cache[cache_key]

        sub = self.select_columns_for_tickers(df, tickers, target_col=target_col)
        if sub.empty or sub.shape[1] <= 1:
            self._cache[cache_key] = 0.0
            return 0.0

        X = sub.drop(columns=[target_col]).values.astype("float32")
        y = sub[target_col].values.astype("float32")

        stratify = y if len(np.unique(y)) > 1 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify
        )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        X_test_t = torch.tensor(X_test, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

        device = device or self.device

        model = _SimpleMLP(X_train_t.size(1), hidden=64).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        train_ds = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_ds, batch_size=min(batch_size, len(train_ds)), shuffle=True)

        model.train()
        for _ in range(epochs):
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(X_test_t.to(device))
            probs = torch.sigmoid(logits).cpu().numpy().ravel()
            preds = (probs > 0.5).astype(int)
            y_true = y_test.astype(int)
            if len(y_true) == 0:
                self._cache[cache_key] = 0.0
                return 0.0
            acc = (preds == y_true).mean()

        acc_f = float(acc)
        self._cache[cache_key] = acc_f
        return acc_f

    def sequential_forward_selection(
        self,
        df: pd.DataFrame,
        candidates: List[str],
        max_features: int = 5,
        patience: int = 1,
        target_col: str = "target",
        **eval_kwargs,
    ) -> Tuple[List[str], float, List[Dict]]:
        selected: List[str] = []
        remaining = list(candidates)
        best_acc = 0.0
        history: List[Dict] = []

        no_improve = 0
        while remaining and len(selected) < max_features and no_improve <= patience:
            best_candidate = None
            best_candidate_acc = best_acc
            for cand in list(remaining):
                trial = selected + [cand]
                acc = self.evaluate_feature_set(df, trial, target_col=target_col, **eval_kwargs)
                history.append({"trial": list(trial), "acc": acc})
                if acc > best_candidate_acc:
                    best_candidate_acc = acc
                    best_candidate = cand

            if best_candidate is None:
                no_improve += 1
                break
            selected.append(best_candidate)
            remaining.remove(best_candidate)
            if best_candidate_acc > best_acc:
                best_acc = best_candidate_acc
                no_improve = 0
            else:
                no_improve += 1

        return selected, best_acc, history

    def run_feature_selection(
        self,
        engineered_csv_or_df: object,
        max_features: int = 5,
        exhaustive_threshold: Optional[int] = None,
        target_col: str = "target",
        **eval_kwargs,
    ) -> Tuple[List[str], float, List[Dict]]:
        # accept either a DataFrame or a path
        if isinstance(engineered_csv_or_df, pd.DataFrame):
            df = engineered_csv_or_df
        else:
            df = pd.read_csv(engineered_csv_or_df, index_col=0)

        if df.empty:
            return [], 0.0, []

        candidates = self.list_candidate_tickers(df, target_col=target_col)
        if not candidates:
            return [], 0.0, []

        if exhaustive_threshold is None:
            exhaustive_threshold = self.exhaustive_threshold


        selected, best_acc, history = self.sequential_forward_selection(
            df, candidates, max_features=max_features, target_col=target_col, **eval_kwargs
        )

        if len(candidates) <= exhaustive_threshold:
            for r in range(1, min(max_features, len(candidates)) + 1):
                for comb in itertools.combinations(candidates, r):
                    acc = self.evaluate_feature_set(df, list(comb), target_col=target_col, **eval_kwargs)
                    history.append({"trial": list(comb), "acc": acc})
                    if acc > best_acc:
                        best_acc = acc
                        selected = list(comb)

        return selected, best_acc, history
