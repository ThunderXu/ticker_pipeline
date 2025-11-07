"""Feature selection utilities for ticker -> target pipeline.

Provides a sequential forward selection routine that picks tickers
whose features (base + lagged columns) improve test accuracy when
training a small PyTorch model.

Usage:
    from feature_selection import run_feature_selection
    best, history = run_feature_selection('engineered.csv', max_features=5, target_col='target')

The module keeps things simple and CPU-friendly: default small model
and modest epochs. For production hyperparameter tuning, increase
epochs and use cross-validation.
"""
from typing import List, Tuple, Dict, Optional
import itertools
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import time
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


# simple in-memory cache for evaluated trials to avoid re-training duplicates
_EVAL_CACHE: dict = {}


class SimpleMLP(nn.Module):
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


def list_candidate_tickers(df: pd.DataFrame, target_col: str = "target") -> List[str]:
    """Return a list of base ticker names found in the engineered DataFrame.

    It scans columns, ignoring the target column, and collapses column
    names like 'AAPL_prev1' to 'AAPL'.
    """
    cols = [c for c in df.columns if c != target_col]
    tickers = set()
    for c in cols:
        base = c.split("_prev")[0]
        tickers.add(base)
    return sorted(tickers)


def _select_columns_for_tickers(df: pd.DataFrame, tickers: List[str], target_col: str = "target") -> pd.DataFrame:
    """Return a DataFrame containing only columns for the given tickers plus the target."""
    keep = []
    for t in tickers:
        for c in df.columns:
            if c == t or c.startswith(f"{t}_"):
                keep.append(c)
    keep = [c for c in keep if c in df.columns]
    if target_col in df.columns:
        keep = keep + [target_col]
    # return a copy with NA rows removed
    return df.loc[:, keep].dropna()


def evaluate_feature_set(
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
    """Train a small model on features from `tickers` and return test accuracy.

    Returns accuracy in [0,1]. If no features available, returns 0.0.
    Uses an in-memory cache keyed by the tickers + eval hyperparameters to avoid
    re-training identical trials.
    """
    # build a cache key that includes the hyperparameters that affect results
    cache_key = (tuple(tickers), target_col, test_size, random_state, epochs, batch_size, lr)
    if cache_key in _EVAL_CACHE:
        return _EVAL_CACHE[cache_key]

    sub = _select_columns_for_tickers(df, tickers, target_col=target_col)
    if sub.empty or sub.shape[1] <= 1:
        _EVAL_CACHE[cache_key] = 0.0
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

    # to tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    model = SimpleMLP(X_train_t.size(1), hidden=64).to(device)
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

    # eval
    model.eval()
    with torch.no_grad():
        logits = model(X_test_t.to(device))
        probs = torch.sigmoid(logits).cpu().numpy().ravel()
        preds = (probs > 0.5).astype(int)
        y_true = y_test.astype(int)
        if len(y_true) == 0:
            _EVAL_CACHE[cache_key] = 0.0
            return 0.0
        acc = (preds == y_true).mean()
    acc_f = float(acc)
    _EVAL_CACHE[cache_key] = acc_f
    return acc_f


def sequential_forward_selection(
    df: pd.DataFrame,
    candidates: List[str],
    max_features: int = 5,
    patience: int = 1,
    target_col: str = "target",
    **eval_kwargs,
) -> Tuple[List[str], float, List[Dict]]:
    """Greedy forward selection over candidate tickers.

    Arguments:
      df: engineered dataframe containing features and the target column
      candidates: list of ticker strings to consider
      max_features: maximum number of tickers to select
      patience: number of iterations with no improvement to stop (default 1)
      eval_kwargs: passed to evaluate_feature_set (epochs, batch_size, etc.)

    Returns: (best_set, best_acc, history)
      history: list of dicts recording each step and accuracies
    """
    selected: List[str] = []
    remaining = list(candidates)
    best_acc = 0.0
    history: List[Dict] = []
    no_improve = 0

    # quick-mode: if caller requested quick and didn't pass epochs, lower epochs
    quick = eval_kwargs.pop("quick", False)
    if quick and "epochs" not in eval_kwargs:
        eval_kwargs["epochs"] = 5

    while len(selected) < min(max_features, len(candidates)) and remaining:
        print(f"  SFS iteration: selected={selected}, remaining={remaining}")
        best_candidate = None
        best_candidate_acc = -1.0
        for c in list(remaining):
            print(f"    Evaluating candidate ticker: {c}")
            trial = selected + [c]
            t0 = time.time()
            acc = evaluate_feature_set(df, trial, target_col=target_col, **eval_kwargs)
            elapsed = time.time() - t0
            history.append({"trial": list(trial), "acc": acc, "time_s": elapsed})
            print(f"    trial {trial} -> acc={acc:.4f} (t={elapsed:.2f}s)")
            if acc > best_candidate_acc:
                best_candidate_acc = acc
                best_candidate = c
        print(f"  Best candidate this round: {best_candidate} with acc={best_candidate_acc:.4f}")

        # check improvement
        if best_candidate is None:
            break
        if best_candidate_acc > best_acc + 1e-6:
            print(f"  Selected ticker {best_candidate} improving acc to {best_candidate_acc:.4f}")
            selected.append(best_candidate)
            remaining.remove(best_candidate)
            best_acc = best_candidate_acc
            no_improve = 0
        else:
            print(f"  No improvement found this round (best acc={best_candidate_acc:.4f})")
            no_improve += 1
            if no_improve >= patience:
                print("  No improvement for several rounds, stopping SFS.")
                break
            # still remove the candidate to avoid retrying same
            remaining.remove(best_candidate)

    return selected, best_acc, history


def run_feature_selection(
    engineered_csv: str,
    max_features: int = 5,
    exhaustive_threshold: int = 4,
    target_col: str = "target",
    **eval_kwargs,
) -> Tuple[List[str], float, List[Dict]]:
    """High-level helper: load CSV, list tickers and perform selection.
    If the number of candidates is <= exhaustive_threshold, the function
    will also try an exhaustive search over all subsets up to size `max_features`.
    Returns the best found subset and accuracy plus history.
    """
    # quick-mode flag is accepted here and passed down to selection
    quick = eval_kwargs.pop("quick", False)
    if quick and "epochs" not in eval_kwargs:
        eval_kwargs["epochs"] = 5
    df = pd.read_csv(engineered_csv, index_col=0)
    if df.empty:
        return [], 0.0, []

    candidates = list_candidate_tickers(df, target_col=target_col)
    if not candidates:
        return [], 0.0, []

    print(f"Found {len(candidates)} candidate tickers: {candidates}")
    # quick greedy search
    selected, best_acc, history = sequential_forward_selection(
        df, candidates, max_features=max_features, target_col=target_col, **eval_kwargs
    )

    # optionally do exhaustive search for small candidate counts
    if len(candidates) <= exhaustive_threshold:
        for r in range(1, min(max_features, len(candidates)) + 1):
            for comb in itertools.combinations(candidates, r):
                acc = evaluate_feature_set(df, list(comb), target_col=target_col, **eval_kwargs)
                history.append({"trial": list(comb), "acc": acc})
                if acc > best_acc:
                    best_acc = acc
                    selected = list(comb)

    return selected, best_acc, history


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Feature selection on engineered CSV")
    parser.add_argument("engineered_csv")
    parser.add_argument("--max-features", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    best_set, best_acc, history = run_feature_selection(
        args.engineered_csv,
        max_features=args.max_features,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
    print("Best set:", best_set)
    print("Best acc:", best_acc)
