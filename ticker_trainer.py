"""Minimal Trainer and SimpleNN used by the pipeline and tests.

This module is a cleaned, generic trainer that uses 'target' as the default
label column. It replaces the legacy vti_from_mag7_predict module.
"""

import os
from typing import Optional

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler


class SimpleNN(nn.Module):
    def __init__(self, input_dim: int):
        super(SimpleNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Trainer:
    def __init__(
        self,
        engineered_csv: str,
        target_col: str = "target",
        batch_size: int = 64,
        epochs: int = 100,
        lr: float = 1e-3,
        train_ratio: float = 0.8,
        device: Optional[torch.device] = None,
    ):
        self.engineered_csv = engineered_csv
        self.target_col = target_col
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.train_ratio = train_ratio
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        torch.manual_seed(0)

        self.model: Optional[nn.Module] = None
        self.scaler: Optional[StandardScaler] = None

    def load_data(self):
        if not os.path.exists(self.engineered_csv):
            raise FileNotFoundError(f"Engineered CSV not found: {self.engineered_csv}")

        df = pd.read_csv(self.engineered_csv, index_col=0)
        if df.empty:
            raise ValueError("Engineered CSV is empty")

        if self.target_col not in df.columns:
            raise ValueError(f"Engineered CSV must contain '{self.target_col}' column")

        X = df.drop(columns=[self.target_col]).values.astype("float32")
        y = df[self.target_col].values.astype("float32")

        # standardize features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

        dataset = TensorDataset(X_tensor, y_tensor)
        train_size = int(len(dataset) * self.train_ratio)
        test_size = len(dataset) - train_size
        if train_size == 0 or test_size == 0:
            raise ValueError("Train/test split produced empty set; adjust train_ratio or provide more data")

        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size)
        self.input_dim = X_tensor.size(1)

    def build_model(self):
        assert hasattr(self, "input_dim"), "Call load_data() before build_model()"
        self.model = SimpleNN(self.input_dim).to(self.device)

    def train(self):
        self.load_data()
        self.build_model()

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            running_loss = 0.0
            for xb, yb in self.train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                preds = self.model(xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * xb.size(0)
            epoch_loss = running_loss / len(self.train_loader.dataset)

            # eval
            self.model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for xb, yb in self.test_loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    probs = torch.sigmoid(self.model(xb))
                    predicted_labels = (probs > 0.5).float()
                    total += yb.size(0)
                    correct += (predicted_labels == yb).sum().item()
            acc = correct / total if total > 0 else 0.0

            if epoch % max(1, self.epochs // 10) == 0 or epoch == 1:
                print(f"Epoch {epoch}/{self.epochs} - Loss: {epoch_loss:.6f} - Test Acc: {acc:.4f}")

        return self.model

    def save_model(self, out_path: str = "ticker_model.pth"):
        if self.model is None:
            raise RuntimeError("No trained model to save")
        torch.save(self.model.state_dict(), out_path)
        print(f"Model saved to {out_path}")
