"""Model architectures: Model B, Observer Network, 4-stat extractor, and Observer Head."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from .utils import set_seed


class MLP256(nn.Module):
    """Standard 784 -> 256 -> 10 classification MLP with ReLU activation."""

    def __init__(self, input_dim: int = 784, hidden_dim: int = 256, num_classes: int = 10):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        h = self.relu(self.fc1(x))
        logits = self.fc2(h)
        return logits

    def forward_with_hidden(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return logits and post-ReLU hidden representations h [batch, 256]."""
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        h = self.relu(self.fc1(x))
        logits = self.fc2(h)
        return logits, h


def extract_4_stats(h: torch.Tensor) -> torch.Tensor:
    """Extract B's 4 internal activation statistics:
    1. Mean activation: h.mean()
    2. Activation spread (std): h.std()
    3. Fraction of awake neurons: (h > 0).mean()
    4. Activation magnitude: |h|.mean()

    Returns:
        Tensor of shape [batch, 4]
    """
    if h.dim() == 1:
        h = h.unsqueeze(0)
    feat1 = h.mean(dim=1, keepdim=True)
    feat2 = h.std(dim=1, keepdim=True, unbiased=False)
    feat3 = (h > 0).float().mean(dim=1, keepdim=True)
    feat4 = h.abs().mean(dim=1, keepdim=True)
    return torch.cat([feat1, feat2, feat3, feat4], dim=1)


class ObserverHead(nn.Module):
    """Observer Head: MLP 256 -> 64 -> 1 reading B's full hidden state to predict error."""

    def __init__(self, input_dim: int = 256, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h_b: torch.Tensor) -> torch.Tensor:
        """Returns raw logits [batch, 1]."""
        return self.net(h_b)

    def predict_proba(self, h_b: torch.Tensor) -> torch.Tensor:
        """Returns error probability in [0, 1]."""
        logits = self.forward(h_b)
        return torch.sigmoid(logits)


class SelfStatsProbe(nn.Module):
    """Probe reading B's 4 coarse internal statistics to predict error."""

    def __init__(self, input_dim: int = 4, hidden_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, stats: torch.Tensor) -> torch.Tensor:
        return self.net(stats)

    def predict_proba(self, stats: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(stats))


def train_mlp_classifier(
    train_loader: DataLoader,
    test_loader: DataLoader,
    seed: int,
    epochs: int = 3,
    lr: float = 1e-3,
    device: str = "cpu",
    name: str = "Model",
) -> Tuple[MLP256, Dict[str, List[float]]]:
    """Train MLP256 on training data for specified epochs with fixed seed."""
    set_seed(seed)
    model = MLP256().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "test_acc": []}
    print(f"--- Training {name} (seed={seed}) for {epochs} epochs ---")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_samples = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)
            total_samples += len(y)

        # Evaluate on test set
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                preds = model(x).argmax(dim=1)
                correct += (preds == y).sum().item()
                total += len(y)

        test_acc = correct / max(1, total)
        avg_loss = total_loss / max(1, total_samples)
        history["train_loss"].append(avg_loss)
        history["test_acc"].append(test_acc)
        print(f"Epoch {epoch}/{epochs} | Loss: {avg_loss:.4f} | Test Acc: {test_acc*100:.2f}%")

    return model, history


def extract_features_and_errors(
    model: MLP256,
    dataloader: DataLoader,
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """Pass dataloader through model and extract:
    - h: hidden representations [N, 256]
    - stats: 4-coarse stats [N, 4]
    - probs: softmax probabilities [N, 10]
    - conf: softmax max confidence [N]
    - conf_error_score: 1 - max confidence [N] (higher = more likely error)
    - preds: predicted class [N]
    - labels: ground truth class [N]
    - error: binary error indicator (1 = error, 0 = correct) [N]
    """
    model.eval()
    all_h, all_probs, all_preds, all_labels = [], [], [], []

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits, h = model.forward_with_hidden(x)
            probs = F.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)

            all_h.append(h.cpu())
            all_probs.append(probs.cpu())
            all_preds.append(preds.cpu())
            all_labels.append(y.cpu())

    h_tensor = torch.cat(all_h, dim=0)
    probs_tensor = torch.cat(all_probs, dim=0)
    preds_tensor = torch.cat(all_preds, dim=0)
    labels_tensor = torch.cat(all_labels, dim=0)

    conf = probs_tensor.max(dim=1).values
    conf_error_score = 1.0 - conf
    errors = (preds_tensor != labels_tensor).long()
    stats = extract_4_stats(h_tensor)

    return {
        "h": h_tensor,
        "stats": stats,
        "probs": probs_tensor,
        "conf": conf,
        "conf_error_score": conf_error_score,
        "preds": preds_tensor,
        "labels": labels_tensor,
        "errors": errors,
    }
