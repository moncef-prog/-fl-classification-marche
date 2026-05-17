"""
BiLSTM bidirectionnel pour classification de troubles de marche.
Utilisable en standalone (notebook centralisé) et comme TorchTrainingPlan Fed-BioMed.

Référence : Lan et al. 2024
Entrée    : (batch, 22, 101) — 22 angles articulaires, 101 pas de temps normalisés
"""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from fedbiomed.common.datamanager import DataManager
from fedbiomed.common.training_plans import TorchTrainingPlan


class GaitDataset(Dataset):
    """Dataset PyTorch pour cycles de marche (X: array, y: labels entiers)."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y.astype(int))

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


class BiLSTMNet(nn.Module):
    """
    Réseau BiLSTM bidirectionnel.

    Architecture :
        - LSTM(input=n_channels, hidden=hidden_size, layers=num_layers, bidirectionnel)
        - Dernier pas de temps → Dropout → FC(hidden*2 → n_classes)
    """

    def __init__(
        self,
        n_channels: int = 22,
        hidden_size: int = 64,
        num_layers: int = 2,
        n_classes: int = 2,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (batch, channels, seq_len) → (batch, seq_len, channels) pour LSTM
        x = x.transpose(1, 2)
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])  # dernier pas de temps
        return self.fc(out)


class BiLSTMTrainingPlan(TorchTrainingPlan):
    """
    Training plan Fed-BioMed encapsulant BiLSTMNet.

    model_args attendus :
        n_channels  (int, défaut 22)
        hidden_size (int, défaut 64)
        num_layers  (int, défaut 2)
        n_classes   (int, défaut 2)
        dropout     (float, défaut 0.5)

    Le nœud doit exposer un répertoire contenant data.npz
    (clés 'X': array (N,22,101), 'y': array (N,) d'entiers).
    """

    def init_dependencies(self):
        return [
            "import os",
            "import numpy as np",
            "import torch",
            "import torch.nn as nn",
            "import torch.nn.functional as F",
            "from torch.utils.data import Dataset",
            "from fedbiomed.common.datamanager import DataManager",
            "from modeles.lstm import BiLSTMNet, GaitDataset",
        ]

    def init_model(self, model_args=None):
        args = model_args or {}
        return BiLSTMNet(
            n_channels=args.get("n_channels", 22),
            hidden_size=args.get("hidden_size", 64),
            num_layers=args.get("num_layers", 2),
            n_classes=args.get("n_classes", 2),
            dropout=args.get("dropout", 0.5),
        )

    def training_data(self):
        npz_path = os.path.join(self.dataset_path, "data.npz")
        data = np.load(npz_path)
        dataset = GaitDataset(data["X"], data["y"])
        return DataManager(dataset=dataset, shuffle=True)

    def training_step(self, data, target):
        output = self.model()(data)
        return F.cross_entropy(output, target)

    def testing_step(self, data, target):
        output = self.model()(data)
        loss = F.cross_entropy(output, target).item()
        predicted = output.argmax(dim=1)
        acc = (predicted == target).float().mean().item()
        return {"loss": loss, "acc": acc}
