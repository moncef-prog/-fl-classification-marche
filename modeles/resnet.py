"""
ResNet1D pour classification de troubles de marche.
Utilisable en standalone et comme TorchTrainingPlan Fed-BioMed.

Référence : Lan et al. 2024 / He et al. 2016 adapté 1D
Entrée    : (batch, 22, 101)
"""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fedbiomed.common.datamanager import DataManager
from fedbiomed.common.training_plans import TorchTrainingPlan

from modeles.lstm import GaitDataset


class ResidualBlock1D(nn.Module):
    """Bloc résiduel 1D avec connexion de contournement."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, stride: int = 1):
        super().__init__()
        pad = kernel // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel, stride=stride, padding=pad, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel, padding=pad, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class ResNet1D(nn.Module):
    """
    ResNet adapté aux séries temporelles de marche.

    Architecture :
        Conv initiale (22→64, k=7) → 3 blocs résiduels → GlobalAvgPool → FC
    """

    def __init__(self, n_channels: int = 22, n_classes: int = 2):
        super().__init__()
        self.init_conv = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.layer1 = ResidualBlock1D(64, 64, kernel=3)
        self.layer2 = ResidualBlock1D(64, 128, kernel=3, stride=2)
        self.layer3 = ResidualBlock1D(128, 128, kernel=3)
        self.fc = nn.Linear(128, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.init_conv(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.mean(dim=2)  # GlobalAvgPool temporal
        return self.fc(out)


class ResNet1DTrainingPlan(TorchTrainingPlan):
    """
    Training plan Fed-BioMed encapsulant ResNet1D.

    model_args attendus :
        n_channels (int, défaut 22)
        n_classes  (int, défaut 2)
    """

    def init_dependencies(self):
        return [
            "import os",
            "import numpy as np",
            "import torch",
            "import torch.nn as nn",
            "import torch.nn.functional as F",
            "from fedbiomed.common.datamanager import DataManager",
            "from modeles.resnet import ResNet1D, ResidualBlock1D",
            "from modeles.lstm import GaitDataset",
        ]

    def init_model(self, model_args=None):
        args = model_args or {}
        return ResNet1D(
            n_channels=args.get("n_channels", 22),
            n_classes=args.get("n_classes", 2),
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
