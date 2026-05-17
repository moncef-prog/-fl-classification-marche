"""
InceptionTime pour classification de troubles de marche.
Utilisable en standalone et comme TorchTrainingPlan Fed-BioMed.

Référence : Ismail Fawaz et al. 2020 / Lan et al. 2024
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

NB_FILTERS = 32  # filtres par branche dans chaque module Inception


class InceptionModule(nn.Module):
    """
    Module Inception pour séries temporelles.

    3 branches convolutives (k=39, 19, 9) + 1 branche MaxPool.
    Précédé d'un goulot d'étranglement (k=1) pour réduire la dimensionnalité.
    """

    def __init__(self, in_channels: int, nb_filters: int = NB_FILTERS):
        super().__init__()
        self.bottleneck = nn.Conv1d(in_channels, nb_filters, kernel_size=1, bias=False)

        self.conv_large = nn.Conv1d(nb_filters, nb_filters, kernel_size=39, padding=19, bias=False)
        self.conv_mid = nn.Conv1d(nb_filters, nb_filters, kernel_size=19, padding=9, bias=False)
        self.conv_small = nn.Conv1d(nb_filters, nb_filters, kernel_size=9, padding=4, bias=False)

        # branche MaxPool → 1x1 conv sur l'entrée originale
        self.maxpool_branch = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, nb_filters, kernel_size=1, bias=False),
        )

        self.bn = nn.BatchNorm1d(nb_filters * 4)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = self.bottleneck(x)
        out = torch.cat([
            self.conv_large(b),
            self.conv_mid(b),
            self.conv_small(b),
            self.maxpool_branch(x),
        ], dim=1)
        return self.relu(self.bn(out))


class InceptionTime(nn.Module):
    """
    Réseau InceptionTime : 3 blocs Inception + connexions résiduelles + GlobalAvgPool.

    Architecture :
        Bloc 1 (22→128) → Bloc 2 (128→128) → Bloc 3 (128→128)
        Résidu tous les 3 blocs (Conv 1×1 si dimensions différentes)
        GlobalAvgPool → FC(128 → n_classes)
    """

    def __init__(self, n_channels: int = 22, n_classes: int = 2, nb_filters: int = NB_FILTERS):
        super().__init__()
        out_ch = nb_filters * 4  # 128

        self.block1 = InceptionModule(n_channels, nb_filters)
        self.block2 = InceptionModule(out_ch, nb_filters)
        self.block3 = InceptionModule(out_ch, nb_filters)

        # Connexion résiduelle sur les 3 blocs
        self.residual = nn.Sequential(
            nn.Conv1d(n_channels, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_ch),
        )
        self.relu = nn.ReLU()

        self.fc = nn.Linear(out_ch, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)

        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)

        out = self.relu(out + residual)
        out = out.mean(dim=2)  # GlobalAvgPool temporal
        return self.fc(out)


class InceptionTimeTrainingPlan(TorchTrainingPlan):
    """
    Training plan Fed-BioMed encapsulant InceptionTime.

    model_args attendus :
        n_channels (int, défaut 22)
        n_classes  (int, défaut 2)
        nb_filters (int, défaut 32)
    """

    def init_dependencies(self):
        return [
            "import os",
            "import numpy as np",
            "import torch",
            "import torch.nn as nn",
            "import torch.nn.functional as F",
            "from fedbiomed.common.datamanager import DataManager",
            "from modeles.inceptiontime import InceptionTime, InceptionModule, NB_FILTERS",
            "from modeles.lstm import GaitDataset",
        ]

    def init_model(self, model_args=None):
        args = model_args or {}
        return InceptionTime(
            n_channels=args.get("n_channels", 22),
            n_classes=args.get("n_classes", 2),
            nb_filters=args.get("nb_filters", NB_FILTERS),
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
