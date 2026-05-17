"""Evalue le modele federe BiLSTM sur le test set centralise."""
import os, sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_SRC = os.path.join(os.path.expanduser("~"), "Downloads",
                        "federated_projet2A_data", "federated_projet2A")
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, DATA_SRC)

from utils.utils import load_dataset_v1, generate_data_for_train
from modeles.lstm import BiLSTMNet, GaitDataset

# ─── Chargement des donnees (meme split que le centralise) ───────────────────
dset_dir = os.path.join(DATA_SRC, "archives", "AQM", "TDvsPC_new_no_shuffle/")
x_list, _ = load_dataset_v1(dset_dir, ["CP.npz", "TD.npz"], channel_first=True, flatten=False)
idx_files = [
    os.path.join(DATA_SRC, "DataCP - all.csv"),
    os.path.join(DATA_SRC, "DataTD - By Subject.csv"),
]
x_tr, x_te, y_tr, y_te, x_val, y_val, ends, X_d, y_d = \
    generate_data_for_train(x_list, idx_files, order=[1, 0])

def get_cycles(pids, ends, X_d, y_d):
    Xl, yl = [], []
    for p in pids:
        s = int(ends[p-1]) if p > 0 else 0
        Xl.append(X_d[s:int(ends[p])])
        yl.append(y_d[s:int(ends[p])])
    return np.concatenate(Xl), np.concatenate(yl)

Xte, Yte = get_cycles(x_te, ends, X_d, y_d)
te_loader = DataLoader(GaitDataset(Xte, Yte), batch_size=64)

# ─── Chargement du modele federe ─────────────────────────────────────────────
model_path = os.path.join(PROJECT_ROOT, "resultats", "bilstm_federe.pt")
params = torch.load(model_path, map_location="cpu")

model = BiLSTMNet(n_channels=22, hidden_size=64, num_layers=2, n_classes=2, dropout=0.5)
model.load_state_dict(params)
model.eval()

# ─── Evaluation ──────────────────────────────────────────────────────────────
all_preds, all_targets = [], []
total_loss = 0.0
with torch.no_grad():
    for X, y in te_loader:
        out = model(X)
        total_loss += F.cross_entropy(out, y).item() * len(y)
        all_preds.extend(out.argmax(1).tolist())
        all_targets.extend(y.tolist())

n = len(all_targets)
acc = sum(p == t for p, t in zip(all_preds, all_targets)) / n
avg_loss = total_loss / n

print(f"\n=== Evaluation modele federe (BiLSTM FedAvg) ===")
print(f"Test set : {n} cycles  ({Xte.shape})")
print(f"Loss     : {avg_loss:.4f}")
print(f"Accuracy : {acc:.4f}  ({sum(p==t for p,t in zip(all_preds,all_targets))}/{n})")
print()
print(classification_report(all_targets, all_preds, target_names=["TD", "CP"], digits=3))
print("Matrice de confusion (lignes=reel, cols=predit):")
print("           TD    CP")
cm = confusion_matrix(all_targets, all_preds)
for label, row in zip(["TD", "CP"], cm):
    print(f"  {label} reel : {row}")
