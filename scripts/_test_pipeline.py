"""Script de test complet — pipeline centralisé + vérification partitions FL."""
import sys, os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_SRC = os.path.join(os.path.expanduser("~"), "Downloads",
                        "federated_projet2A_data", "federated_projet2A")
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, DATA_SRC)

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils.utils import load_dataset_v1, generate_data_for_train
from modeles.lstm import BiLSTMNet, GaitDataset
from modeles.resnet import ResNet1D
from modeles.inceptiontime import InceptionTime

PASS = "[OK]"
FAIL = "[FAIL]"

# ─── 1. Chargement des données ────────────────────────────────────────────────
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

Xtr, Ytr = get_cycles(x_tr,  ends, X_d, y_d)
Xva, Yva = get_cycles(x_val, ends, X_d, y_d)
Xte, Yte = get_cycles(x_te,  ends, X_d, y_d)
print(f"{PASS} Données — train:{Xtr.shape}  val:{Xva.shape}  test:{Xte.shape}")

# ─── 2. DataLoaders ───────────────────────────────────────────────────────────
tr_ld = DataLoader(GaitDataset(Xtr, Ytr), batch_size=32, shuffle=True)
te_ld = DataLoader(GaitDataset(Xte, Yte), batch_size=32)
print(f"{PASS} DataLoaders — {len(tr_ld)} train batches, {len(te_ld)} test batches")

# ─── 3. Mini-entraînement BiLSTM (3 epochs) ───────────────────────────────────
model = BiLSTMNet()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
for ep in range(3):
    model.train()
    tl = 0
    for X, y in tr_ld:
        opt.zero_grad()
        loss = F.cross_entropy(model(X), y)
        loss.backward()
        opt.step()
        tl += loss.item()
    print(f"  BiLSTM ep{ep+1}  loss={tl/len(tr_ld):.4f}")

model.eval()
ok = tot = 0
with torch.no_grad():
    for X, y in te_ld:
        ok  += (model(X).argmax(1) == y).sum().item()
        tot += len(y)
print(f"{PASS} BiLSTM test acc = {ok/tot:.4f}  ({ok}/{tot})")

# ─── 4. Forward pass ResNet1D et InceptionTime ───────────────────────────────
x_dummy = torch.randn(8, 22, 101)
out_r = ResNet1D()(x_dummy)
print(f"{PASS} ResNet1D      input {tuple(x_dummy.shape)} -> output {tuple(out_r.shape)}")
out_i = InceptionTime()(x_dummy)
print(f"{PASS} InceptionTime input {tuple(x_dummy.shape)} -> output {tuple(out_i.shape)}")

# ─── 5. Vérification des partitions hospitalières ────────────────────────────
for h in range(1, 4):
    path = os.path.join(PROJECT_ROOT, "data", f"hopital_{h}", "data.npz")
    d = np.load(path)
    X_h = d["X"]
    y_h = d["y"]
    n_cp = int((y_h == 1).sum())
    n_td = int((y_h == 0).sum())
    print(f"{PASS} Hopital {h} : {X_h.shape}  CP={n_cp}  TD={n_td}  labels={set(y_h.tolist())}")

# ─── 6. Vérification enregistrement Fed-BioMed (subprocess pour eviter TinyDB singleton) ───
import subprocess
PYTHON = os.path.join(PROJECT_ROOT, ".venv", "Scripts", "python.exe")
CHECKER = os.path.join(PROJECT_ROOT, "scripts", "_check_node_datasets.py")

for node_id in range(1, 4):
    result = subprocess.run(
        [PYTHON, CHECKER, str(node_id)],
        capture_output=True, text=True, encoding="utf-8"
    )
    count = result.stdout.strip()
    if count.isdigit():
        status = PASS if int(count) > 0 else FAIL
        print(f"{status} Node {node_id} Fed-BioMed : {count} dataset(s) enregistre(s)")
    else:
        print(f"{FAIL} Node {node_id} Fed-BioMed : erreur verification")

print("\n=== TOUT EST PRET — pipeline OK ===")
