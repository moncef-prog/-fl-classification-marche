"""
Partitionne les données brutes en 3 partitions hospitalières pour la simulation FL.

Usage :
    python scripts/prepare_data.py

Sorties : data/hopital_1/data.npz, hopital_2/data.npz, hopital_3/data.npz
Chaque fichier contient :
    X : (N, 22, 101)  — angles articulaires, channels-first
    y : (N,)          — labels (0=TD, 1=CP)

Répartition des patients (niveau patient, sans fuite inter-hôpital) :
    Hôpital 1 : CP[0:50]    + TD[0:20]
    Hôpital 2 : CP[50:99]   + TD[20:39]
    Hôpital 3 : CP[99:148]  + TD[39:59]
"""

import os
import sys

import numpy as np
import pandas as pd

# ─── Chemins ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_SRC = os.path.join(
    os.path.expanduser("~"),
    "Downloads", "federated_projet2A_data", "federated_projet2A",
)
DATA_OUT = os.path.join(PROJECT_ROOT, "data")

NPZ_CP = os.path.join(DATA_SRC, "archives", "AQM", "TDvsPC_new_no_shuffle", "CP.npz")
NPZ_TD = os.path.join(DATA_SRC, "archives", "AQM", "TDvsPC_new_no_shuffle", "TD.npz")
CSV_CP = os.path.join(DATA_SRC, "DataCP - all.csv")
CSV_TD = os.path.join(DATA_SRC, "DataTD - By Subject.csv")

# Indices des 22 canaux sélectionnés (angles articulaires pertinents)
CHANNEL_IDX = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 29]

# Labels : CP=1, TD=0  (convention utils.py du prof, order=[1,0])
LABEL_CP = 1
LABEL_TD = 0


def load_raw(npz_path: str, csv_path: str, label: int):
    """Charge un fichier .npz et retourne (X, y, idx_patients).

    Returns:
        X          : (N, 22, 101) array float32 — channels-first, 22 canaux sélectionnés
        y          : (N,) array int — label répété pour chaque cycle
        patient_ends : (P,) cumulative end indices par patient
    """
    npz = np.load(npz_path)
    X_raw = npz["Input"]  # (N, 101, 30)
    X = X_raw[:, :, CHANNEL_IDX]  # (N, 101, 22)
    X = X.transpose(0, 2, 1).astype(np.float32)  # (N, 22, 101) channels-first

    y = np.full(len(X), label, dtype=np.int64)

    idx_csv = pd.read_csv(csv_path, header=None)[1].to_numpy()
    return X, y, idx_csv


def patient_cycles(X: np.ndarray, y: np.ndarray, patient_ends: np.ndarray, patient_ids):
    """Extrait les cycles appartenant aux patients indiqués."""
    X_out, y_out = [], []
    for p in patient_ids:
        start = int(patient_ends[p - 1]) if p > 0 else 0
        end = int(patient_ends[p])
        X_out.append(X[start:end])
        y_out.append(y[start:end])
    return np.concatenate(X_out), np.concatenate(y_out)


def save_hospital(hopital_id: int, X: np.ndarray, y: np.ndarray):
    out_dir = os.path.join(DATA_OUT, f"hopital_{hopital_id}")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "data.npz")
    np.savez_compressed(out_path, X=X, y=y)
    n_cp = int((y == LABEL_CP).sum())
    n_td = int((y == LABEL_TD).sum())
    print(f"  Hopital {hopital_id} : {len(X)} cycles  (CP={n_cp}, TD={n_td})  -> {out_path}")


def main():
    print("Chargement des données sources...")
    X_cp, y_cp, ends_cp = load_raw(NPZ_CP, CSV_CP, LABEL_CP)
    X_td, y_td, ends_td = load_raw(NPZ_TD, CSV_TD, LABEL_TD)

    n_cp = len(ends_cp)  # 148 patients CP
    n_td = len(ends_td)  # 59 patients TD
    print(f"  CP : {n_cp} patients, {len(X_cp)} cycles")
    print(f"  TD : {n_td} patients, {len(X_td)} cycles")

    # ── Partition des patients en 3 hôpitaux ──────────────────────────────────
    splits_cp = [
        range(0, 50),    # Hôpital 1
        range(50, 99),   # Hôpital 2
        range(99, 148),  # Hôpital 3
    ]
    splits_td = [
        range(0, 20),    # Hôpital 1
        range(20, 39),   # Hôpital 2
        range(39, 59),   # Hôpital 3
    ]

    print("\nCréation des partitions hospitalières...")
    for h in range(3):
        X_h_cp, y_h_cp = patient_cycles(X_cp, y_cp, ends_cp, splits_cp[h])
        X_h_td, y_h_td = patient_cycles(X_td, y_td, ends_td, splits_td[h])

        X_h = np.concatenate([X_h_cp, X_h_td])
        y_h = np.concatenate([y_h_cp, y_h_td])

        # Mélange pour ne pas que le modèle apprenne l'ordre CP puis TD
        rng = np.random.default_rng(seed=42 + h)
        perm = rng.permutation(len(X_h))
        save_hospital(h + 1, X_h[perm], y_h[perm])

    print("\nPartition terminée.")


if __name__ == "__main__":
    main()
