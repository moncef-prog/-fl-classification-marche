"""
Lance les 3 noeuds Fed-BioMed et demarre l'experience FL BiLSTM + FedAvg.

Usage :
    .venv\\Scripts\\python.exe scripts\\run_fl_experiment.py

Le script :
  1. Demarre les 3 noeuds en processus background
  2. Cree l'Experiment (researcher gRPC demarre)
  3. Attend la connexion des noeuds (auto-trust du certificat)
  4. Execute 10 rounds FedAvg
  5. Sauvegarde le modele federe dans resultats/bilstm_federe.pt
"""

import os
import subprocess
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

PYTHON = os.path.join(PROJECT_ROOT, ".venv", "Scripts", "python.exe")
RESEARCHER_CONFIG = os.path.join(PROJECT_ROOT, "nodes_config", "researcher")

# Doit etre defini AVANT l'import de fedbiomed.researcher pour que le singleton
# de ResearcherConfig soit initialise sur notre repertoire.
os.environ["FBM_RESEARCHER_COMPONENT_ROOT"] = RESEARCHER_CONFIG

TAG = "gait_marche"
N_ROUNDS = 10
N_EPOCHS_LOCAL = 3
BATCH_SIZE = 32
NODE_WAIT_SEC = 30   # secondes max pour attendre la connexion des noeuds

# ─── 1. Demarrage des noeuds en background ────────────────────────────────────
# Nettoyage des queues et etats obsoletes pour eviter les replays entre sessions
import shutil
for node_id in range(1, 4):
    var_dir = os.path.join(PROJECT_ROOT, "nodes_config", f"node{node_id}", "var")
    for entry in os.listdir(var_dir) if os.path.isdir(var_dir) else []:
        full = os.path.join(var_dir, entry)
        if entry.startswith("queue_") or entry.startswith("node_state_") or entry == "tmp":
            shutil.rmtree(full, ignore_errors=True)
            os.makedirs(full, exist_ok=True)

print("Demarrage des 3 noeuds Fed-BioMed en arriere-plan...")
os.makedirs(os.path.join(PROJECT_ROOT, "resultats"), exist_ok=True)
node_procs = []
node_logs = []
start_script = os.path.join(PROJECT_ROOT, "scripts", "start_node.py")
for node_id in range(1, 4):
    log_path = os.path.join(PROJECT_ROOT, "resultats", f"node{node_id}.log")
    log_file = open(log_path, "w", encoding="utf-8")
    node_logs.append((log_path, log_file))
    proc = subprocess.Popen(
        [PYTHON, start_script, "--node-id", str(node_id)],
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    node_procs.append(proc)
    print(f"  Noeud {node_id} PID={proc.pid} log -> {log_path}")

print("Noeuds lances. Initialisation du researcher...\n")
time.sleep(5)

# ─── 2. Creation de l'Experiment ─────────────────────────────────────────────
import torch
from fedbiomed.researcher.federated_workflows import Experiment
from fedbiomed.researcher.aggregators import FedAverage
from modeles.lstm import BiLSTMTrainingPlan

model_args = {
    "n_channels":  22,
    "hidden_size": 64,
    "num_layers":  2,
    "n_classes":   2,
    "dropout":     0.5,
}

training_args = {
    "loader_args":     {"batch_size": BATCH_SIZE},
    "optimizer_args":  {"lr": 1e-3},
    "epochs":          N_EPOCHS_LOCAL,
    "dry_run":         False,
    "batch_maxnum":    0,
}

print("Creation de l'Experiment FL (demarrage du serveur gRPC researcher)...")
exp = Experiment(
    tags=[TAG],
    training_plan_class=BiLSTMTrainingPlan,
    model_args=model_args,
    training_args=training_args,
    round_limit=N_ROUNDS,
    aggregator=FedAverage(),
    config_path=RESEARCHER_CONFIG,
)

# ─── 3. Attente connexion des noeuds ─────────────────────────────────────────
print(f"Attente de la connexion des noeuds (max {NODE_WAIT_SEC}s)...")
deadline = time.time() + NODE_WAIT_SEC
while time.time() < deadline:
    node_ids = list(exp.training_data().node_ids()) if exp.training_data() else []
    if len(node_ids) >= 3:
        break
    time.sleep(3)

node_ids = list(exp.training_data().node_ids()) if exp.training_data() else []
print(f"Noeuds connectes ({len(node_ids)}) : {node_ids}\n")

if len(node_ids) == 0:
    print("[ERREUR] Aucun noeud connecte. Arret.")
    for p in node_procs:
        p.terminate()
    sys.exit(1)

# ─── 4. Entrainement FL ───────────────────────────────────────────────────────
print(f"=== DEBUT FL : {N_ROUNDS} rounds FedAvg, {N_EPOCHS_LOCAL} epochs/round/noeud ===")
exp.run()
print("=== ENTRAINEMENT FL TERMINE ===\n")

# ─── 5. Sauvegarde du modele ─────────────────────────────────────────────────
os.makedirs(os.path.join(PROJECT_ROOT, "resultats"), exist_ok=True)
final_params = exp.aggregated_params()
if final_params:
    last_round = max(final_params.keys())
    model_path = os.path.join(PROJECT_ROOT, "resultats", "bilstm_federe.pt")
    torch.save(final_params[last_round]["params"], model_path)
    print(f"Modele federe (round {last_round}) sauvegarde -> {model_path}")
else:
    print("[AVERTISSEMENT] Aucun parametre agrege disponible.")

# ─── 6. Arret des noeuds ─────────────────────────────────────────────────────
print("\nArret des noeuds...")
for i, (proc, (log_path, log_file)) in enumerate(zip(node_procs, node_logs), 1):
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
    log_file.close()
    # Afficher les erreurs du log
    with open(log_path, encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    errors = [l.rstrip() for l in lines if "Error" in l or "Traceback" in l or "rror" in l]
    if errors:
        print(f"  [Noeud {i} erreurs] (log complet: {log_path})")
        for e in errors[-8:]:
            print(f"    {e}")
    print(f"  Noeud {i} arrete (code={proc.returncode})")

print("\nExperience FL terminee.")
