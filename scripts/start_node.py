"""
Démarre un nœud Fed-BioMed.

Usage :
    .venv\\Scripts\\python.exe scripts\\start_node.py --node-id 1
    .venv\\Scripts\\python.exe scripts\\start_node.py --node-id 2
    .venv\\Scripts\\python.exe scripts\\start_node.py --node-id 3

Le nœud se connecte au researcher sur localhost:50051 (FBM_RESEARCHER_PORT).
Démarrez d'abord le researcher (notebook 03) avant de lancer les nœuds.
"""

import argparse
import os
import sys

# Racine du projet dans le path pour que les nœuds puissent importer modeles/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from fedbiomed.node.cli import start_node
from fedbiomed.node.config import NodeConfig

NODES_CONFIG_DIR = os.path.join(PROJECT_ROOT, "nodes_config")


def main():
    parser = argparse.ArgumentParser(description="Démarre un nœud Fed-BioMed")
    parser.add_argument("--node-id", type=int, required=True, choices=[1, 2, 3],
                        help="Identifiant du nœud (1, 2 ou 3)")
    args = parser.parse_args()

    node_root = os.path.join(NODES_CONFIG_DIR, f"node{args.node_id}")

    if not os.path.exists(node_root):
        print(f"[ERREUR] Config nœud introuvable : {node_root}")
        print("→ Lancez d'abord : python scripts/register_datasets.py")
        sys.exit(1)

    print(f"Démarrage du nœud {args.node_id} (root={node_root})...")
    config = NodeConfig(root=node_root, alias=f"hopital_{args.node_id}")
    start_node(config, node_args={"gpu": False, "debug": False})


if __name__ == "__main__":
    main()
