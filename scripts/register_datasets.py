"""
Enregistre les partitions hospitalières dans chaque nœud Fed-BioMed.

Chaque nœud est traité dans un sous-processus séparé pour éviter
le conflit du singleton TinyDB de Fed-BioMed.

Usage :
    .venv\\Scripts\\python.exe scripts\\register_datasets.py
"""

import os
import subprocess
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON = os.path.join(PROJECT_ROOT, ".venv", "Scripts", "python.exe")
HELPER = os.path.join(PROJECT_ROOT, "scripts", "_register_single_node.py")


def main():
    print("Enregistrement des datasets sur les noeuds Fed-BioMed...\n")

    for node_id in range(1, 4):
        data_path = os.path.join(PROJECT_ROOT, "data", f"hopital_{node_id}", "data.npz")
        if not os.path.exists(data_path):
            print(f"  [ERREUR] Donnees manquantes : {data_path}")
            print("  Lancez d'abord : python scripts/prepare_data.py")
            continue

        result = subprocess.run(
            [PYTHON, HELPER, str(node_id)],
            capture_output=True, text=True, encoding="utf-8"
        )

        # Filtrer les warnings Fed-BioMed pour ne garder que notre sortie
        for line in result.stdout.splitlines():
            if line.startswith("ENREGISTRE:"):
                _, nid, path = line.split(":", 2)
                print(f"  Noeud {nid} : dataset enregistre  -> {path}")
            elif line.startswith("DEJA_ENREGISTRE:"):
                _, nid, count = line.split(":")
                print(f"  Noeud {nid} : dataset deja enregistre ({count} entree(s))")

        if result.returncode != 0:
            print(f"  [ERREUR] Noeud {node_id}:")
            for line in result.stderr.splitlines():
                if "Traceback" in line or "Error" in line:
                    print(f"    {line}")

    print("\nEnregistrement termine.")
    print("Demarrez ensuite les noeuds avec : scripts\\start_all_nodes.bat")


if __name__ == "__main__":
    main()
