"""Compte les datasets enregistres sur un noeud — appele en sous-processus."""
import sys, os, json, glob

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
node_id = int(sys.argv[1])

var_dir = os.path.join(PROJECT_ROOT, "nodes_config", f"node{node_id}", "var")
total = 0
for db_file in glob.glob(os.path.join(var_dir, "db_NODE_*.json")):
    try:
        with open(db_file, encoding="utf-8") as f:
            data = json.load(f)
        total += len(data.get("Datasets", {}))
    except Exception:
        pass

print(total)
