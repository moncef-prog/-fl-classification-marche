"""Enregistre un seul nœud — appelé en sous-processus par register_datasets.py."""
import sys, os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

node_id = int(sys.argv[1])

from fedbiomed.node.config import NodeConfig
from fedbiomed.node.dataset_manager import DatasetManager
from fedbiomed.common.constants import CONFIG_FOLDER_NAME

TAG = "gait_marche"
node_root = os.path.join(PROJECT_ROOT, "nodes_config", f"node{node_id}")
data_path = os.path.join(PROJECT_ROOT, "data", f"hopital_{node_id}")

os.makedirs(os.path.join(node_root, "etc"), exist_ok=True)
config = NodeConfig(root=node_root, alias=f"hopital_{node_id}")
# Persiste config.ini pour que les appels suivants (start_node, vérification)
# retrouvent le même UUID et la même base de données.
config.write()
db_path = os.path.abspath(
    os.path.join(config.root, CONFIG_FOLDER_NAME, config.get("default", "db"))
)
os.makedirs(os.path.dirname(db_path), exist_ok=True)

dm = DatasetManager(path=db_path, min_samples=0)
existing = [d for d in dm.list_my_datasets(verbose=False) if TAG in d.get("tags", [])]

if existing:
    print(f"DEJA_ENREGISTRE:{node_id}:{len(existing)}")
else:
    dm.add_database(
        name=f"Donnees Hopital {node_id} Marche",
        data_type="custom",
        tags=[TAG],
        description=f"Cycles TD/CP hopital {node_id}",
        path=data_path,
    )
    print(f"ENREGISTRE:{node_id}:{data_path}")
