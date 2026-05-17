# Classification de troubles de marche — Apprentissage Fédéré

Classification TD vs CP sur cycles de marche (22 angles articulaires, 101 pas de temps)
via simulation locale de 3 hôpitaux avec **Fed-BioMed v6**.

**Résultats obtenus :** BiLSTM fédéré (FedAvg, 10 rounds) → **99.4 % accuracy** sur le test set.

---

## Prérequis

- **Python 3.11** (obligatoire — Fed-BioMed 6 ne supporte pas 3.12+)
- **Node.js** ≥ 18 avec `yarn` (requis par Fed-BioMed)
- Les données brutes dans `~/Downloads/federated_projet2A_data/federated_projet2A/`

Vérifier Python 3.11 :
```
py -3.11 --version
```
Si absent : `winget install Python.Python.3.11`

Vérifier yarn :
```
yarn --version
```
Si absent : `npm install -g yarn`

---

## Installation

```bash
# Créer l'environnement virtuel Python 3.11
py -3.11 -m venv .venv

# Activer
.venv\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt
```

L'installation de Fed-BioMed peut prendre 5–10 minutes (compile des binaires gRPC).

---

## Préparation des données

Les données brutes ne sont pas dans le dépôt (données médicales).
Elles doivent être dans :
```
~/Downloads/federated_projet2A_data/federated_projet2A/
    archives/AQM/TDvsPC_new_no_shuffle/CP.npz
    archives/AQM/TDvsPC_new_no_shuffle/TD.npz
    DataCP - all.csv
    DataTD - By Subject.csv
```

Puis lancer le partitionnement en 3 hôpitaux :
```bash
.venv\Scripts\python.exe scripts\prepare_data.py
```

Sortie attendue :
```
Hopital 1 : 1074 cycles  (CP=890, TD=184)
Hopital 2 : 1167 cycles  (CP=954, TD=213)
Hopital 3 : 1138 cycles  (CP=885, TD=253)
```

---

## Enregistrement des datasets Fed-BioMed

```bash
.venv\Scripts\python.exe scripts\register_datasets.py
```

---

## Vérification complète du pipeline

```bash
.venv\Scripts\python.exe scripts\_test_pipeline.py
```

Tous les checks doivent afficher `[OK]` :
- Chargement et split des données
- BiLSTM 3 epochs (~90 % accuracy)
- ResNet1D et InceptionTime forward pass
- Partitions hospitalières
- Datasets enregistrés sur les 3 noeuds Fed-BioMed

---

## Lancer l'expérience fédérée

```bash
.venv\Scripts\python.exe scripts\run_fl_experiment.py
```

Ce script :
1. Démarre les 3 noeuds en arrière-plan
2. Lance le researcher (serveur gRPC port 50051)
3. Exécute 10 rounds FedAvg avec BiLSTM
4. Sauvegarde le modèle dans `resultats/bilstm_federe.pt`

Durée : ~3–5 minutes.

---

## Évaluation du modèle fédéré

```bash
.venv\Scripts\python.exe scripts\evaluate_federe.py
```

---

## Structure du projet

```
modeles/
    lstm.py          # BiLSTM + TorchTrainingPlan Fed-BioMed
    resnet.py        # ResNet1D + TorchTrainingPlan
    inceptiontime.py # InceptionTime + TorchTrainingPlan

scripts/
    prepare_data.py          # Partitionne les données en 3 hôpitaux
    register_datasets.py     # Enregistre les datasets sur les noeuds
    run_fl_experiment.py     # Lance l'expérience FL complète
    evaluate_federe.py       # Évalue le modèle fédéré sur le test set
    start_node.py            # Démarre un noeud Fed-BioMed
    start_all_nodes.bat      # Démarre les 3 noeuds dans des fenêtres séparées
    _test_pipeline.py        # Tests de bout-en-bout

notebooks/
    01_exploration_donnees.ipynb   # Analyse exploratoire
    02_modele_centralise.ipynb     # Entraînement centralisé (référence)
    03_apprentissage_federe.ipynb  # Expérience FL (FedAvg + SCAFFOLD)
```

---

## Modèles disponibles

| Modèle        | Paramètres | Notes |
|---------------|-----------|-------|
| BiLSTM        | ~150k     | LSTM bidirectionnel, dropout 0.5 |
| ResNet1D      | ~270k     | 3 blocs résiduels, GlobalAvgPool |
| InceptionTime | ~220k     | 3 modules Inception + résidu |

Les 3 modèles sont disponibles comme `TorchTrainingPlan` Fed-BioMed
(`BiLSTMTrainingPlan`, `ResNet1DTrainingPlan`, `InceptionTimeTrainingPlan`).
