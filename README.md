# IDS-GAT — Système de Détection d'Intrusion
**Université Moulay Ismail · Projet de Fin d'Études**

Graph Attention Network pour la détection d'anomalies réseau.
Entraîné sur CIC-IDS 2017 + CIC-IDS 2018.

---

## Structure du projet

```
ids_gat/
├── app.py                  # Application Flask principale
├── capture_agent.py        # Script de capture (Kali/Ubuntu VM)
├── requirements.txt
├── models/
│   ├── gat_ids_binary.pth  # Modèle binaire (Normal/Attaque)
│   └── gat_ids_multi.pth   # Modèle multi-classe (8 types)
├── utils/
│   ├── model.py            # Définition architecture GAT
│   └── inference.py        # Preprocessing + inférence
└── templates/
    ├── index.html          # Page d'accueil
    ├── upload.html         # Analyse CSV
    └── realtime.html       # Dashboard temps réel
```

---

## Installation et lancement

### 1. Sur la machine hôte (Windows/Linux)

```bash
# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# Installer les dépendances
pip install -r requirements.txt

# (Optionnel) PyTorch Geometric pour le graphe KNN complet
pip install torch-geometric

# Lancer le serveur
python app.py
```

Le serveur démarre sur **http://0.0.0.0:5000**

---

## Mode 1 : Analyse CSV

1. Ouvrir **http://localhost:5000/upload**
2. Déposer un fichier CSV au format CIC-IDS (colonnes comme le dataset d'entraînement)
3. Cliquer sur "Lancer l'analyse"
4. Voir les résultats : statistiques, graphiques, tableau détaillé

---

## Mode 2 : Capture Temps Réel (VirtualBox)

### Configuration réseau VirtualBox recommandée

**Option A — Host-Only Network (recommandée)**
- VM Kali/Ubuntu : Adaptateur → "Réseau hôte uniquement"
- L'IP hôte depuis la VM : `192.168.56.1` (par défaut)
- Vérifier avec `ip route` dans la VM

**Option B — NAT Network**
- VM : Adaptateur → "NAT"
- L'IP hôte depuis la VM : `10.0.2.2` (passerelle NAT)
- Redirection de port nécessaire dans VirtualBox :
  `Réseau → Avancé → Redirection de ports → TCP 5000→5000`

### Sur la VM Kali ou Ubuntu

```bash
# Installer les dépendances de capture
pip install scapy pandas requests scikit-learn
# ou
sudo apt install python3-scapy
pip install pandas requests scikit-learn

# Copier capture_agent.py sur la VM (via dossier partagé VirtualBox ou scp)

# Lancer la capture (nécessite sudo pour Scapy)
sudo python3 capture_agent.py \
    --interface eth0 \
    --server http://192.168.56.1:5000 \
    --interval 2 \
    --batch 50
```

### Dans l'interface web

1. Ouvrir **http://localhost:5000/realtime**
2. Cliquer sur **"Démarrer la capture"**
3. Lancer `capture_agent.py` sur la VM
4. Observer les flux arriver en temps réel toutes les 2 secondes

---

## Classes détectées

| ID | Classe      | Description                          |
|----|-------------|--------------------------------------|
| 0  | Normal      | Trafic légitime                      |
| 1  | DDoS        | Distributed Denial of Service        |
| 2  | DoS         | Denial of Service mono-source        |
| 3  | Botnet      | Machine zombie contrôlée à distance  |
| 4  | Infiltration| Accès non autorisé au réseau interne |
| 5  | BruteForce  | Tentatives de connexion répétées     |
| 6  | WebAttack   | SQL Injection, XSS…                  |
| 7  | PortScan    | Reconnaissance de ports              |

---

## Architecture GAT

```
Input (53 features) → KNN Graph (k=5, cosine similarity)
    ↓
GATConv(53 → 128, heads=8) → BatchNorm → ELU → Dropout(0.5)
    ↓
GATConv(1024 → 64, heads=4) → BatchNorm → ELU
    ↓
Linear(256 → 2)   → Binaire : Normal / Attaque
Linear(256 → 8)   → Multi   : 8 types d'attaques
```

**Performances :**
- Accuracy binaire   : **91.5%**
- Accuracy multi-classe : **85.2%**
- F1-Score (macro)   : **~88%**

---

## Notes

- Le serveur Flask limite l'analyse CSV à **5000 flux** (configurable dans `app.py`)
- Sans PyTorch Geometric, le modèle fonctionne en mode MLP (sans graphe) — précision légèrement réduite
- Le script `capture_agent.py` extrait les mêmes 53 features que le dataset CIC-IDS
