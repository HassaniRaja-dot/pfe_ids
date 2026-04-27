from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import os

# ── Application Flask ─────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

# ── Route : servir le HTML ────────────────────────────────────────
@app.route('/')
def index():
    return send_from_directory('.', 'ids_secureguard.html')

# ── Architecture GAT (identique au notebook) ─────────────────────
class GAT_IDS(nn.Module):
    def __init__(self, in_channels, num_classes=2, dropout=0.3):
        super().__init__()
        self.conv1 = GATConv(in_channels, 128, heads=8,
                             dropout=dropout, concat=True)
        self.conv2 = GATConv(1024, 64, heads=4,
                             dropout=dropout, concat=True)
        self.classifier = nn.Linear(256, num_classes)
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.bn1(self.conv1(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.bn2(self.conv2(x, edge_index)))
        return self.classifier(x)

# ── Chargement des modèles au démarrage ──────────────────────────
IN_CHANNELS = 53

print("Chargement des modèles GAT...")
try:
    model_b = GAT_IDS(IN_CHANNELS, num_classes=2)
    model_b.load_state_dict(
        torch.load('gat_ids_binary.pth', map_location='cpu')
    )
    model_b.eval()
    print("  gat_ids_binary.pth  chargé ✓")

    model_m = GAT_IDS(IN_CHANNELS, num_classes=8)
    model_m.load_state_dict(
        torch.load('gat_ids_multi.pth', map_location='cpu')
    )
    model_m.eval()
    print("  gat_ids_multi.pth   chargé ✓")

except FileNotFoundError as e:
    print(f"\nERREUR : fichier .pth introuvable — {e}")
    print(f"Dossier courant : {os.getcwd()}")
    print(f"Fichiers présents : {os.listdir('.')}")
    model_b, model_m = None, None

LABEL_NAMES = [
    "Normal", "DDoS", "DoS", "Botnet",
    "Infiltration", "BruteForce", "WebAttack", "PortScan"
]

ATTACK_INFO = {
    "Normal":      {"behaviors": [], "advice": ["Surveillance régulière du réseau", "Maintenir les systèmes à jour"]},
    "DDoS":        {"behaviors": ["Volume de paquets extrêmement élevé", "Sources IP très diversifiées", "Durée de flux très courte"], "advice": ["Activer la protection anti-DDoS (Cloudflare)", "Configurer un rate limiting sur le pare-feu", "Bloquer les plages IP malveillantes"]},
    "DoS":         {"behaviors": ["Forte activité depuis une seule IP", "Paquets très nombreux", "Connexions TCP semi-ouvertes"], "advice": ["Bloquer l'IP source dans le pare-feu", "Activer SYN cookies sur le serveur", "Connexions max par IP"]},
    "BruteForce":  {"behaviors": ["Nombreuses tentatives de connexion", "Même IP vers même port", "Intervalle régulier"], "advice": ["Installer Fail2Ban", "Activer l'authentification 2FA", "Utiliser des clés SSH uniquement"]},
    "Botnet":      {"behaviors": ["Communications périodiques vers IP externe", "Trafic chiffré à intervalles fixes"], "advice": ["Isoler la machine infectée", "Scanner avec un antivirus à jour", "Bloquer les C&C connus"]},
    "Infiltration":{"behaviors": ["Connexions depuis IP inconnues", "Accès à des services inhabituels"], "advice": ["Changer tous les mots de passe", "Auditer les accès récents", "Segmenter le réseau (VLAN)"]},
    "WebAttack":   {"behaviors": ["Requêtes HTTP avec payloads suspects", "Tentatives sur /admin, /config"], "advice": ["Installer un WAF", "Valider toutes les entrées utilisateur", "Mettre à jour les frameworks web"]},
    "PortScan":    {"behaviors": ["Connexions rapides vers nombreux ports", "Paquets SYN sans handshake complet"], "advice": ["Configurer un IPS (Snort/Suricata)", "Fermer tous les ports non utilisés", "Utiliser le port knocking"]},
}

# ── Route : analyse un CSV ────────────────────────────────────────
@app.route('/analyze', methods=['POST'])
def analyze():
    if model_b is None or model_m is None:
        return jsonify({'error': 'Modèles non chargés — vérifier les fichiers .pth'}), 500

    file = request.files.get('file')
    if not file:
        return jsonify({'error': 'Aucun fichier reçu'}), 400

    try:
        df = pd.read_csv(file)
        df = df.select_dtypes(include=[np.number])
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

        if len(df) < 2:
            return jsonify({'error': 'Le fichier doit contenir au moins 2 lignes'}), 400

        X = df.values.astype(np.float32)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X).astype(np.float32)

        # Adapter IN_CHANNELS si différent de 53
        n_feat = X_scaled.shape[1]
        if n_feat != IN_CHANNELS:
            # Recréer les modèles avec le bon nombre de features
            mb = GAT_IDS(n_feat, num_classes=2)
            mb.load_state_dict(torch.load('gat_ids_binary.pth', map_location='cpu'))
            mb.eval()
            mm = GAT_IDS(n_feat, num_classes=8)
            mm.load_state_dict(torch.load('gat_ids_multi.pth', map_location='cpu'))
            mm.eval()
        else:
            mb, mm = model_b, model_m

        k = min(5, len(X_scaled) - 1)
        A = kneighbors_graph(X_scaled, n_neighbors=k, metric='cosine',
                             mode='connectivity', include_self=False, n_jobs=-1)
        A = A + A.T
        A.data[:] = 1
        cx = A.tocoo()
        edge_index = torch.tensor(
            np.vstack([cx.row, cx.col]), dtype=torch.long
        )
        graph = Data(
            x=torch.tensor(X_scaled, dtype=torch.float),
            edge_index=edge_index
        )

        with torch.no_grad():
            out_b = mb(graph.x, graph.edge_index)
            out_m = mm(graph.x, graph.edge_index)
            preds_b = out_b.argmax(dim=1).numpy()
            probs_m = F.softmax(out_m, dim=1).numpy()
            preds_m = out_m.argmax(dim=1).numpy()

        n_attack = int((preds_b == 1).sum())
        n_normal = int((preds_b == 0).sum())

        if n_attack > 0:
            dominant = int(pd.Series(preds_m[preds_b == 1]).mode()[0])
        else:
            dominant = 0

        dominant_name = LABEL_NAMES[dominant]
        confidence = float(probs_m[:, dominant].mean() * 100)
        by_type = {LABEL_NAMES[i]: int((preds_m == i).sum()) for i in range(8)}

        return jsonify({
            'total':         len(df),
            'n_attack':      n_attack,
            'n_normal':      n_normal,
            'dominant_type': dominant_name,
            'confidence':    round(confidence, 1),
            'by_type':       by_type,
            'behaviors':     ATTACK_INFO.get(dominant_name, {}).get('behaviors', []),
            'advice':        ATTACK_INFO.get(dominant_name, {}).get('advice', []),
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("\nServeur IDS SecureGuard démarré !")
    print("Ouvrir : http://localhost:5000\n")
    app.run(host='0.0.0.0', port=5000, debug=False)