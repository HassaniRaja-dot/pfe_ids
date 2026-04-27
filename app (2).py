from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit
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

# ── Application ───────────────────────────────────────────────────
app = Flask(__name__)
CORS(app, origins="*")
socketio = SocketIO(app, cors_allowed_origins="*")

# ── Servir le HTML ────────────────────────────────────────────────
@app.route('/')
def index():
    return send_from_directory('.', 'ids_secureguard.html')

# ── Architecture GAT ──────────────────────────────────────────────
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

# ── Chargement des modèles ────────────────────────────────────────
IN_CHANNELS = 53
LABEL_NAMES = ["Normal","DDoS","DoS","Botnet",
               "Infiltration","BruteForce","WebAttack","PortScan"]

ATTACK_INFO = {
    "Normal":      {"behaviors":[], "advice":["Surveillance régulière","Maintenir les systèmes à jour"]},
    "DDoS":        {"behaviors":["Volume de paquets extrêmement élevé","Sources IP très diversifiées","Durée de flux très courte"],"advice":["Activer la protection anti-DDoS","Rate limiting sur le pare-feu","Bloquer les IPs malveillantes"]},
    "DoS":         {"behaviors":["Forte activité depuis une seule IP","Paquets très nombreux","Connexions TCP semi-ouvertes"],"advice":["Bloquer l'IP source","Activer SYN cookies","Limiter connexions par IP"]},
    "BruteForce":  {"behaviors":["Nombreuses tentatives de connexion","Même IP vers même port","Intervalle régulier"],"advice":["Installer Fail2Ban","Activer 2FA","Clés SSH uniquement"]},
    "Botnet":      {"behaviors":["Communications périodiques vers IP externe","Trafic chiffré à intervalles fixes"],"advice":["Isoler la machine infectée","Scanner avec antivirus","Bloquer les C&C"]},
    "Infiltration":{"behaviors":["Connexions depuis IP inconnues","Accès aux services inhabituels"],"advice":["Changer tous les mots de passe","Auditer les accès","Segmenter le réseau"]},
    "WebAttack":   {"behaviors":["Requêtes HTTP suspectes","Tentatives sur /admin, /config"],"advice":["Installer un WAF","Valider les entrées","Mettre à jour les frameworks"]},
    "PortScan":    {"behaviors":["Connexions vers nombreux ports","Paquets SYN sans handshake","Exploration systématique"],"advice":["Configurer Snort/Suricata","Fermer les ports inutilisés","Port knocking"]},
}

print("Chargement des modèles GAT...")
try:
    model_b = GAT_IDS(IN_CHANNELS, num_classes=2)
    model_b.load_state_dict(torch.load('gat_ids_binary.pth', map_location='cpu'))
    model_b.eval()
    print("  gat_ids_binary.pth  ✓")

    model_m = GAT_IDS(IN_CHANNELS, num_classes=8)
    model_m.load_state_dict(torch.load('gat_ids_multi.pth', map_location='cpu'))
    model_m.eval()
    print("  gat_ids_multi.pth   ✓")

except FileNotFoundError as e:
    print(f"Modèles non trouvés : {e}")
    print(f"Fichiers présents : {os.listdir('.')}")
    model_b = model_m = None


def analyze_csv(file):
    """Analyse un fichier CSV et retourne les prédictions GAT."""
    df = pd.read_csv(file)
    df = df.select_dtypes(include=[np.number])
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    if len(df) < 2:
        raise ValueError("Le fichier doit contenir au moins 2 lignes")

    X = df.values.astype(np.float32)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)

    n_feat = X_scaled.shape[1]
    if n_feat != IN_CHANNELS:
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
    edge_index = torch.tensor(np.vstack([cx.row, cx.col]), dtype=torch.long)
    graph = Data(x=torch.tensor(X_scaled, dtype=torch.float),
                 edge_index=edge_index)

    with torch.no_grad():
        out_b = mb(graph.x, graph.edge_index)
        out_m = mm(graph.x, graph.edge_index)
        preds_b = out_b.argmax(dim=1).numpy()
        probs_m = F.softmax(out_m, dim=1).numpy()
        preds_m = out_m.argmax(dim=1).numpy()

    n_attack = int((preds_b == 1).sum())
    dominant = int(pd.Series(preds_m[preds_b == 1]).mode()[0]) if n_attack > 0 else 0
    dominant_name = LABEL_NAMES[dominant]

    return {
        'total':          len(df),
        'n_attack':       n_attack,
        'n_normal':       int((preds_b == 0).sum()),
        'dominant_type':  dominant_name,
        'confidence':     round(float(probs_m[:, dominant].mean() * 100), 1),
        'by_type':        {LABEL_NAMES[i]: int((preds_m == i).sum()) for i in range(8)},
        'behaviors':      ATTACK_INFO.get(dominant_name, {}).get('behaviors', []),
        'advice':         ATTACK_INFO.get(dominant_name, {}).get('advice', []),
    }


# ── Route analyse classique ───────────────────────────────────────
@app.route('/analyze', methods=['POST'])
def analyze():
    if model_b is None:
        return jsonify({'error': 'Modèles non chargés'}), 500
    file = request.files.get('file')
    if not file:
        return jsonify({'error': 'Aucun fichier reçu'}), 400
    try:
        result = analyze_csv(file)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── Route streaming (depuis agent_streaming.py sur ton PC) ────────
@app.route('/analyze-stream', methods=['POST'])
def analyze_stream():
    """
    Reçoit un CSV depuis l'agent sur ton PC,
    analyse avec le GAT, et pousse vers tous les navigateurs
    connectés via WebSocket en temps réel.
    """
    if model_b is None:
        return jsonify({'error': 'Modèles non chargés'}), 500
    file = request.files.get('file')
    if not file:
        return jsonify({'error': 'Aucun fichier reçu'}), 400
    try:
        result = analyze_csv(file)
        # Pousser vers tous les navigateurs connectés
        socketio.emit('detection_result', result)
        print(f"Résultat émis : {result['n_attack']} attaques / {result['total']} flux")
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── WebSocket events ──────────────────────────────────────────────
@socketio.on('connect')
def on_connect():
    print(f"Client connecté")
    emit('status', {'msg': 'Connecté au serveur IDS SecureGuard', 'ok': True})

@socketio.on('disconnect')
def on_disconnect():
    print("Client déconnecté")


if __name__ == '__main__':
    print("\nServeur IDS SecureGuard")
    print("Ouvrir : http://localhost:5000\n")
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)
