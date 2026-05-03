import gradio as gr
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import threading
import queue
import time
from datetime import datetime

# ── Architecture GAT ──────────────────────────────────────────────
class GAT_IDS(nn.Module):
    def __init__(self, in_channels, num_classes=2, dropout=0.3):
        super().__init__()
        self.conv1 = GATConv(in_channels, 128, heads=8, dropout=dropout, concat=True)
        self.conv2 = GATConv(1024, 64, heads=4, dropout=dropout, concat=True)
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

# ── Données sur les attaques ──────────────────────────────────────
LABEL_NAMES = ["Normal","DDoS","DoS","Botnet",
               "Infiltration","BruteForce","WebAttack","PortScan"]

ATTACK_INFO = {
    "Normal":      {"emoji":"✅","color":"#22a05a","desc":"Trafic réseau légitime.","behaviors":[],"advice":["Surveiller les logs régulièrement","Maintenir les systèmes à jour"]},
    "DDoS":        {"emoji":"🚨","color":"#dc2626","desc":"Attaque DDoS — saturation par des milliers de sources.","behaviors":["Volume de paquets extrêmement élevé","Sources IP très diversifiées","Durée de flux très courte"],"advice":["Activer protection anti-DDoS (Cloudflare)","Rate limiting sur le pare-feu","Bloquer les IPs malveillantes"]},
    "DoS":         {"emoji":"🔴","color":"#ef4444","desc":"Déni de service depuis une source unique.","behaviors":["Forte activité depuis une seule IP","Connexions TCP semi-ouvertes"],"advice":["Bloquer l'IP source immédiatement","Activer SYN cookies"]},
    "BruteForce":  {"emoji":"🔑","color":"#f97316","desc":"Tentatives répétées de connexion (SSH, FTP).","behaviors":["Nombreuses tentatives échouées","Même IP vers même port"],"advice":["Installer Fail2Ban","Activer 2FA","Clés SSH uniquement"]},
    "Botnet":      {"emoji":"🤖","color":"#7c3aed","desc":"Machine infectée communiquant avec C&C.","behaviors":["Communications périodiques vers IP externe","Trafic chiffré à intervalles fixes"],"advice":["Isoler la machine infectée","Scanner avec antivirus"]},
    "Infiltration":{"emoji":"🕵️","color":"#374151","desc":"Accès non autorisé au réseau interne.","behaviors":["Connexions depuis IP inconnues","Accès aux services inhabituels"],"advice":["Changer tous les mots de passe","Segmenter le réseau"]},
    "WebAttack":   {"emoji":"🕸️","color":"#db2777","desc":"Attaques web : SQL injection, XSS, CSRF.","behaviors":["Requêtes HTTP avec payloads suspects","Tentatives sur /admin, /config"],"advice":["Installer un WAF","Valider les entrées utilisateur"]},
    "PortScan":    {"emoji":"🔍","color":"#2563eb","desc":"Scan des ports pour cartographier les services.","behaviors":["Connexions rapides vers nombreux ports","Paquets SYN sans handshake"],"advice":["Configurer Snort/Suricata","Fermer les ports non utilisés"]},
}

# ── Chargement des modèles ────────────────────────────────────────
IN_CHANNELS = 53
MODELS_OK = False
model_b = model_m = None

print("Chargement des modèles GAT...")
try:
    model_b = GAT_IDS(IN_CHANNELS, num_classes=2)
    model_b.load_state_dict(torch.load('gat_ids_binary.pth', map_location='cpu'))
    model_b.eval()
    model_m = GAT_IDS(IN_CHANNELS, num_classes=8)
    model_m.load_state_dict(torch.load('gat_ids_multi.pth', map_location='cpu'))
    model_m.eval()
    MODELS_OK = True
    print("Modèles chargés ✓")
except Exception as e:
    print(f"Erreur : {e}")

# ── File d'attente pour les résultats en temps réel ───────────────
results_queue = queue.Queue(maxsize=100)
stream_history = []

def predict_from_df(df):
    """Prédit les attaques depuis un DataFrame."""
    df = df.select_dtypes(include=[np.number])
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    if len(df) < 2:
        return None

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
    graph = Data(x=torch.tensor(X_scaled, dtype=torch.float), edge_index=edge_index)

    with torch.no_grad():
        out_b = mb(graph.x, graph.edge_index)
        out_m = mm(graph.x, graph.edge_index)
        preds_b = out_b.argmax(dim=1).numpy()
        probs_m = F.softmax(out_m, dim=1).numpy()
        preds_m = out_m.argmax(dim=1).numpy()

    n_attack = int((preds_b == 1).sum())
    dominant = int(pd.Series(preds_m[preds_b==1]).mode()[0]) if n_attack > 0 else 0

    return {
        'timestamp':     datetime.now().strftime("%H:%M:%S"),
        'total':         len(df),
        'n_attack':      n_attack,
        'n_normal':      int((preds_b == 0).sum()),
        'dominant_type': LABEL_NAMES[dominant],
        'confidence':    round(float(probs_m[:, dominant].mean() * 100), 1),
        'by_type':       {LABEL_NAMES[i]: int((preds_m==i).sum()) for i in range(8)},
    }

def build_output(result):
    """Construit les textes de sortie depuis un résultat."""
    if result is None:
        return "⚠️ Données insuffisantes", "", "", ""

    dom = result['dominant_type']
    info = ATTACK_INFO[dom]
    n_attack = result['n_attack']
    n_total = result['total']
    conf = result['confidence']
    ts = result['timestamp']
    pct = round(n_attack / n_total * 100, 1) if n_total > 0 else 0
    taux = round((1 - n_attack/n_total)*100, 1) if n_total > 0 else 100

    # Résultat principal
    if n_attack == 0:
        main = f"""## {info['emoji']} Trafic Normal — {ts}

| Métrique | Valeur |
|:---|---:|
| Flux analysés | **{n_total:,}** |
| Flux normaux | **{n_total:,}** |
| Flux suspects | **0** |
| Taux de sécurité | **100%** |

*Aucune intrusion détectée dans cette fenêtre temporelle.*"""
    else:
        main = f"""## 🚨 ALERTE — {ts}

| Métrique | Valeur |
|:---|---:|
| Flux analysés | **{n_total:,}** |
| Flux suspects | **{n_attack:,} ({pct}%)** |
| Type dominant | **{info['emoji']} {dom}** |
| Confiance GAT | **{conf}%** |
| Taux sécurité | **{taux}%** |

{info['desc']}"""

    # Distribution
    rows = ""
    for i, name in enumerate(LABEL_NAMES):
        cnt = result['by_type'].get(name, 0)
        if cnt > 0:
            p = round(cnt/n_total*100, 1)
            bar = "█" * max(1, int(p/5))
            rows += f"| {ATTACK_INFO[name]['emoji']} {name} | {cnt:,} | {p}% | `{bar}` |\n"
    dist = f"## Distribution des flux\n\n| Type | Flux | % | |\n|:---|---:|---:|:---|\n{rows}"

    # Comportement
    if n_attack > 0 and info['behaviors']:
        beh = f"## Comportement détecté : {dom}\n\n" + "\n".join(f"- {b}" for b in info['behaviors'])
    else:
        beh = "## ✅ Aucun comportement anormal"

    # Conseils
    adv = f"## 🛡️ Recommandations\n\n" + "\n".join(f"{i+1}. {a}" for i, a in enumerate(info['advice']))

    return main, dist, beh, adv

# ── Interface Gradio ──────────────────────────────────────────────
css = """
.gradio-container { max-width: 1100px !important; }
footer { display:none !important; }
"""

with gr.Blocks(title="IDS SecureGuard — GAT Temps Réel",
               css=css, theme=gr.themes.Soft(primary_hue="slate")) as demo:

    gr.Markdown("""
    # 🛡️ IDS SecureGuard — Détection d'Intrusion en Temps Réel
    **Graph Attention Network | CIC-IDS 2018 | EST Meknès PFE 2026**

    Système de détection d'intrusion basé sur les Graph Attention Networks.
    Analyse le trafic réseau capturé entre machines virtuelles en temps réel.
    """)

    with gr.Tabs():

        # ── Onglet 1 : Streaming temps réel ─────────────────────────
        with gr.TabItem("📡 Analyse en Temps Réel"):
            gr.Markdown("""
            ### Comment utiliser
            1. Lancer `capture_agent.py` sur la VM Ubuntu (cible)
            2. Lancer les attaques depuis la VM Kali
            3. Les flux arriveront automatiquement ici toutes les **10 secondes**
            4. Cliquer **Rafraîchir** pour voir les derniers résultats
            """)

            with gr.Row():
                btn_refresh = gr.Button("🔄 Rafraîchir les résultats", variant="primary", size="lg")
                status_txt = gr.Textbox(label="Statut du système", value="En attente de flux réseau...",
                                        interactive=False, max_lines=1)

            with gr.Row():
                rt_main = gr.Markdown("*En attente du premier flux réseau...*")

            with gr.Row():
                rt_dist = gr.Markdown()
                rt_beh  = gr.Markdown()
            rt_adv = gr.Markdown()

            gr.Markdown("### Historique des détections (10 dernières)")
            rt_history = gr.Dataframe(
                headers=["Heure", "Flux", "Attaques", "Type", "Confiance"],
                datatype=["str","number","number","str","str"],
                label="",
                interactive=False
            )

            def refresh_results():
                if not stream_history:
                    return ("*En attente du premier flux...*", "", "", "",
                            pd.DataFrame(columns=["Heure","Flux","Attaques","Type","Confiance"]),
                            "En attente de flux réseau...")

                last = stream_history[-1]
                main, dist, beh, adv = build_output(last)

                hist_df = pd.DataFrame([
                    [r['timestamp'], r['total'], r['n_attack'],
                     r['dominant_type'], f"{r['confidence']}%"]
                    for r in stream_history[-10:]
                ], columns=["Heure","Flux","Attaques","Type","Confiance"])

                status = f"Dernière analyse : {last['timestamp']} — {last['n_attack']} attaques détectées"
                return main, dist, beh, adv, hist_df, status

            btn_refresh.click(
                fn=refresh_results,
                outputs=[rt_main, rt_dist, rt_beh, rt_adv, rt_history, status_txt]
            )

        # ── Onglet 2 : Upload CSV manuel ─────────────────────────────
        with gr.TabItem("📂 Analyser un fichier CSV"):
            gr.Markdown("Charger un fichier CSV généré par CICFlowMeter pour une analyse ponctuelle.")
            with gr.Row():
                with gr.Column(scale=1):
                    file_in = gr.File(label="Fichier CSV", file_types=[".csv"])
                    btn_csv = gr.Button("🔍 Analyser", variant="primary", size="lg")
                with gr.Column(scale=2):
                    csv_main = gr.Markdown()
            with gr.Row():
                csv_dist = gr.Markdown()
                csv_beh  = gr.Markdown()
            csv_adv = gr.Markdown()

            def analyze_csv(file):
                if file is None:
                    return "⚠️ Aucun fichier", "", "", ""
                try:
                    df = pd.read_csv(file.name)
                    result = predict_from_df(df)
                    if result:
                        stream_history.append(result)
                    return build_output(result)
                except Exception as e:
                    return f"❌ Erreur: {e}", "", "", ""

            btn_csv.click(fn=analyze_csv, inputs=[file_in],
                         outputs=[csv_main, csv_dist, csv_beh, csv_adv])

        # ── Onglet 3 : Démonstration ─────────────────────────────────
        with gr.TabItem("🎯 Démonstration"):
            gr.Markdown("Tester avec des scénarios simulés — aucun fichier ou VM requis.")
            with gr.Row():
                with gr.Column(scale=1):
                    scenario = gr.Radio(
                        choices=["Trafic Normal","Attaque DDoS","Scan de ports (Nmap)",
                                 "BruteForce SSH","Attaque DoS","Infection Botnet",
                                 "Attaque Web (SQLi)","Mix d'attaques"],
                        value="Attaque DDoS", label="Scénario"
                    )
                    btn_demo = gr.Button("▶ Lancer", variant="primary", size="lg")
                with gr.Column(scale=2):
                    demo_main = gr.Markdown()
            with gr.Row():
                demo_dist = gr.Markdown()
                demo_beh  = gr.Markdown()
            demo_adv = gr.Markdown()

            DEMO_DATA = {
                "Trafic Normal":       (200, 0,   "Normal",     99.1),
                "Attaque DDoS":        (250, 180, "DDoS",       96.3),
                "Scan de ports (Nmap)":(150, 95,  "PortScan",   92.7),
                "BruteForce SSH":      (180, 120, "BruteForce", 89.4),
                "Attaque DoS":         (160, 110, "DoS",        91.2),
                "Infection Botnet":    (130, 85,  "Botnet",     87.6),
                "Attaque Web (SQLi)":  (100, 65,  "WebAttack",  84.3),
                "Mix d'attaques":      (300, 190, "DDoS",       88.9),
            }

            def run_demo(sc):
                n_total, n_attack, dom, conf = DEMO_DATA.get(sc, (100,0,"Normal",99))
                result = {
                    'timestamp': datetime.now().strftime("%H:%M:%S"),
                    'total': n_total, 'n_attack': n_attack,
                    'n_normal': n_total - n_attack,
                    'dominant_type': dom, 'confidence': conf,
                    'by_type': {LABEL_NAMES[LABEL_NAMES.index(dom)]: n_attack,
                                "Normal": n_total - n_attack,
                                **{n: 0 for n in LABEL_NAMES if n not in [dom, "Normal"]}}
                }
                stream_history.append(result)
                return build_output(result)

            btn_demo.click(fn=run_demo, inputs=[scenario],
                          outputs=[demo_main, demo_dist, demo_beh, demo_adv])

        # ── Onglet 4 : API endpoint (pour l'agent) ───────────────────
        with gr.TabItem("🔌 Endpoint Agent"):
            gr.Markdown("""
            ### Endpoint pour `capture_agent.py`

            Ton agent sur la VM Ubuntu envoie les CSV à cette URL :
            ```
            POST https://HassaniRaja-dot-IDS-SecureGuard.hf.space/api/predict
            ```

            ### Format de la requête
            ```python
            import requests
            with open('traffic.csv', 'rb') as f:
                resp = requests.post(URL, files={'file': f})
            print(resp.json())
            ```

            ### Format de la réponse
            ```json
            {
              "timestamp": "14:32:11",
              "total": 150,
              "n_attack": 87,
              "dominant_type": "DDoS",
              "confidence": 94.2,
              "by_type": {"DDoS": 87, "Normal": 63, ...}
            }
            ```
            """)

        # ── Onglet 5 : À propos ──────────────────────────────────────
        with gr.TabItem("📊 Modèle"):
            gr.Markdown("""
            ## Architecture GAT_IDS

            | Couche | Entrée → Sortie |
            |:---|:---|
            | GATConv 1 | 53 → 128×8 = 1024 dims |
            | BatchNorm + ELU + Dropout | Stabilisation |
            | GATConv 2 | 1024 → 64×4 = 256 dims |
            | Linear classificateur | 256 → 2 (binaire) ou 256 → 8 (multi) |

            ## Résultats

            | Tâche | Accuracy | F1-Score |
            |:---|---:|---:|
            | Binaire (Normal/Attaque) | **91.52%** | **90.27%** |
            | Multi-classe (8 types) | **85.16%** | **67.29%** |

            ## Dataset CIC-IDS 2018

            | Paramètre | Valeur |
            |:---|:---|
            | Flux réseau | 1 672 723 |
            | Échantillon | 50 000 (stratifié) |
            | Features | 53 (après sélection) |
            | Graphe | KNN k=5 — cosine |

            ---
            *Raja Hassani — EST Meknès — Université Moulay Ismail — 2026*
            """)

    # ── API endpoint interne ──────────────────────────────────────
    def api_receive(file):
        """Endpoint appelé par capture_agent.py."""
        try:
            df = pd.read_csv(file.name)
            result = predict_from_df(df)
            if result:
                stream_history.append(result)
                if len(stream_history) > 200:
                    stream_history.pop(0)
            return result or {"error": "Analyse échouée"}
        except Exception as e:
            return {"error": str(e)}

    demo.load(fn=None)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True)
