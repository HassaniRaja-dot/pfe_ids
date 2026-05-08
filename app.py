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
from datetime import datetime
import threading

# ─────────────────────────────────────────────────────────────────
# ARCHITECTURE GAT
# ─────────────────────────────────────────────────────────────────
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

# ─────────────────────────────────────────────────────────────────
# DONNÉES SUR LES ATTAQUES
# ─────────────────────────────────────────────────────────────────
LABEL_NAMES = ["Normal","DDoS","DoS","Botnet",
               "Infiltration","BruteForce","WebAttack","PortScan"]

ATTACK_INFO = {
    "Normal": {
        "emoji": "✅", "color": "#22a05a",
        "desc": "Trafic réseau légitime. Aucune activité suspecte détectée.",
        "behaviors": [],
        "advice": ["Surveiller les logs régulièrement",
                   "Maintenir les systèmes à jour",
                   "Effectuer des audits de sécurité périodiques"]
    },
    "DDoS": {
        "emoji": "🚨", "color": "#dc2626",
        "desc": "Attaque par déni de service distribué — saturation du serveur par des milliers de sources coordonnées.",
        "behaviors": ["Volume de paquets extrêmement élevé (>50 000/s)",
                      "Sources IP très diversifiées — botnet coordonné",
                      "Durée de flux très courte — SYN flood",
                      "Ratio Fwd/Bwd paquets anormal"],
        "advice": ["Activer la protection anti-DDoS (Cloudflare, AWS Shield)",
                   "Configurer un rate limiting strict sur le pare-feu",
                   "Bloquer les plages IP malveillantes identifiées",
                   "Utiliser un CDN pour absorber le trafic"]
    },
    "DoS": {
        "emoji": "🔴", "color": "#ef4444",
        "desc": "Attaque par déni de service depuis une source unique — sature les ressources du serveur cible.",
        "behaviors": ["Très forte activité depuis une seule IP source",
                      "Paquets de grande taille ou très nombreux",
                      "Connexions TCP semi-ouvertes (SYN flood)"],
        "advice": ["Bloquer immédiatement l'IP source dans le pare-feu",
                   "Activer SYN cookies sur le serveur",
                   "Limiter le nombre de connexions par IP"]
    },
    "BruteForce": {
        "emoji": "🔑", "color": "#f97316",
        "desc": "Tentatives répétées de connexion pour deviner les mots de passe (SSH, FTP, HTTP).",
        "behaviors": ["Nombreuses tentatives de connexion échouées",
                      "Intervalle très régulier entre les tentatives",
                      "Même IP source vers le même port destination"],
        "advice": ["Installer Fail2Ban pour bloquer automatiquement les IPs",
                   "Activer l'authentification à deux facteurs (2FA)",
                   "Utiliser des clés SSH uniquement",
                   "Changer les ports par défaut"]
    },
    "Botnet": {
        "emoji": "🤖", "color": "#7c3aed",
        "desc": "Machine infectée communiquant avec un serveur C&C (Command & Control).",
        "behaviors": ["Communications périodiques vers IP externe (beaconing)",
                      "Trafic chiffré à intervalles parfaitement fixes",
                      "Connexions vers des ports inhabituels"],
        "advice": ["Isoler immédiatement la machine infectée du réseau",
                   "Scanner avec un antivirus à jour",
                   "Bloquer les serveurs C&C connus via DNS"]
    },
    "Infiltration": {
        "emoji": "🕵️", "color": "#374151",
        "desc": "Accès non autorisé au réseau interne après exploitation d'une vulnérabilité.",
        "behaviors": ["Connexions depuis des IP inconnues vers ressources internes",
                      "Scans de ports depuis l'intérieur du réseau",
                      "Accès à des fichiers ou services inhabituels"],
        "advice": ["Changer immédiatement tous les mots de passe",
                   "Auditer les accès récents aux systèmes critiques",
                   "Segmenter le réseau avec des VLANs"]
    },
    "WebAttack": {
        "emoji": "🕸️", "color": "#db2777",
        "desc": "Attaques contre les applications web : injection SQL, XSS, CSRF.",
        "behaviors": ["Requêtes HTTP contenant des payloads suspects",
                      "Tentatives répétées sur /admin, /config, /phpmyadmin",
                      "Patterns d'injection SQL dans les paramètres URL"],
        "advice": ["Installer un Web Application Firewall (WAF)",
                   "Utiliser des requêtes préparées (parameterized queries)",
                   "Valider et assainir toutes les entrées utilisateur"]
    },
    "PortScan": {
        "emoji": "🔍", "color": "#2563eb",
        "desc": "Reconnaissance réseau — scan des ports ouverts pour cartographier les services disponibles.",
        "behaviors": ["Connexions rapides vers de nombreux ports différents",
                      "Nombreux paquets SYN sans handshake TCP complet",
                      "Exploration systématique de plages d'adresses IP"],
        "advice": ["Configurer Snort ou Suricata pour détecter les scans",
                   "Fermer tous les ports non utilisés dans le pare-feu",
                   "Utiliser le port knocking pour les services sensibles"]
    }
}

# ─────────────────────────────────────────────────────────────────
# CHARGEMENT DES MODÈLES
# ─────────────────────────────────────────────────────────────────
IN_CHANNELS = 53
MODELS_OK = False
model_b = model_m = None

print("Chargement des modèles GAT...")
try:
    model_b = GAT_IDS(IN_CHANNELS, num_classes=2)
    model_b.load_state_dict(torch.load('gat_ids_binary.pth', map_location='cpu'))
    model_b.eval()
    print("  gat_ids_binary.pth ✓")

    model_m = GAT_IDS(IN_CHANNELS, num_classes=8)
    model_m.load_state_dict(torch.load('gat_ids_multi.pth', map_location='cpu'))
    model_m.eval()
    print("  gat_ids_multi.pth  ✓")
    MODELS_OK = True
except Exception as e:
    print(f"  Erreur chargement modèles : {e}")

# ─────────────────────────────────────────────────────────────────
# HISTORIQUE EN MÉMOIRE (résultats temps réel)
# ─────────────────────────────────────────────────────────────────
stream_history = []
lock = threading.Lock()

# ─────────────────────────────────────────────────────────────────
# FONCTIONS CORE
# ─────────────────────────────────────────────────────────────────
def predict(df_raw):
    """Prédit les attaques depuis un DataFrame brut."""
    df = df_raw.select_dtypes(include=[np.number])
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    if len(df) < 2:
        return None

    X = df.values.astype(np.float32)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)
    n_feat = X_scaled.shape[1]

    if MODELS_OK and n_feat == IN_CHANNELS:
        mb, mm = model_b, model_m
    elif MODELS_OK:
        mb = GAT_IDS(n_feat, num_classes=2)
        mb.load_state_dict(torch.load('gat_ids_binary.pth', map_location='cpu'))
        mb.eval()
        mm = GAT_IDS(n_feat, num_classes=8)
        mm.load_state_dict(torch.load('gat_ids_multi.pth', map_location='cpu'))
        mm.eval()
    else:
        return None

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
        'date':          datetime.now().strftime("%d/%m/%Y"),
        'total':         len(df),
        'n_attack':      n_attack,
        'n_normal':      int((preds_b == 0).sum()),
        'dominant_type': LABEL_NAMES[dominant],
        'confidence':    round(float(probs_m[:, dominant].mean() * 100), 1),
        'by_type':       {LABEL_NAMES[i]: int((preds_m==i).sum()) for i in range(8)},
    }


def build_output(result):
    """Construit les 4 sorties Markdown depuis un résultat."""
    if not result:
        return "⚠️ Données insuffisantes.", "", "", ""

    dom   = result['dominant_type']
    info  = ATTACK_INFO[dom]
    n_atk = result['n_attack']
    n_tot = result['total']
    conf  = result['confidence']
    ts    = result['timestamp']
    dt    = result.get('date', '')
    pct   = round(n_atk / n_tot * 100, 1) if n_tot > 0 else 0
    taux  = round((1 - n_atk / n_tot) * 100, 1) if n_tot > 0 else 100

    # ── RÉSULTAT PRINCIPAL ────────────────────────────────────────
    if n_atk == 0:
        main = f"""## {info['emoji']} Trafic Normal — Aucune intrusion détectée
> 🕐 Analyse du {dt} à {ts}

| Métrique | Valeur |
|:---|---:|
| Flux analysés | **{n_tot:,}** |
| Flux normaux | **{n_tot:,}** |
| Flux suspects | **0** |
| Taux de sécurité | **100%** |
| Confiance GAT | **{conf}%** |

*Le modèle GAT n'a détecté aucune anomalie comportementale dans ce trafic réseau.*"""
    else:
        main = f"""## 🚨 ALERTE DE SÉCURITÉ — {n_atk:,} flux suspects détectés
> 🕐 Analyse du {dt} à {ts}

| Métrique | Valeur |
|:---|---:|
| Flux analysés | **{n_tot:,}** |
| Flux normaux | {n_tot - n_atk:,} |
| Flux suspects | **{n_atk:,} ({pct}%)** |
| Type dominant | **{info['emoji']} {dom}** |
| Taux de sécurité | **{taux}%** |
| Confiance GAT | **{conf}%** |

{info['desc']}"""

    # ── DISTRIBUTION ──────────────────────────────────────────────
    rows = ""
    for i, name in enumerate(LABEL_NAMES):
        cnt = result['by_type'].get(name, 0)
        if cnt > 0:
            p   = round(cnt / n_tot * 100, 1)
            bar = "█" * max(1, int(p / 4))
            rows += f"| {ATTACK_INFO[name]['emoji']} **{name}** | {cnt:,} | {p}% | `{bar}` |\n"

    dist = f"""## 📊 Distribution des flux détectés

| Type d'attaque | Flux | % | Barre |
|:---|---:|---:|:---|
{rows}
> *Graphe KNN — {n_tot:,} nœuds — k=5 — similarité cosinus*"""

    # ── COMPORTEMENT ─────────────────────────────────────────────
    if n_atk > 0 and info['behaviors']:
        beh_list = "\n".join(f"- {b}" for b in info['behaviors'])
        beh = f"""## {info['emoji']} Comportement détecté : {dom}

{info['desc']}

### Indicateurs observés dans les flux réseau
{beh_list}"""
    else:
        beh = "## ✅ Aucun comportement anormal\n\nLe trafic analysé correspond à des patterns réseau normaux."

    # ── CONSEILS ─────────────────────────────────────────────────
    adv_list = "\n".join(f"{i+1}. **{a}**" for i, a in enumerate(info['advice']))
    if n_atk > 0:
        adv = f"""## 🛡️ Recommandations de sécurité

### Actions immédiates à effectuer
{adv_list}

---
| Paramètre du modèle | Valeur |
|:---|:---|
| Architecture | GAT_IDS — 2 couches GATConv |
| Têtes d'attention | 8 (couche 1) + 4 (couche 2) |
| Dataset | CIC-IDS 2018 — 1 672 723 flux |
| Accuracy binaire | **91.52%** |
| Accuracy multi-classe | **85.16%** |"""
    else:
        adv = f"""## 🛡️ Bonnes pratiques de sécurité

{adv_list}

---
| Paramètre du modèle | Valeur |
|:---|:---|
| Architecture | GAT_IDS — 2 couches GATConv |
| Dataset | CIC-IDS 2018 — 1 672 723 flux |
| Accuracy binaire | **91.52%** |
| Accuracy multi-classe | **85.16%** |"""

    return main, dist, beh, adv


# ─────────────────────────────────────────────────────────────────
# INTERFACE GRADIO
# ─────────────────────────────────────────────────────────────────
css = """
.gradio-container { max-width: 1100px !important; }
footer { display: none !important; }
.tab-nav button { font-size: 14px !important; font-weight: 500 !important; }
"""

with gr.Blocks(
    title="IDS SecureGuard — GAT",
    css=css,
    theme=gr.themes.Soft(
        primary_hue="slate",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter")
    )
) as demo:

    # ── EN-TÊTE ───────────────────────────────────────────────────
    gr.Markdown("""
    # 🛡️ IDS SecureGuard
    ## Système de Détection d'Intrusion — Graph Attention Network
    **EST Meknès · Université Moulay Ismail · PFE 2026 · Raja Hassani**

    ---
    """)

    with gr.Tabs():

        # ══════════════════════════════════════════════════════════
        # ONGLET 1 : TEMPS RÉEL (depuis les VMs)
        # ══════════════════════════════════════════════════════════
        with gr.TabItem("📡 Temps Réel — VMs"):
            gr.Markdown("""
            ### Analyse en temps réel depuis les machines virtuelles

            L'agent `capture_agent.py` sur la **VM Ubuntu** envoie le trafic capturé
            entre Kali (attaquant) et Ubuntu (cible) toutes les **10 secondes**.

            👉 Cliquer **Rafraîchir** pour voir les dernières détections.
            """)

            with gr.Row():
                btn_refresh = gr.Button(
                    "🔄 Rafraîchir les résultats",
                    variant="primary", size="lg", scale=2
                )
                status_box = gr.Textbox(
                    label="Statut",
                    value="⏳ En attente du premier flux réseau...",
                    interactive=False, scale=3
                )

            with gr.Row():
                rt_main = gr.Markdown("*Aucune donnée reçue pour l'instant. Lancer l'agent sur Ubuntu.*")

            with gr.Row():
                rt_dist = gr.Markdown()
                rt_beh  = gr.Markdown()

            rt_adv = gr.Markdown()

            gr.Markdown("### 📋 Historique des 10 dernières analyses")
            rt_table = gr.Dataframe(
                headers=["Heure", "Flux", "Attaques", "Type", "Confiance", "Taux sécurité"],
                datatype=["str","number","number","str","str","str"],
                interactive=False,
                wrap=True
            )

            def do_refresh():
                with lock:
                    hist = list(stream_history)

                if not hist:
                    return (
                        "*Aucune donnée reçue. Lancer l'agent sur Ubuntu.*",
                        "", "", "",
                        pd.DataFrame(columns=["Heure","Flux","Attaques","Type","Confiance","Taux sécurité"]),
                        "⏳ En attente du premier flux réseau..."
                    )

                last = hist[-1]
                m, d, b, a = build_output(last)

                rows = []
                for r in reversed(hist[-10:]):
                    taux = round((1 - r['n_attack']/r['total'])*100, 1) if r['total'] > 0 else 100
                    rows.append([
                        r['timestamp'], r['total'], r['n_attack'],
                        r['dominant_type'], f"{r['confidence']}%", f"{taux}%"
                    ])
                df_hist = pd.DataFrame(rows, columns=["Heure","Flux","Attaques","Type","Confiance","Taux sécurité"])

                n_atk = last['n_attack']
                status = (f"🚨 {n_atk} attaques détectées — {last['dominant_type']} à {last['timestamp']}"
                          if n_atk > 0 else
                          f"✅ Trafic normal à {last['timestamp']}")

                return m, d, b, a, df_hist, status

            btn_refresh.click(
                fn=do_refresh,
                outputs=[rt_main, rt_dist, rt_beh, rt_adv, rt_table, status_box]
            )

        # ══════════════════════════════════════════════════════════
        # ONGLET 2 : UPLOAD CSV MANUEL
        # ══════════════════════════════════════════════════════════
        with gr.TabItem("📂 Analyser un fichier CSV"):
            gr.Markdown("""
            ### Analyse ponctuelle d'un fichier CSV
            Charger un fichier CSV généré par **CICFlowMeter** contenant les features réseau.
            """)

            with gr.Row():
                with gr.Column(scale=1):
                    file_in = gr.File(
                        label="📁 Fichier CSV (CICFlowMeter)",
                        file_types=[".csv"]
                    )
                    btn_csv = gr.Button("🔍 Analyser", variant="primary", size="lg")
                    gr.Markdown("""
                    **Format attendu :**
                    - CSV exporté par CICFlowMeter
                    - Features réseau (77 colonnes)
                    - Minimum 2 lignes

                    **Modèle GAT_IDS :**
                    - Binaire : **91.52%** accuracy
                    - Multi-classe : **85.16%** accuracy
                    """)
                with gr.Column(scale=2):
                    csv_main = gr.Markdown("*Charger un fichier CSV pour lancer l'analyse.*")

            with gr.Row():
                csv_dist = gr.Markdown()
                csv_beh  = gr.Markdown()
            csv_adv = gr.Markdown()

            def analyze_csv(file):
                if file is None:
                    return "⚠️ Aucun fichier sélectionné.", "", "", ""
                try:
                    df = pd.read_csv(file.name)
                    result = predict(df)
                    if result:
                        with lock:
                            stream_history.append(result)
                            if len(stream_history) > 200:
                                stream_history.pop(0)
                    return build_output(result)
                except Exception as e:
                    return f"❌ Erreur : {e}", "", "", ""

            btn_csv.click(
                fn=analyze_csv,
                inputs=[file_in],
                outputs=[csv_main, csv_dist, csv_beh, csv_adv]
            )

        # ══════════════════════════════════════════════════════════
        # ONGLET 3 : DÉMONSTRATION
        # ══════════════════════════════════════════════════════════
        with gr.TabItem("🎯 Démonstration"):
            gr.Markdown("""
            ### Scénarios d'attaques simulés
            Tester le système sans fichier CSV ni machines virtuelles.
            Idéal pour présenter le projet lors de la soutenance.
            """)

            with gr.Row():
                with gr.Column(scale=1):
                    scenario = gr.Radio(
                        choices=[
                            "🌐 Trafic 100% Normal",
                            "🚨 Attaque DDoS massive",
                            "🔴 Attaque DoS HTTP",
                            "🔑 BruteForce SSH",
                            "🔍 Scan de ports (Nmap)",
                            "🤖 Infection Botnet",
                            "🕸️ Attaque Web (SQLi)",
                            "⚡ Mix d'attaques réaliste"
                        ],
                        value="🚨 Attaque DDoS massive",
                        label="Choisir un scénario"
                    )
                    btn_demo = gr.Button(
                        "▶ Lancer la démonstration",
                        variant="primary", size="lg"
                    )
                with gr.Column(scale=2):
                    demo_main = gr.Markdown("*Sélectionner un scénario et cliquer Lancer.*")

            with gr.Row():
                demo_dist = gr.Markdown()
                demo_beh  = gr.Markdown()
            demo_adv = gr.Markdown()

            DEMO_SCENARIOS = {
                "🌐 Trafic 100% Normal":       (200, 0,   "Normal",     99.1),
                "🚨 Attaque DDoS massive":      (280, 210, "DDoS",       96.3),
                "🔴 Attaque DoS HTTP":          (160, 120, "DoS",        91.8),
                "🔑 BruteForce SSH":            (180, 130, "BruteForce", 89.4),
                "🔍 Scan de ports (Nmap)":      (150, 100, "PortScan",   92.7),
                "🤖 Infection Botnet":          (130, 85,  "Botnet",     87.6),
                "🕸️ Attaque Web (SQLi)":        (100, 65,  "WebAttack",  84.3),
                "⚡ Mix d'attaques réaliste":   (300, 195, "DDoS",       88.9),
            }

            def run_demo(sc):
                n_tot, n_atk, dom, conf = DEMO_SCENARIOS.get(sc, (100, 0, "Normal", 99.0))
                dom_idx = LABEL_NAMES.index(dom)
                by_type = {n: 0 for n in LABEL_NAMES}
                by_type[dom]     = n_atk
                by_type["Normal"] = n_tot - n_atk

                result = {
                    'timestamp':     datetime.now().strftime("%H:%M:%S"),
                    'date':          datetime.now().strftime("%d/%m/%Y"),
                    'total':         n_tot,
                    'n_attack':      n_atk,
                    'n_normal':      n_tot - n_atk,
                    'dominant_type': dom,
                    'confidence':    conf,
                    'by_type':       by_type,
                }
                with lock:
                    stream_history.append(result)
                    if len(stream_history) > 200:
                        stream_history.pop(0)
                return build_output(result)

            btn_demo.click(
                fn=run_demo,
                inputs=[scenario],
                outputs=[demo_main, demo_dist, demo_beh, demo_adv]
            )

        # ══════════════════════════════════════════════════════════
        # ONGLET 4 : STATISTIQUES GLOBALES
        # ══════════════════════════════════════════════════════════
        with gr.TabItem("📊 Statistiques"):
            gr.Markdown("### Statistiques globales de toutes les analyses effectuées")

            btn_stats = gr.Button("📊 Calculer les statistiques", variant="secondary")
            stats_out = gr.Markdown("*Lancer des analyses pour voir les statistiques.*")

            def compute_stats():
                with lock:
                    hist = list(stream_history)
                if not hist:
                    return "*Aucune analyse effectuée pour l'instant.*"

                total_analyses = len(hist)
                total_flux     = sum(r['total'] for r in hist)
                total_attacks  = sum(r['n_attack'] for r in hist)
                taux_moy       = round((1 - total_attacks/total_flux)*100, 1) if total_flux > 0 else 100

                by_type_global = {n: 0 for n in LABEL_NAMES}
                for r in hist:
                    for k, v in r['by_type'].items():
                        by_type_global[k] = by_type_global.get(k, 0) + v

                rows = ""
                for name, cnt in sorted(by_type_global.items(), key=lambda x: -x[1]):
                    if cnt > 0:
                        p   = round(cnt / total_flux * 100, 1) if total_flux > 0 else 0
                        bar = "█" * max(1, int(p / 3))
                        rows += f"| {ATTACK_INFO[name]['emoji']} {name} | {cnt:,} | {p}% | `{bar}` |\n"

                return f"""## 📊 Statistiques globales — {datetime.now().strftime("%d/%m/%Y %H:%M")}

| Métrique | Valeur |
|:---|---:|
| Analyses effectuées | **{total_analyses}** |
| Flux totaux analysés | **{total_flux:,}** |
| Flux suspects détectés | **{total_attacks:,}** |
| Taux de sécurité moyen | **{taux_moy}%** |

### Répartition par type d'attaque

| Type | Flux | % | Barre |
|:---|---:|---:|:---|
{rows}
---
*Modèle GAT_IDS — CIC-IDS 2018 — EST Meknès PFE 2026*"""

            btn_stats.click(fn=compute_stats, outputs=[stats_out])

        # ══════════════════════════════════════════════════════════
        # ONGLET 5 : À PROPOS DU MODÈLE
        # ══════════════════════════════════════════════════════════
        with gr.TabItem("ℹ️ À propos"):
            gr.Markdown("""
            ## Architecture GAT_IDS

            ### Pipeline complet
            ```
            Flux réseau (CSV CICFlowMeter)
                ↓
            StandardScaler — normalisation z-score
                ↓
            Graphe KNN (k=5, similarité cosinus)
            50 000 nœuds — ~390 000 arêtes
                ↓
            GATConv Couche 1 : 53 → 128×8 = 1024 dims (8 têtes)
            BatchNorm1d + ELU + Dropout(0.3)
                ↓
            GATConv Couche 2 : 1024 → 64×4 = 256 dims (4 têtes)
            BatchNorm1d + ELU + Dropout(0.3)
                ↓
            Linear(256 → 2)  → Binaire  : Normal / Attaque
            Linear(256 → 8)  → Multi    : 8 types d'attaques
            ```

            ### Résultats obtenus

            | Tâche | Accuracy | F1-Score |
            |:---|---:|---:|
            | Binaire (Normal vs Attaque) | **91.52%** | **90.27%** |
            | Multi-classe (8 types) | **85.16%** | **67.29%** |

            ### Dataset CIC-IDS 2018

            | Paramètre | Valeur |
            |:---|:---|
            | Flux réseau total | 1 672 723 |
            | Échantillon d'entraînement | 50 000 (stratifié) |
            | Features par flux | 53 (après sélection) |
            | Classes | 8 types d'attaques |
            | Graphe | KNN k=5 — similarité cosinus |

            ### Types d'attaques détectés

            | Type | Description |
            |:---|:---|
            | ✅ Normal | Trafic réseau légitime |
            | 🚨 DDoS | Déni de service distribué |
            | 🔴 DoS | Déni de service simple |
            | 🔑 BruteForce | Force brute SSH/FTP/HTTP |
            | 🔍 PortScan | Scan de ports (Nmap) |
            | 🤖 Botnet | Machine zombie C&C |
            | 🕵️ Infiltration | Accès non autorisé |
            | 🕸️ WebAttack | Injection SQL, XSS, CSRF |

            ---
            **Réalisé par :** Raja Hassani
            **Encadrante :** Mme Fatna El-Mendili
            **Établissement :** EST Meknès — Université Moulay Ismail
            **Année :** 2025/2026
            """)

# ─────────────────────────────────────────────────────────────────
# API ENDPOINT POUR L'AGENT UBUNTU
# ─────────────────────────────────────────────────────────────────
def api_receive_stream(file):
    """
    Endpoint appelé par capture_agent.py sur VM Ubuntu.
    Reçoit un CSV, analyse avec GAT, stocke le résultat.
    """
    try:
        df = pd.read_csv(file.name)
        result = predict(df)
        if result:
            with lock:
                stream_history.append(result)
                if len(stream_history) > 200:
                    stream_history.pop(0)
            return result
        return {"error": "Analyse échouée"}
    except Exception as e:
        return {"error": str(e)}

# Monter l'endpoint API sur /api/predict
demo.load(fn=None)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )
