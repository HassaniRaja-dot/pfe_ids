"""
IDS-GAT — Système de Détection d'Intrusion basé sur Graph Attention Network
Université Moulay Ismail — Projet de Fin d'Études
"""

import os
import json
import time
import threading
import queue
import numpy as np
import pandas as pd
from datetime import datetime
from flask import (
    Flask, render_template, request, jsonify,
    Response, stream_with_context
)
from werkzeug.utils import secure_filename

from utils.model import load_models, LABEL_NAMES_BINARY, LABEL_NAMES_MULTI, ATTACK_COLORS
from utils.inference import preprocess_dataframe, predict_batch, compute_stats

# ── Configuration ─────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
MODEL_DIR  = os.path.join(BASE_DIR, "models")

os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_DIR
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200 MB
ALLOWED_EXTENSIONS = {"csv"}

# ── Chargement des modèles ─────────────────────────────────────────────────────
print("Chargement des modèles GAT...")
model_binary, model_multi = load_models(MODEL_DIR)
print("Modèles chargés.")

# ── File d'attente pour le temps réel ─────────────────────────────────────────
realtime_queue = queue.Queue(maxsize=500)
realtime_stats = {
    "running":        False,
    "total":          0,
    "attacks":        0,
    "normals":        0,
    "detection_rate": 0.0,
    "attack_types":   {},
    "last_flows":     [],   # 20 derniers flux
    "started_at":     None,
}
realtime_lock = threading.Lock()


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTES PRINCIPALES
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload")
def upload_page():
    return render_template("upload.html")


@app.route("/realtime")
def realtime_page():
    return render_template("realtime.html")


# ══════════════════════════════════════════════════════════════════════════════
#  API — ANALYSE CSV
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/upload", methods=["POST"])
def api_upload():
    if "file" not in request.files:
        return jsonify({"error": "Aucun fichier fourni"}), 400

    f = request.files["file"]
    if f.filename == "" or not allowed_file(f.filename):
        return jsonify({"error": "Fichier invalide (CSV uniquement)"}), 400

    filename = secure_filename(f.filename)
    filepath = os.path.join(UPLOAD_DIR, filename)
    f.save(filepath)

    try:
        df = pd.read_csv(filepath, low_memory=False)
    except Exception as e:
        return jsonify({"error": f"Erreur lecture CSV : {str(e)}"}), 400

    # Limiter à 5000 lignes pour la démo
    sample_size = min(len(df), 5000)
    df = df.sample(n=sample_size, random_state=42) if len(df) > sample_size else df
    df = df.reset_index(drop=True)

    try:
        X, scaler, features, y_true = preprocess_dataframe(df)
    except Exception as e:
        return jsonify({"error": f"Erreur preprocessing : {str(e)}"}), 400

    # Prédiction
    use_graph = len(X) >= 10
    try:
        results = predict_batch(X, model_binary, model_multi, use_graph=use_graph)
    except Exception as e:
        return jsonify({"error": f"Erreur prédiction : {str(e)}"}), 500

    stats = compute_stats(results)

    # Préparer un aperçu (100 premiers résultats)
    preview = results[:100]

    return jsonify({
        "success":     True,
        "filename":    filename,
        "rows":        len(df),
        "features":    len(features),
        "sample_size": sample_size,
        "stats":       stats,
        "preview":     preview,
        "label_names_multi": LABEL_NAMES_MULTI,
        "attack_colors":     ATTACK_COLORS,
    })


# ══════════════════════════════════════════════════════════════════════════════
#  API — TEMPS RÉEL (SSE)
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/realtime/start", methods=["POST"])
def realtime_start():
    with realtime_lock:
        realtime_stats["running"]        = True
        realtime_stats["total"]          = 0
        realtime_stats["attacks"]        = 0
        realtime_stats["normals"]        = 0
        realtime_stats["detection_rate"] = 0.0
        realtime_stats["attack_types"]   = {}
        realtime_stats["last_flows"]     = []
        realtime_stats["started_at"]     = datetime.now().isoformat()

    # Vider la file
    while not realtime_queue.empty():
        try:
            realtime_queue.get_nowait()
        except queue.Empty:
            break

    return jsonify({"success": True, "message": "Capture démarrée"})


@app.route("/api/realtime/stop", methods=["POST"])
def realtime_stop():
    with realtime_lock:
        realtime_stats["running"] = False
    return jsonify({"success": True, "message": "Capture arrêtée"})


@app.route("/api/realtime/ingest", methods=["POST"])
def realtime_ingest():
    """
    Point d'entrée pour le script de capture sur Kali/Ubuntu.
    Reçoit un JSON avec 'flows' : liste de dicts de features réseau.
    """
    if not realtime_stats.get("running"):
        return jsonify({"error": "Capture non démarrée"}), 400

    data = request.get_json(silent=True)
    if not data or "flows" not in data:
        return jsonify({"error": "Format invalide, attendu: {flows: [...]}"}), 400

    flows = data["flows"]
    if not flows:
        return jsonify({"ok": True, "processed": 0})

    try:
        df = pd.DataFrame(flows)
        X, _, _, _ = preprocess_dataframe(df)
        results = predict_batch(X, model_binary, model_multi, use_graph=False)

        with realtime_lock:
            for r in results:
                r["timestamp"] = datetime.now().strftime("%H:%M:%S")
                realtime_stats["total"] += 1
                if r["is_attack"]:
                    realtime_stats["attacks"] += 1
                    t = r["multi_label"]
                    realtime_stats["attack_types"][t] = realtime_stats["attack_types"].get(t, 0) + 1
                else:
                    realtime_stats["normals"] += 1

                realtime_stats["last_flows"].append(r)
                if len(realtime_stats["last_flows"]) > 20:
                    realtime_stats["last_flows"].pop(0)

            n = realtime_stats["total"]
            if n > 0:
                realtime_stats["detection_rate"] = round(realtime_stats["attacks"] / n * 100, 2)

        # Envoyer dans la file SSE
        try:
            realtime_queue.put_nowait(results)
        except queue.Full:
            pass

        return jsonify({"ok": True, "processed": len(results)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/realtime/stream")
def realtime_stream():
    """Server-Sent Events — le frontend s'abonne ici."""
    def generate():
        yield "data: {\"type\":\"connected\"}\n\n"
        while True:
            with realtime_lock:
                running = realtime_stats.get("running", False)
                stats_snapshot = {
                    "type":           "stats",
                    "running":        running,
                    "total":          realtime_stats["total"],
                    "attacks":        realtime_stats["attacks"],
                    "normals":        realtime_stats["normals"],
                    "detection_rate": realtime_stats["detection_rate"],
                    "attack_types":   realtime_stats["attack_types"],
                    "last_flows":     realtime_stats["last_flows"][-5:],
                }
            yield f"data: {json.dumps(stats_snapshot)}\n\n"
            time.sleep(2)

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control":               "no-cache",
            "X-Accel-Buffering":           "no",
            "Access-Control-Allow-Origin": "*",
        }
    )


@app.route("/api/realtime/stats")
def realtime_stats_api():
    with realtime_lock:
        return jsonify(realtime_stats)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000, threaded=True)
