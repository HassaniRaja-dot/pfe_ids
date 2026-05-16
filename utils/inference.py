import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph

from utils.model import COLS_TO_DROP, LABEL_NAMES_BINARY, LABEL_NAMES_MULTI, ATTACK_COLORS

try:
    from torch_geometric.data import Data
    HAS_PYG = True
except ImportError:
    HAS_PYG = False


def preprocess_dataframe(df: pd.DataFrame):
    """
    Applique le même preprocessing qu'à l'entraînement :
    - supprime les colonnes inutiles
    - supprime Label si présent
    - remplace inf/nan
    - normalise avec StandardScaler
    Retourne (X_tensor, scaler, feature_names, y_true si Label présent)
    """
    df = df.copy()

    # Récupérer les labels si présents
    y_true = None
    label_col = None
    for c in ["Label", "label", "CLASS", "class"]:
        if c in df.columns:
            label_col = c
            break

    if label_col:
        y_true = df[label_col].values
        df = df.drop(columns=[label_col])

    # Supprimer colonnes inutiles
    to_drop = [c for c in COLS_TO_DROP if c in df.columns]
    df = df.drop(columns=to_drop, errors='ignore')

    # Garder seulement colonnes numériques
    df = df.select_dtypes(include=[np.number])

    # Remplacer inf et nan
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    feature_names = df.columns.tolist()
    X = df.values.astype(np.float32)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return torch.tensor(X_scaled, dtype=torch.float32), scaler, feature_names, y_true


def build_knn_graph(X_tensor, k=5):
    """Construit le graphe KNN (même que l'entraînement)."""
    if not HAS_PYG:
        return None
    try:
        X_np = X_tensor.numpy()
        A = kneighbors_graph(X_np, n_neighbors=k, mode='connectivity',
                             metric='cosine', include_self=False, n_jobs=-1)
        A = A + A.T
        A.data[:] = 1
        cx = A.tocoo()
        edge_index = torch.tensor(
            np.vstack([cx.row, cx.col]), dtype=torch.long
        )
        return edge_index
    except Exception:
        return None


def predict_batch(X_tensor, model_binary, model_multi, use_graph=False, k=5):
    """
    Prédit sur un batch de flux réseau.
    Retourne liste de dicts par flux.
    """
    model_binary.eval()
    model_multi.eval()

    edge_index = None
    if use_graph and HAS_PYG and len(X_tensor) > k:
        edge_index = build_knn_graph(X_tensor, k=k)

    results = []

    with torch.no_grad():
        # Binaire
        if edge_index is not None:
            out_bin = model_binary(X_tensor, edge_index)
        else:
            out_bin = model_binary(X_tensor)
        prob_bin = F.softmax(out_bin, dim=1).numpy()
        pred_bin = np.argmax(prob_bin, axis=1)

        # Multi-classe
        if edge_index is not None:
            out_mul = model_multi(X_tensor, edge_index)
        else:
            out_mul = model_multi(X_tensor)
        prob_mul = F.softmax(out_mul, dim=1).numpy()
        pred_mul = np.argmax(prob_mul, axis=1)

    n = len(X_tensor)
    for i in range(n):
        binary_label = LABEL_NAMES_BINARY[pred_bin[i]]
        multi_label  = LABEL_NAMES_MULTI[pred_mul[i]]
        is_attack    = pred_bin[i] == 1

        results.append({
            "id":             i + 1,
            "is_attack":      is_attack,
            "binary_label":   binary_label,
            "binary_conf":    float(prob_bin[i][pred_bin[i]]) * 100,
            "multi_label":    multi_label,
            "multi_conf":     float(prob_mul[i][pred_mul[i]]) * 100,
            "color":          ATTACK_COLORS.get(multi_label, "#94a3b8"),
            "prob_normal":    float(prob_bin[i][0]) * 100,
            "prob_attack":    float(prob_bin[i][1]) * 100,
        })

    return results


def compute_stats(results):
    """Calcule les statistiques globales d'une analyse."""
    n = len(results)
    if n == 0:
        return {}

    attacks = [r for r in results if r["is_attack"]]
    normals = [r for r in results if not r["is_attack"]]

    # Distribution des types d'attaques
    attack_types = {}
    for r in attacks:
        t = r["multi_label"]
        attack_types[t] = attack_types.get(t, 0) + 1

    # Taux de détection
    detection_rate = len(attacks) / n * 100

    # Confiance moyenne
    avg_conf_bin = np.mean([r["binary_conf"] for r in results])
    avg_conf_mul = np.mean([r["multi_conf"] for r in results])

    return {
        "total":          n,
        "attacks":        len(attacks),
        "normals":        len(normals),
        "detection_rate": round(detection_rate, 2),
        "avg_conf_binary":round(float(avg_conf_bin), 2),
        "avg_conf_multi": round(float(avg_conf_mul), 2),
        "attack_types":   attack_types,
        "attack_colors":  {k: ATTACK_COLORS.get(k, "#94a3b8") for k in attack_types},
    }
