import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GATConv
    HAS_PYG = True
except ImportError:
    HAS_PYG = False

# ── Colonnes à supprimer (même preprocessing que l'entraînement) ─────────────
COLS_TO_DROP = [
    "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate",
    "Bwd PSH Flags", "Fwd URG Flags",
    "Avg Fwd Segment Size", "Avg Bwd Segment Size",
    "Subflow Fwd Packets", "Subflow Fwd Bytes",
    "Subflow Bwd Packets", "Subflow Bwd Bytes",
    "Packet Length Variance",
    "Flow Bytes/s", "Flow Packets/s",
    "Fwd Packets/s", "Bwd Packets/s",
    "Active Mean", "Active Std", "Active Max", "Active Min",
    "Idle Mean", "Idle Std", "Idle Max", "Idle Min",
]

LABEL_MAP = {
    "Normal": 0, "DDoS": 1, "DoS": 2, "Botnet": 3,
    "Infiltration": 4, "BruteForce": 5, "WebAttack": 6, "PortScan": 7
}
LABEL_NAMES_BINARY = ["Normal", "Attaque"]
LABEL_NAMES_MULTI  = list(LABEL_MAP.keys())

# Couleurs par classe pour l'interface
ATTACK_COLORS = {
    "Normal":      "#22c55e",
    "Attaque":     "#ef4444",
    "DDoS":        "#ef4444",
    "DoS":         "#f97316",
    "Botnet":      "#a855f7",
    "Infiltration":"#ec4899",
    "BruteForce":  "#f59e0b",
    "WebAttack":   "#3b82f6",
    "PortScan":    "#06b6d4",
}

IN_CHANNELS = 53  # features après suppression

class GAT_IDS(nn.Module):
    def __init__(self, in_channels=IN_CHANNELS, hidden1=128, hidden2=64,
                 heads1=8, heads2=4, num_classes=2, dropout=0.5):
        super(GAT_IDS, self).__init__()
        self.dropout = dropout

        if HAS_PYG:
            self.conv1 = GATConv(in_channels, hidden1, heads=heads1, dropout=dropout, concat=True)
            self.conv2 = GATConv(hidden1 * heads1, hidden2, heads=heads2, dropout=dropout, concat=True)
            self.bn1   = nn.BatchNorm1d(hidden1 * heads1)
            self.bn2   = nn.BatchNorm1d(hidden2 * heads2)
            self.classifier = nn.Linear(hidden2 * heads2, num_classes)
        else:
            # Fallback MLP si PyG non disponible
            self.fc1 = nn.Linear(in_channels, hidden1 * heads1)
            self.fc2 = nn.Linear(hidden1 * heads1, hidden2 * heads2)
            self.bn1 = nn.BatchNorm1d(hidden1 * heads1)
            self.bn2 = nn.BatchNorm1d(hidden2 * heads2)
            self.classifier = nn.Linear(hidden2 * heads2, num_classes)

    def forward(self, x, edge_index=None, return_attention=False):
        if HAS_PYG and edge_index is not None:
            x, attn1 = self.conv1(x, edge_index, return_attention_weights=True)
            x = self.bn1(x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x, attn2 = self.conv2(x, edge_index, return_attention_weights=True)
            x = self.bn2(x)
            x = F.elu(x)
            out = self.classifier(x)
            if return_attention:
                return out, attn1, attn2
            return out
        else:
            # Fallback : traitement nœud par nœud sans graphe
            x = F.elu(self.bn1(self.fc1(x)))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.elu(self.bn2(self.fc2(x)))
            return self.classifier(x)


def load_models(model_dir="models", device="cpu"):
    """Charge les deux modèles GAT."""
    model_binary = GAT_IDS(in_channels=IN_CHANNELS, num_classes=2)
    model_multi  = GAT_IDS(in_channels=IN_CHANNELS, num_classes=8)

    import os
    binary_path = os.path.join(model_dir, "gat_ids_binary.pth")
    multi_path  = os.path.join(model_dir, "gat_ids_multi.pth")

    if os.path.exists(binary_path):
        state = torch.load(binary_path, map_location=device)
        model_binary.load_state_dict(state, strict=False)

    if os.path.exists(multi_path):
        state = torch.load(multi_path, map_location=device)
        model_multi.load_state_dict(state, strict=False)

    model_binary.eval()
    model_multi.eval()
    return model_binary, model_multi
