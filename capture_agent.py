#!/usr/bin/env python3
"""
capture_agent.py — Script de capture réseau temps réel
Université Moulay Ismail · IDS-GAT · Projet de Fin d'Études

Usage:
    python3 capture_agent.py --interface eth0 --server http://10.0.2.2:5000
    python3 capture_agent.py --interface eth0 --server http://192.168.56.1:5000 --interval 2

Requis:
    pip install scapy pandas requests scikit-learn
    sudo apt install python3-scapy   (ou exécuter avec sudo)
"""

import argparse
import time
import math
import threading
import requests
import traceback
from collections import defaultdict, deque
from datetime import datetime

# Scapy doit être importé après avoir désactivé les warnings IPv6
import logging
logging.getLogger("scapy.runtime").setLevel(logging.ERROR)
from scapy.all import sniff, IP, TCP, UDP, ICMP

# ── Paramètres par défaut ─────────────────────────────────────────────────────
DEFAULT_SERVER   = "http://10.0.2.2:5000"
DEFAULT_INTERVAL = 2      # secondes entre envois
DEFAULT_BATCH    = 50     # nombre maximum de flux par batch
DEFAULT_IFACE    = "eth0"

# ── Mapping des features CIC-IDS ─────────────────────────────────────────────
# On extrait les features les plus importantes du dataset CIC-IDS
# à partir des paquets capturés par Scapy.
FEATURE_COLUMNS = [
    "Destination Port","Flow Duration","Total Fwd Packets","Total Backward Packets",
    "Total Length of Fwd Packets","Total Length of Bwd Packets",
    "Fwd Packet Length Max","Fwd Packet Length Min","Fwd Packet Length Mean","Fwd Packet Length Std",
    "Bwd Packet Length Max","Bwd Packet Length Min","Bwd Packet Length Mean","Bwd Packet Length Std",
    "Flow IAT Mean","Flow IAT Std","Flow IAT Max","Flow IAT Min",
    "Fwd IAT Total","Fwd IAT Mean","Fwd IAT Std","Fwd IAT Max","Fwd IAT Min",
    "Bwd IAT Total","Bwd IAT Mean","Bwd IAT Std","Bwd IAT Max","Bwd IAT Min",
    "Fwd PSH Flags","Bwd PSH Flags_drop",
    "Fwd Header Length","Bwd Header Length",
    "min_seg_size_forward","Init_Win_bytes_forward","Init_Win_bytes_backward",
    "act_data_pkt_fwd","min_seg_size_forward_2",
    "SYN Flag Count","RST Flag Count","PSH Flag Count","ACK Flag Count",
    "URG Flag Count","CWE Flag Count","ECE Flag Count",
    "Down/Up Ratio","Average Packet Size","Avg Fwd Segment Size_drop","Avg Bwd Segment Size_drop",
    "Fwd Header Length.1",
    "Packet Length Mean","Packet Length Std","Packet Length Max","Packet Length Min",
]

# ── Stockage des flux ─────────────────────────────────────────────────────────
flows = {}          # clé : (src_ip, dst_ip, src_port, dst_port, proto)
flows_lock = threading.Lock()
completed_flows = deque()


def flow_key(pkt):
    """Calcule la clé d'un flux à partir d'un paquet."""
    if IP not in pkt:
        return None
    src, dst = pkt[IP].src, pkt[IP].dst
    proto = pkt[IP].proto
    sport, dport = 0, 0
    if TCP in pkt:
        sport, dport = pkt[TCP].sport, pkt[TCP].dport
    elif UDP in pkt:
        sport, dport = pkt[UDP].sport, pkt[UDP].dport
    # Normaliser la clé (bidirectionnel)
    key = tuple(sorted([(src, sport), (dst, dport)])) + (proto,)
    return key


def safe_div(a, b, default=0.0):
    return a / b if b else default


def extract_flow_features(fdata):
    """Extrait les features CIC-IDS d'un flux capturé."""
    fwd = fdata["fwd_pkts"]
    bwd = fdata["bwd_pkts"]
    all_pkts = fwd + bwd

    n_fwd = len(fwd)
    n_bwd = len(bwd)

    # Tailles
    fwd_lens = [p["size"] for p in fwd]
    bwd_lens = [p["size"] for p in bwd]
    all_lens = fwd_lens + bwd_lens

    def stats(lst):
        if not lst:
            return 0, 0, 0, 0
        mn, mx = min(lst), max(lst)
        mean = sum(lst) / len(lst)
        if len(lst) > 1:
            var = sum((x-mean)**2 for x in lst)/(len(lst)-1)
            std = math.sqrt(var)
        else:
            std = 0
        return mx, mn, mean, std

    fwd_mx, fwd_mn, fwd_mean, fwd_std = stats(fwd_lens)
    bwd_mx, bwd_mn, bwd_mean, bwd_std = stats(bwd_lens)
    all_mx, all_mn, all_mean, all_std = stats(all_lens)

    # Durée du flux (en µs comme CIC-IDS)
    ts_list = [p["ts"] for p in all_pkts]
    if len(ts_list) >= 2:
        duration_us = int((max(ts_list) - min(ts_list)) * 1_000_000)
    else:
        duration_us = 0

    # IAT (Inter-Arrival Time)
    def iats(pkt_list):
        times = sorted(p["ts"] for p in pkt_list)
        if len(times) < 2:
            return []
        return [(times[i+1] - times[i]) * 1e6 for i in range(len(times)-1)]

    flow_iat = iats(all_pkts)
    fwd_iat  = iats(fwd)
    bwd_iat  = iats(bwd)

    flow_iat_mean,_,flow_iat_max,flow_iat_min = stats(flow_iat) if flow_iat else (0,0,0,0)
    _,_,flow_iat_std,_ = stats(flow_iat) if flow_iat else (0,0,0,0)
    flow_iat_mx,flow_iat_mn,flow_iat_m,flow_iat_s = stats(flow_iat)

    fwd_iat_tot = sum(fwd_iat)
    fwd_iat_mx,fwd_iat_mn,fwd_iat_m,fwd_iat_s = stats(fwd_iat)
    bwd_iat_tot = sum(bwd_iat)
    bwd_iat_mx,bwd_iat_mn,bwd_iat_m,bwd_iat_s = stats(bwd_iat)

    # Flags TCP
    flags = fdata.get("flags", defaultdict(int))
    syn = flags["S"]; rst = flags["R"]; psh = flags["P"]
    ack = flags["A"]; urg = flags["U"]; cwe = flags["C"]; ece = flags["E"]

    # Ports
    dst_port = fdata.get("dst_port", 0)

    # Taille moyenne paquet
    total_size = sum(all_lens)
    avg_pkt    = safe_div(total_size, len(all_pkts))

    features = {
        "Destination Port":          dst_port,
        "Flow Duration":             duration_us,
        "Total Fwd Packets":         n_fwd,
        "Total Backward Packets":    n_bwd,
        "Total Length of Fwd Packets": sum(fwd_lens),
        "Total Length of Bwd Packets": sum(bwd_lens),
        "Fwd Packet Length Max":     fwd_mx,
        "Fwd Packet Length Min":     fwd_mn,
        "Fwd Packet Length Mean":    fwd_mean,
        "Fwd Packet Length Std":     fwd_std,
        "Bwd Packet Length Max":     bwd_mx,
        "Bwd Packet Length Min":     bwd_mn,
        "Bwd Packet Length Mean":    bwd_mean,
        "Bwd Packet Length Std":     bwd_std,
        "Flow IAT Mean":             flow_iat_m,
        "Flow IAT Std":              flow_iat_s,
        "Flow IAT Max":              flow_iat_mx,
        "Flow IAT Min":              flow_iat_mn,
        "Fwd IAT Total":             fwd_iat_tot,
        "Fwd IAT Mean":              fwd_iat_m,
        "Fwd IAT Std":               fwd_iat_s,
        "Fwd IAT Max":               fwd_iat_mx,
        "Fwd IAT Min":               fwd_iat_mn,
        "Bwd IAT Total":             bwd_iat_tot,
        "Bwd IAT Mean":              bwd_iat_m,
        "Bwd IAT Std":               bwd_iat_s,
        "Bwd IAT Max":               bwd_iat_mx,
        "Bwd IAT Min":               bwd_iat_mn,
        "Fwd PSH Flags":             psh,
        "Bwd PSH Flags_drop":        0,
        "Fwd Header Length":         n_fwd * 20,
        "Bwd Header Length":         n_bwd * 20,
        "min_seg_size_forward":      fwd_mn if fwd_mn else 0,
        "Init_Win_bytes_forward":    fdata.get("init_win_fwd", 0),
        "Init_Win_bytes_backward":   fdata.get("init_win_bwd", 0),
        "act_data_pkt_fwd":          n_fwd,
        "min_seg_size_forward_2":    fwd_mn if fwd_mn else 0,
        "SYN Flag Count":            syn,
        "RST Flag Count":            rst,
        "PSH Flag Count":            psh,
        "ACK Flag Count":            ack,
        "URG Flag Count":            urg,
        "CWE Flag Count":            cwe,
        "ECE Flag Count":            ece,
        "Down/Up Ratio":             safe_div(n_bwd, n_fwd),
        "Average Packet Size":       avg_pkt,
        "Avg Fwd Segment Size_drop": fwd_mean,
        "Avg Bwd Segment Size_drop": bwd_mean,
        "Fwd Header Length.1":       n_fwd * 20,
        "Packet Length Mean":        all_mean,
        "Packet Length Std":         all_std,
        "Packet Length Max":         all_mx,
        "Packet Length Min":         all_mn,
    }
    return features


def packet_callback(pkt):
    """Callback appelé par Scapy pour chaque paquet capturé."""
    if IP not in pkt:
        return
    key = flow_key(pkt)
    if key is None:
        return

    ts   = float(pkt.time)
    size = len(pkt)

    with flows_lock:
        if key not in flows:
            flows[key] = {
                "fwd_pkts":    [],
                "bwd_pkts":    [],
                "flags":       defaultdict(int),
                "first_ts":    ts,
                "last_ts":     ts,
                "dst_port":    pkt[TCP].dport if TCP in pkt else (pkt[UDP].dport if UDP in pkt else 0),
                "init_win_fwd": pkt[TCP].window if TCP in pkt else 0,
                "init_win_bwd": 0,
                "pkt_count":   0,
            }

        fd = flows[key]
        fd["last_ts"]  = ts
        fd["pkt_count"] += 1

        pkt_info = {"ts": ts, "size": size}

        # Direction : fwd = paquet vers le port destination
        is_fwd = True
        if TCP in pkt:
            is_fwd = pkt[TCP].dport == fd["dst_port"]
            for f in str(pkt[TCP].flags):
                fd["flags"][f] += 1

        if is_fwd:
            fd["fwd_pkts"].append(pkt_info)
        else:
            fd["bwd_pkts"].append(pkt_info)
            if fd["init_win_bwd"] == 0 and TCP in pkt:
                fd["init_win_bwd"] = pkt[TCP].window

        # Compléter le flux après 30 paquets ou timeout
        should_complete = fd["pkt_count"] >= 30
        if should_complete:
            try:
                feat = extract_flow_features(fd)
                completed_flows.append(feat)
            except Exception:
                pass
            del flows[key]


def timeout_flows(interval):
    """Thread qui complète les flux inactifs depuis plus d'interval*2 secondes."""
    while True:
        time.sleep(interval)
        now = time.time()
        with flows_lock:
            to_delete = []
            for key, fd in flows.items():
                if now - fd["last_ts"] > interval * 2:
                    if fd["pkt_count"] >= 3:
                        try:
                            feat = extract_flow_features(fd)
                            completed_flows.append(feat)
                        except Exception:
                            pass
                    to_delete.append(key)
            for k in to_delete:
                del flows[k]


def send_loop(server, interval, batch_size):
    """Thread principal d'envoi des flux au serveur Flask."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Envoi vers {server}/api/realtime/ingest toutes les {interval}s")
    session = requests.Session()

    while True:
        time.sleep(interval)
        batch = []
        while completed_flows and len(batch) < batch_size:
            batch.append(completed_flows.popleft())

        if not batch:
            continue

        try:
            r = session.post(
                f"{server}/api/realtime/ingest",
                json={"flows": batch},
                timeout=5
            )
            if r.status_code == 200:
                d = r.json()
                print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                      f"Envoyé {len(batch)} flux → {d.get('processed',0)} analysés")
            else:
                print(f"[WARN] Serveur: {r.status_code} — {r.text[:100]}")
        except requests.exceptions.ConnectionError:
            print(f"[WARN] Impossible de joindre {server} — réessai dans {interval}s")
        except Exception as e:
            print(f"[ERR] {e}")


def main():
    parser = argparse.ArgumentParser(description="IDS-GAT Capture Agent")
    parser.add_argument("--interface", "-i", default=DEFAULT_IFACE,
                        help=f"Interface réseau (défaut: {DEFAULT_IFACE})")
    parser.add_argument("--server",    "-s", default=DEFAULT_SERVER,
                        help=f"URL du serveur Flask (défaut: {DEFAULT_SERVER})")
    parser.add_argument("--interval",  "-t", type=float, default=DEFAULT_INTERVAL,
                        help=f"Intervalle d'envoi en secondes (défaut: {DEFAULT_INTERVAL})")
    parser.add_argument("--batch",     "-b", type=int, default=DEFAULT_BATCH,
                        help=f"Taille du batch (défaut: {DEFAULT_BATCH})")
    args = parser.parse_args()

    print("=" * 60)
    print("  IDS-GAT Capture Agent — Université Moulay Ismail")
    print("=" * 60)
    print(f"  Interface : {args.interface}")
    print(f"  Serveur   : {args.server}")
    print(f"  Intervalle: {args.interval}s")
    print(f"  Batch     : {args.batch} flux")
    print("=" * 60)
    print()

    # Thread timeout flows
    t_timeout = threading.Thread(target=timeout_flows, args=(args.interval,), daemon=True)
    t_timeout.start()

    # Thread envoi
    t_send = threading.Thread(target=send_loop, args=(args.server, args.interval, args.batch), daemon=True)
    t_send.start()

    # Capture Scapy (bloquant)
    print(f"Capture sur {args.interface}… (Ctrl+C pour arrêter)")
    try:
        sniff(
            iface=args.interface,
            prn=packet_callback,
            store=False,
            filter="ip",  # Seulement le trafic IP
        )
    except KeyboardInterrupt:
        print("\nArrêt de la capture.")
    except PermissionError:
        print("[ERREUR] Permissions insuffisantes. Utilisez sudo.")
    except Exception as e:
        print(f"[ERREUR] {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
