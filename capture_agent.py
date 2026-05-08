"""
capture_agent.py
================
À installer sur la VM Ubuntu (cible).
Capture le trafic réseau entre Kali et Ubuntu
toutes les 10 secondes et envoie au site Hugging Face.

Installation :
    sudo apt install -y python3-pip tcpdump tshark
    pip3 install requests pandas numpy scikit-learn

Lancement :
    sudo python3 capture_agent.py
"""

import subprocess
import os
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime

# ─────────────────────────────────────────────────────────────────
# CONFIGURATION — modifier ces 2 valeurs
# ─────────────────────────────────────────────────────────────────
HF_URL       = "https://HassaniRaja-dot-IDS-SecureGuard.hf.space/api/predict"
CAPTURE_SEC  = 10   # capturer toutes les 10 secondes
# ─────────────────────────────────────────────────────────────────


def detect_interface():
    """Détecte automatiquement l'interface réseau active."""
    try:
        result = subprocess.run(
            ['ip', 'route', 'get', '8.8.8.8'],
            capture_output=True, text=True, timeout=5
        )
        tokens = result.stdout.split()
        if 'dev' in tokens:
            return tokens[tokens.index('dev') + 1]
    except Exception:
        pass
    return "eth0"


def capture_traffic(interface, duration):
    """Capture le trafic réseau avec tcpdump."""
    pcap = f"/tmp/ids_{int(time.time())}.pcap"
    try:
        subprocess.run(
            ["tcpdump", "-i", interface, "-w", pcap,
             "-G", str(duration), "-W", "1", "-q"],
            timeout=duration + 5,
            stderr=subprocess.DEVNULL
        )
    except subprocess.TimeoutExpired:
        pass
    return pcap if os.path.exists(pcap) and os.path.getsize(pcap) > 0 else None


def pcap_to_features(pcap_path):
    """Convertit le fichier pcap en features réseau via tshark."""
    try:
        result = subprocess.run([
            "tshark", "-r", pcap_path,
            "-T", "fields",
            "-e", "ip.proto",
            "-e", "frame.len",
            "-e", "frame.time_delta",
            "-e", "tcp.flags",
            "-e", "tcp.srcport",
            "-e", "tcp.dstport",
            "-e", "udp.srcport",
            "-e", "udp.dstport",
            "-E", "separator=,",
            "-E", "header=y"
        ], capture_output=True, text=True, timeout=30)

        lines = result.stdout.strip().split('\n')
        if len(lines) < 3:
            return None

        rows = []
        for line in lines[1:]:
            parts = line.split(',')
            if len(parts) < 3:
                continue

            row = [0.0] * 53
            try:
                # Feature 0  : Protocole (6=TCP, 17=UDP)
                row[0]  = float(parts[0]) if parts[0] else 0
                # Feature 5  : Taille du paquet
                row[5]  = float(parts[1]) if parts[1] else 0
                # Feature 1  : Durée du flux (µs)
                row[1]  = float(parts[2]) * 1e6 if parts[2] else 0
                # Flags TCP
                if parts[3]:
                    try:
                        flags = int(parts[3], 16)
                        row[40] = float((flags >> 0) & 1)  # FIN
                        row[41] = float((flags >> 1) & 1)  # SYN
                        row[42] = float((flags >> 2) & 1)  # RST
                        row[43] = float((flags >> 3) & 1)  # PSH
                        row[44] = float((flags >> 4) & 1)  # ACK
                    except ValueError:
                        pass
                # Ports source/dest
                row[36] = float(parts[4]) if len(parts) > 4 and parts[4] else 0
                row[37] = float(parts[5]) if len(parts) > 5 and parts[5] else 0
            except (ValueError, IndexError):
                pass

            rows.append(row)

        if len(rows) < 2:
            return None

        cols = [f"feature_{i}" for i in range(53)]
        return pd.DataFrame(rows, columns=cols)

    except Exception as e:
        print(f"  Erreur tshark : {e}")
        return None


def send_to_server(df):
    """Envoie le CSV au site Hugging Face et retourne le résultat."""
    csv_path = "/tmp/ids_traffic_send.csv"
    df.to_csv(csv_path, index=False)

    try:
        with open(csv_path, 'rb') as f:
            resp = requests.post(
                HF_URL,
                files={'file': ('traffic.csv', f, 'text/csv')},
                timeout=120
            )

        os.remove(csv_path)

        if resp.status_code == 200:
            d = resp.json()
            n_atk = d.get('n_attack', 0)
            n_tot = d.get('total', 0)
            typ   = d.get('dominant_type', 'Normal')
            conf  = d.get('confidence', 0)
            emoji = "🚨" if n_atk > 0 else "✅"
            print(f"  {emoji} Résultat : {n_atk}/{n_tot} flux suspects — {typ} ({conf}%)")
            return d
        else:
            print(f"  ❌ Erreur serveur HTTP {resp.status_code}")
            return None

    except requests.exceptions.Timeout:
        print("  ❌ Timeout — serveur trop lent à répondre")
        return None
    except requests.exceptions.ConnectionError:
        print("  ❌ Impossible de joindre le serveur HF — vérifier la connexion internet")
        return None
    except Exception as e:
        print(f"  ❌ Erreur envoi : {e}")
        return None
    finally:
        if os.path.exists(csv_path):
            os.remove(csv_path)


# ─────────────────────────────────────────────────────────────────
# BOUCLE PRINCIPALE
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    iface = detect_interface()

    print("=" * 60)
    print("  IDS SecureGuard — Agent de capture réseau")
    print("=" * 60)
    print(f"  Serveur   : {HF_URL}")
    print(f"  Interface : {iface}")
    print(f"  Intervalle: {CAPTURE_SEC} secondes")
    print("  Appuyer Ctrl+C pour arrêter")
    print("=" * 60)
    print()

    while True:
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] Capture {CAPTURE_SEC}s sur {iface}...")

        try:
            # 1. Capturer le trafic
            pcap = capture_traffic(iface, CAPTURE_SEC)

            if pcap is None:
                print(f"[{ts}] Aucun paquet capturé — attendre du trafic réseau")
                time.sleep(CAPTURE_SEC)
                continue

            # 2. Convertir en features
            df = pcap_to_features(pcap)

            # Nettoyer le pcap
            try:
                os.remove(pcap)
            except Exception:
                pass

            if df is None or len(df) < 2:
                n = len(df) if df is not None else 0
                print(f"[{ts}] Trafic insuffisant ({n} paquets) — continuer la capture")
                time.sleep(2)
                continue

            print(f"[{ts}] {len(df)} paquets capturés → envoi au serveur...")

            # 3. Envoyer au serveur
            send_to_server(df)

        except KeyboardInterrupt:
            print("\n\nAgent arrêté. Au revoir !")
            break
        except Exception as e:
            print(f"[{ts}] Erreur inattendue : {e}")

        time.sleep(2)
