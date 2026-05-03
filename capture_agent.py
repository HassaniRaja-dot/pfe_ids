"""
capture_agent.py — À installer sur la VM Ubuntu (cible)
Capture le trafic réseau toutes les 10 secondes et envoie au site Hugging Face
"""
import subprocess, os, time, requests, pandas as pd, numpy as np
from datetime import datetime

# ── CONFIGURATION — modifier selon ton espace HF ─────────────────
HF_URL = "https://HassaniRaja-dot-IDS-SecureGuard.hf.space/api/predict"
INTERFACE = "eth0"          # interface réseau (eth0 sur DigitalOcean/GCP)
CAPTURE_SEC = 10            # capture toutes les 10 secondes
# ─────────────────────────────────────────────────────────────────

def get_interface():
    """Détecte automatiquement l'interface réseau active."""
    result = subprocess.run(['ip', 'route', 'get', '8.8.8.8'],
                            capture_output=True, text=True)
    for part in result.stdout.split():
        if part == 'dev':
            idx = result.stdout.split().index('dev')
            return result.stdout.split()[idx + 1]
    return INTERFACE

def capture_traffic(interface, duration):
    """Capture le trafic avec tcpdump."""
    pcap = f"/tmp/ids_cap_{int(time.time())}.pcap"
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Capture {duration}s sur {interface}...")
    try:
        subprocess.run(
            ["tcpdump", "-i", interface, "-w", pcap,
             "-G", str(duration), "-W", "1", "-q"],
            timeout=duration + 5,
            stderr=subprocess.DEVNULL
        )
    except subprocess.TimeoutExpired:
        pass
    return pcap if os.path.exists(pcap) else None

def pcap_to_csv(pcap_path):
    """Convertit le pcap en features CSV via tshark."""
    result = subprocess.run([
        "tshark", "-r", pcap_path,
        "-T", "fields",
        "-e", "ip.proto",
        "-e", "frame.len",
        "-e", "frame.time_delta",
        "-e", "tcp.flags",
        "-e", "ip.src",
        "-e", "ip.dst",
        "-e", "tcp.srcport",
        "-e", "tcp.dstport",
        "-E", "separator=,",
        "-E", "header=y"
    ], capture_output=True, text=True, timeout=30)

    lines = result.stdout.strip().split('\n')
    if len(lines) < 3:
        return None

    rows = []
    for line in lines[1:]:
        parts = line.split(',')
        if len(parts) < 4:
            continue
        row = [0.0] * 53
        try:
            row[0]  = float(parts[0]) if parts[0] else 0    # Protocol
            row[5]  = float(parts[1]) if parts[1] else 0    # Packet length
            row[1]  = float(parts[2]) * 1e6 if parts[2] else 0  # Duration µs
            flags   = int(parts[3], 16) if parts[3] else 0
            row[40] = float((flags >> 0) & 1)   # FIN
            row[41] = float((flags >> 1) & 1)   # SYN
            row[42] = float((flags >> 2) & 1)   # RST
            row[43] = float((flags >> 3) & 1)   # PSH
            row[44] = float((flags >> 4) & 1)   # ACK
        except (ValueError, IndexError):
            pass
        rows.append(row)

    if len(rows) < 2:
        return None

    cols = [f"feature_{i}" for i in range(53)]
    return pd.DataFrame(rows, columns=cols)

def send_to_hf(df):
    """Envoie le CSV au site Hugging Face."""
    csv_path = "/tmp/ids_traffic.csv"
    df.to_csv(csv_path, index=False)

    with open(csv_path, 'rb') as f:
        resp = requests.post(
            HF_URL,
            files={'file': ('traffic.csv', f, 'text/csv')},
            timeout=120
        )

    os.remove(csv_path)

    if resp.status_code == 200:
        d = resp.json()
        atk = d.get('n_attack', 0)
        tot = d.get('total', 0)
        typ = d.get('dominant_type', 'Normal')
        conf = d.get('confidence', 0)
        emoji = "🚨" if atk > 0 else "✅"
        print(f"  {emoji} {atk}/{tot} flux suspects — {typ} ({conf}%)")
        return d
    else:
        print(f"  ❌ Erreur serveur : {resp.status_code}")
        return None

# ── Boucle principale ─────────────────────────────────────────────
print("=" * 55)
print("  Agent IDS SecureGuard — Capture en temps réel")
print(f"  Serveur : {HF_URL}")
print(f"  Interface : {INTERFACE} | Intervalle : {CAPTURE_SEC}s")
print("=" * 55)

# Détection auto de l'interface
iface = get_interface()
print(f"  Interface détectée : {iface}")
print("  Démarrage... (Ctrl+C pour arrêter)")
print()

while True:
    try:
        pcap = capture_traffic(iface, CAPTURE_SEC)
        if pcap is None:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Aucun paquet capturé")
            continue

        df = pcap_to_csv(pcap)

        try:
            os.remove(pcap)
        except:
            pass

        if df is None or len(df) < 2:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Trafic insuffisant ({len(df) if df is not None else 0} paquets)")
            continue

        print(f"[{datetime.now().strftime('%H:%M:%S')}] {len(df)} paquets capturés → envoi au serveur...")
        send_to_hf(df)

    except KeyboardInterrupt:
        print("\nAgent arrêté.")
        break
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Erreur : {e}")

    time.sleep(2)
