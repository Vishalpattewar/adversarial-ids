# =============================================================
# src/live_ids.py
# =============================================================
# PURPOSE:
#   Real-time network intrusion detection using the trained
#   IDSNet model. Captures live packets, extracts NSL-KDD-style
#   features, classifies each flow and raises alerts.
#
# WHAT THIS FILE DOES:
#   1. Loads trained IDSNet model + scaler from models/
#   2. Captures live IP packets using Scapy
#   3. Tracks per-flow statistics (src_bytes, counts, flags)
#   4. Extracts 38 NSL-KDD numeric features per packet
#   5. Classifies each flow as Normal (0) or Attack (1)
#   6. Displays a Rich terminal dashboard with live stats
#   7. Raises alerts and logs them to results/live_ids_alerts.log
#
# HOW TO RUN:
#   cd ~/A-IDS/adversarial_ids/src
#   sudo ../venv/bin/python3 live_ids.py
#
#   NOTE: sudo required for raw packet capture
#
# PRE-REQUISITE:
#   run_experiment.py must be run first to train and save models.
#
# BUGS FIXED VS ORIGINAL:
#   1. CRITICAL — Feature order mismatch
#      Original built feature vector in wrong order and injected
#      protocol_type (not in NUMERIC_FEATURES) at index 1.
#      Also missing num_outbound_cmds (16), is_host_login (17),
#      is_guest_login (18) — only 36 features instead of 38.
#      Fix: rebuilt to exactly match NUMERIC_FEATURES order.
#
#   2. CRITICAL — Memory leak
#      flow_stats dict grew unboundedly — every unique flow key
#      accumulated forever, eventually consuming all RAM.
#      Fix: added FLOW_TTL_SECONDS + purge_stale_flows() called
#      every 100 packets.
#
#   3. src_bytes counted full Ethernet frame including headers
#      Fix: use len(packet[IP].payload) — application data only
#
#   4. weights_only=False — security vulnerability
#      Fix: weights_only=True in torch.load()
#
#   5. input_dim hardcoded as 38
#      Fix: use len(NUMERIC_FEATURES) — stays correct if features change
#
# CDAC ITISS — Adversarial Input Attack on ML-based IDS
# =============================================================

import os
import sys
import time
import datetime
import threading
import torch
import numpy as np
import joblib

# ── Allow Python to find src/ modules ─────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from ids_model import IDSNet, NUMERIC_FEATURES

# ── Scapy for packet capture ───────────────────────────────────
from scapy.all import sniff, IP, TCP, UDP, ICMP, conf

# Suppress Scapy warnings (IPv6 interface warnings on WSL)
conf.verb = 0

# ── Rich for terminal dashboard ────────────────────────────────
from rich.console     import Console
from rich.table       import Table
from rich.panel       import Panel
from rich.layout      import Layout
from rich.live        import Live
from rich.text        import Text
from rich.columns     import Columns
from rich import box


# ─────────────────────────────────────────────────────────────
#  CONFIGURATION
#  Modify these values to change runtime behaviour
# ─────────────────────────────────────────────────────────────

# Path to trained model — using adversarial (hardened) model
MODEL_PATH  = '../models/ids_adversarial.pth'
SCALER_PATH = '../models/scaler_adversarial.pkl'
METRICS_PATH = '../models/metrics_adversarial.json'

# Classification threshold
# prob >= THRESHOLD → ATTACK, prob < THRESHOLD → NORMAL
THRESHOLD   = 0.5

# Network interface to sniff on (None = auto-detect)
INTERFACE   = 'lo'

# Alert log file path
LOG_FILE    = '../results/live_ids_alerts.log'

# Flow TTL: seconds before an idle flow is removed from memory
# Prevents unbounded memory growth (memory leak fix)
FLOW_TTL_SECONDS = 120

# How many recent alerts to show in the dashboard
MAX_ALERTS_DISPLAY = 8


# ─────────────────────────────────────────────────────────────
#  GLOBAL STATE
#  Shared between packet_handler() and the Rich dashboard.
#  Uses threading.Lock for thread-safe updates.
# ─────────────────────────────────────────────────────────────

# Lock for thread-safe access to shared state
state_lock = threading.Lock()

# Live statistics (updated per packet)
stats = {
    'packet_count' : 0,       # total packets processed
    'alert_count'  : 0,       # total alerts raised
    'normal_count' : 0,       # packets classified as normal
    'attack_count' : 0,       # packets classified as attack
    'last_prob'    : 0.0,     # last classification probability
    'start_time'   : time.time(),
}

# Recent alerts for dashboard display
recent_alerts = []

# Per-flow statistics dictionary
# Key   : "src_ip:src_port-dst_ip:dst_port"
# Value : dict of flow counters
# FIX   : includes last_seen for TTL-based cleanup
from collections import defaultdict

flow_stats = defaultdict(lambda: {
    'count'         : 0,      # total packets in this flow
    'src_bytes'     : 0,      # total payload bytes from source
    'dst_bytes'     : 0,      # total payload bytes from dest
    'syn_count'     : 0,      # SYN flag count
    'fin_count'     : 0,      # FIN flag count
    'rst_count'     : 0,      # RST flag count
    'start_time'    : time.time(),
    'last_seen'     : time.time(),  # FIX: for TTL-based cleanup
})

console = Console()


# ─────────────────────────────────────────────────────────────
#  MODEL LOADING
# ─────────────────────────────────────────────────────────────
def load_model():
    """
    Load trained IDSNet model and StandardScaler from disk.

    FIX: weights_only=True — secure loading, prevents arbitrary
         code execution via pickle (PyTorch 2.x requirement).

    FIX: input_dim=len(NUMERIC_FEATURES) instead of hardcoded 38
         — stays correct if feature list changes.

    Returns:
        model : IDSNet in eval mode
        scaler: Fitted StandardScaler
    """
    console.print("\n[cyan][*] Loading IDSNet model...[/cyan]")

    # Check model files exist
    for path in [MODEL_PATH, SCALER_PATH, METRICS_PATH]:
        if not os.path.exists(path):
            console.print(
                f"[red][✗] File not found: {path}[/red]\n"
                f"[yellow]    Run run_experiment.py first.[/yellow]"
            )
            sys.exit(1)

    try:
        # Load scaler
        scaler = joblib.load(SCALER_PATH)

        # Build model with correct input dimension
        # FIX: use len(NUMERIC_FEATURES) not hardcoded 38
        model = IDSNet(input_dim=len(NUMERIC_FEATURES))

        # Load weights
        # FIX: weights_only=True — secure loading
        state_dict = torch.load(
            MODEL_PATH,
            map_location='cpu',
            weights_only=True
        )
        model.load_state_dict(state_dict)

        # Set to eval mode — disables dropout during inference
        model.eval()

        console.print(
            f"[green][✓] Model loaded: {MODEL_PATH}[/green]"
        )
        console.print(
            f"[green][✓] Scaler loaded: {SCALER_PATH}[/green]"
        )
        return model, scaler

    except Exception as e:
        console.print(f"[red][✗] Load failed: {e}[/red]")
        sys.exit(1)


# ─────────────────────────────────────────────────────────────
#  FLOW CLEANUP — prevents memory leak
#
#  WHY THIS EXISTS:
#    Without cleanup, flow_stats grows forever — every unique
#    src:port-dst:port combination accumulates in RAM.
#    On a busy network with many connections, this eventually
#    consumes all available memory (memory leak).
#
#  FIX: Remove flows not seen for FLOW_TTL_SECONDS (120s).
#       Called every 100 packets from packet_handler().
# ─────────────────────────────────────────────────────────────
def purge_stale_flows():
    """
    Remove flow entries not seen for FLOW_TTL_SECONDS.
    Prevents unbounded memory growth.
    """
    now   = time.time()
    stale = [
        key for key, val in flow_stats.items()
        if now - val['last_seen'] > FLOW_TTL_SECONDS
    ]
    for key in stale:
        del flow_stats[key]


# ─────────────────────────────────────────────────────────────
#  FEATURE EXTRACTION
#
#  Converts a raw Scapy packet + flow statistics into the
#  38 NSL-KDD numeric features the model expects.
#
#  FIX (CRITICAL): Feature vector now exactly matches
#  NUMERIC_FEATURES order defined in ids_model.py.
#
#  Original code had:
#    - Wrong order (protocol_type injected at index 1)
#    - Missing features at indices 16, 17, 18
#    - Only 36-37 features instead of 38
#  This caused the model to receive completely wrong inputs
#  → all classifications were meaningless.
#
#  NUMERIC_FEATURES order (38 features):
#   0  duration                  19 count
#   1  src_bytes                 20 srv_count
#   2  dst_bytes                 21 serror_rate
#   3  land                      22 srv_serror_rate
#   4  wrong_fragment            23 rerror_rate
#   5  urgent                    24 srv_rerror_rate
#   6  hot                       25 same_srv_rate
#   7  num_failed_logins         26 diff_srv_rate
#   8  logged_in                 27 srv_diff_host_rate
#   9  num_compromised           28 dst_host_count
#  10  root_shell                29 dst_host_srv_count
#  11  su_attempted              30 dst_host_same_srv_rate
#  12  num_root                  31 dst_host_diff_srv_rate
#  13  num_file_creations        32 dst_host_same_src_port_rate
#  14  num_shells                33 dst_host_srv_diff_host_rate
#  15  num_access_files          34 dst_host_serror_rate
#  16  num_outbound_cmds         35 dst_host_srv_serror_rate
#  17  is_host_login             36 dst_host_rerror_rate
#  18  is_guest_login            37 dst_host_srv_rerror_rate
# ─────────────────────────────────────────────────────────────
def extract_features(packet, flow_key: str) -> np.ndarray:
    """
    Extract 38 NSL-KDD features from a packet + flow state.

    Args:
        packet  : Scapy packet object
        flow_key: String identifying this flow (src:port-dst:port)

    Returns:
        features: np.ndarray shape (38,) dtype float32
                  Aligned exactly to NUMERIC_FEATURES order
    """
    flow = flow_stats[flow_key]

    # ── Update flow counters ───────────────────────────────────
    flow['count']    += 1
    flow['last_seen'] = time.time()   # FIX: update TTL timestamp

    # FIX: src_bytes = IP payload only (not full Ethernet frame)
    # Original used len(packet) which included Ethernet+IP+TCP headers
    # causing inflated src_bytes → false positives
    if IP in packet:
        flow['src_bytes'] += len(packet[IP].payload)

    # ── TCP flag analysis ─────────────────────────────────────
    if TCP in packet:
        flags = packet[TCP].flags
        if flags & 0x02: flow['syn_count'] += 1   # SYN
        if flags & 0x01: flow['fin_count'] += 1   # FIN
        if flags & 0x04: flow['rst_count'] += 1   # RST

    # ── Compute derived values ─────────────────────────────────
    duration  = time.time() - flow['start_time']
    count     = max(flow['count'], 1)             # avoid div by zero
    src_bytes = float(flow['src_bytes'])
    dst_bytes = float(flow['dst_bytes'])
    syn_count = flow['syn_count']
    fin_count = flow['fin_count']
    rst_count = flow['rst_count']

    # Derived rate features
    serror_rate   = rst_count / count   # SYN error rate
    rerror_rate   = fin_count / count   # REJ error rate
    same_srv_rate = syn_count / count   # same service rate
    diff_srv_rate = 0.0                 # not trackable per-packet
    srv_diff_rate = syn_count / count   # service diff host rate

    # ── Build feature vector in exact NUMERIC_FEATURES order ──
    # FIX: This order MUST match NUMERIC_FEATURES in ids_model.py
    # Any mismatch = wrong features fed to model = garbage output
    features = np.array([
        duration,                          #  0  duration
        src_bytes,                         #  1  src_bytes
        dst_bytes,                         #  2  dst_bytes
        0.0,                               #  3  land
        0.0,                               #  4  wrong_fragment
        0.0,                               #  5  urgent
        0.0,                               #  6  hot
        0.0,                               #  7  num_failed_logins
        0.0,                               #  8  logged_in
        0.0,                               #  9  num_compromised
        0.0,                               # 10  root_shell
        0.0,                               # 11  su_attempted
        0.0,                               # 12  num_root
        0.0,                               # 13  num_file_creations
        0.0,                               # 14  num_shells
        0.0,                               # 15  num_access_files
        0.0,                               # 16  num_outbound_cmds  ← FIX was missing
        0.0,                               # 17  is_host_login      ← FIX was missing
        0.0,                               # 18  is_guest_login     ← FIX was missing
        float(count),                      # 19  count
        float(syn_count),                  # 20  srv_count
        serror_rate,                       # 21  serror_rate
        serror_rate,                       # 22  srv_serror_rate
        rerror_rate,                       # 23  rerror_rate
        rerror_rate,                       # 24  srv_rerror_rate
        same_srv_rate,                     # 25  same_srv_rate
        diff_srv_rate,                     # 26  diff_srv_rate
        srv_diff_rate,                     # 27  srv_diff_host_rate
        float(min(count * 2, 255)),        # 28  dst_host_count
        float(min(syn_count * 2, 255)),    # 29  dst_host_srv_count
        same_srv_rate,                     # 30  dst_host_same_srv_rate
        diff_srv_rate,                     # 31  dst_host_diff_srv_rate
        same_srv_rate,                     # 32  dst_host_same_src_port_rate
        0.0,                               # 33  dst_host_srv_diff_host_rate
        serror_rate,                       # 34  dst_host_serror_rate
        serror_rate,                       # 35  dst_host_srv_serror_rate
        rerror_rate,                       # 36  dst_host_rerror_rate
        rerror_rate,                       # 37  dst_host_srv_rerror_rate
    ], dtype=np.float32)

    # Safety check — feature count must match model input
    assert len(features) == len(NUMERIC_FEATURES), (
        f"Feature count mismatch: "
        f"got {len(features)}, expected {len(NUMERIC_FEATURES)}"
    )

    return features


# ─────────────────────────────────────────────────────────────
#  CLASSIFICATION
#  Feed extracted features through the trained model.
# ─────────────────────────────────────────────────────────────
def classify(features: np.ndarray,
             model,
             scaler) -> float:
    """
    Classify a single packet's features as Normal or Attack.

    FIX: Uses IDSNet directly with squeeze(-1) via model forward.
         Single-sample input (batch_size=1) is safe because
         ids_model.py uses squeeze(-1) not squeeze().

    Args:
        features: Raw feature array shape (38,)
        model   : Loaded IDSNet
        scaler  : Fitted StandardScaler

    Returns:
        prob: Attack probability [0.0 → 1.0]
              >= 0.5 = Attack, < 0.5 = Normal
    """
    # Reshape to (1, 38) — scaler expects 2D array
    features_2d = features.reshape(1, -1)

    # Scale using same scaler as training
    try:
        features_scaled = scaler.transform(features_2d)
    except Exception:
        # If scaler fails, use raw features
        features_scaled = features_2d

    # Run through model — no gradient needed for inference
    with torch.no_grad():
        tensor = torch.tensor(
            features_scaled, dtype=torch.float32
        )
        # model() returns shape (1, 1) → .item() gives scalar
        prob = model(tensor).item()

    return float(prob)


# ─────────────────────────────────────────────────────────────
#  ALERT SYSTEM
# ─────────────────────────────────────────────────────────────
def raise_alert(packet, flow_key: str, prob: float):
    """
    Raise an intrusion alert for a detected attack flow.

    Updates global state (for dashboard) and writes to log file.

    Args:
        packet  : Scapy packet that triggered the alert
        flow_key: Flow identifier string
        prob    : Attack probability from model
    """
    global recent_alerts

    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Extract IP/port info safely
    src_ip   = packet[IP].src if IP in packet else 'unknown'
    dst_ip   = packet[IP].dst if IP in packet else 'unknown'
    protocol = ('TCP' if TCP in packet else
                'UDP' if UDP in packet else 'ICMP')

    src_port = (packet[TCP].sport if TCP in packet else
                packet[UDP].sport if UDP in packet else 0)
    dst_port = (packet[TCP].dport if TCP in packet else
                packet[UDP].dport if UDP in packet else 0)

    severity = 'HIGH' if prob > 0.8 else 'MEDIUM'

    # Build alert record
    alert = {
        'time'    : timestamp,
        'src'     : f"{src_ip}:{src_port}",
        'dst'     : f"{dst_ip}:{dst_port}",
        'protocol': protocol,
        'prob'    : prob,
        'severity': severity,
    }

    # Update shared state (thread-safe)
    with state_lock:
        stats['alert_count'] += 1
        recent_alerts.append(alert)

        # Keep only last MAX_ALERTS_DISPLAY alerts in memory
        if len(recent_alerts) > MAX_ALERTS_DISPLAY:
            recent_alerts = recent_alerts[-MAX_ALERTS_DISPLAY:]

    # Write to log file
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, 'a') as f:
        f.write(
            f"[{timestamp}] ALERT #{stats['alert_count']} | "
            f"{src_ip}:{src_port} → {dst_ip}:{dst_port} | "
            f"{protocol} | {prob*100:.1f}% | {severity}\n"
        )


# ─────────────────────────────────────────────────────────────
#  PACKET HANDLER
#  Called by Scapy for every captured packet.
# ─────────────────────────────────────────────────────────────
def packet_handler(packet, model, scaler):
    """
    Process a single captured packet.

    Called by Scapy's sniff() for every packet matching filter.
    Extracts features, classifies, updates stats, raises alerts.

    Args:
        packet: Scapy packet object
        model : IDSNet model
        scaler: StandardScaler
    """
    # Only process IP packets
    if IP not in packet:
        return

    # Build flow key: "src_ip:src_port-dst_ip:dst_port"
    src_ip   = packet[IP].src
    dst_ip   = packet[IP].dst
    src_port = (packet[TCP].sport if TCP in packet else
                packet[UDP].sport if UDP in packet else 0)
    dst_port = (packet[TCP].dport if TCP in packet else
                packet[UDP].dport if UDP in packet else 0)
    flow_key = f"{src_ip}:{src_port}-{dst_ip}:{dst_port}"

    # Extract 38 NSL-KDD features
    features = extract_features(packet, flow_key)

    # Classify packet
    prob = classify(features, model, scaler)

    # Update global stats (thread-safe)
    with state_lock:
        stats['packet_count'] += 1
        stats['last_prob']     = prob

        if prob >= THRESHOLD:
            stats['attack_count'] += 1
        else:
            stats['normal_count'] += 1

    # Raise alert if attack detected
    if prob >= THRESHOLD:
        raise_alert(packet, flow_key, prob)

    # FIX: Purge stale flows every 100 packets
    # Prevents flow_stats dict from growing unboundedly (memory leak)
    if stats['packet_count'] % 100 == 0:
        purge_stale_flows()


# ─────────────────────────────────────────────────────────────
#  RICH DASHBOARD — builds the terminal UI layout
# ─────────────────────────────────────────────────────────────
def build_dashboard() -> Layout:
    """
    Build the Rich terminal dashboard layout.

    Layout structure:
      ┌──────────────────────────────────────┐
      │              HEADER                  │
      ├───────────────────┬──────────────────┤
      │     LIVE STATS    │   RECENT ALERTS  │
      ├───────────────────┴──────────────────┤
      │               FOOTER                 │
      └──────────────────────────────────────┘

    Returns:
        Rich Layout object ready to render
    """
    layout = Layout()

    # Define top-level rows
    layout.split_column(
        Layout(name='header', size=7),
        Layout(name='body'),
        Layout(name='footer', size=3),
    )

    # Split body into two columns
    layout['body'].split_row(
        Layout(name='stats',  ratio=1),
        Layout(name='alerts', ratio=2),
    )

    # ── Header ─────────────────────────────────────────────────
    header_text = Text(justify='center')
    header_text.append(
        '\n  Adversarial IDS — Live Network Monitor\n',
        style='bold cyan'
    )
    header_text.append(
        '  Powered by IDSNet | CDAC ITISS\n',
        style='dim white'
    )
    header_text.append(
        f'  Model: Adversarially Hardened | '
        f'Threshold: {THRESHOLD} | '
        f'Interface: {INTERFACE or "Auto-detect"}\n',
        style='dim white'
    )
    layout['header'].update(
        Panel(header_text, style='cyan', box=box.DOUBLE_EDGE)
    )

    # ── Live Stats Panel ───────────────────────────────────────
    with state_lock:
        elapsed = int(time.time() - stats['start_time'])
        hours   = elapsed // 3600
        minutes = (elapsed % 3600) // 60
        seconds = elapsed % 60

        total   = max(stats['packet_count'], 1)
        attack_pct = stats['attack_count'] / total * 100
        normal_pct = stats['normal_count'] / total * 100

        stats_table = Table(
            box=box.SIMPLE,
            show_header=False,
            padding=(0, 1)
        )
        stats_table.add_column('Metric', style='cyan',    width=20)
        stats_table.add_column('Value',  style='white',   width=15)

        stats_table.add_row(
            'Runtime',
            f'{hours:02d}:{minutes:02d}:{seconds:02d}'
        )
        stats_table.add_row(
            'Total Packets',
            f"{stats['packet_count']:,}"
        )
        stats_table.add_row(
            'Normal',
            f"[green]{stats['normal_count']:,} "
            f"({normal_pct:.1f}%)[/green]"
        )
        stats_table.add_row(
            'Attack',
            f"[red]{stats['attack_count']:,} "
            f"({attack_pct:.1f}%)[/red]"
        )
        stats_table.add_row(
            'Total Alerts',
            f"[bold red]{stats['alert_count']}[/bold red]"
        )
        stats_table.add_row(
            'Last Prob',
            f"[yellow]{stats['last_prob']*100:.1f}%[/yellow]"
        )
        stats_table.add_row(
            'Active Flows',
            f"{len(flow_stats):,}"
        )

    layout['stats'].update(
        Panel(
            stats_table,
            title='[bold cyan] Live Statistics [/bold cyan]',
            style='cyan',
            box=box.ROUNDED
        )
    )

    # ── Recent Alerts Panel ────────────────────────────────────
    alerts_table = Table(
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style='bold red',
        expand=True
    )
    alerts_table.add_column('Time',     style='dim white',  width=19)
    alerts_table.add_column('Source',   style='yellow',     width=21)
    alerts_table.add_column('Target',   style='yellow',     width=21)
    alerts_table.add_column('Proto',    style='cyan',       width=6)
    alerts_table.add_column('Conf',     style='red',        width=7)
    alerts_table.add_column('Severity', style='bold',       width=8)

    with state_lock:
        # Show most recent alerts first
        for alert in reversed(recent_alerts):
            severity_style = (
                'bold red' if alert['severity'] == 'HIGH'
                else 'bold yellow'
            )
            alerts_table.add_row(
                alert['time'],
                alert['src'],
                alert['dst'],
                alert['protocol'],
                f"{alert['prob']*100:.1f}%",
                f"[{severity_style}]{alert['severity']}[/{severity_style}]"
            )

    if not recent_alerts:
        alerts_table.add_row(
            '—', '—', '—', '—', '—',
            '[green]No alerts[/green]'
        )

    layout['alerts'].update(
        Panel(
            alerts_table,
            title='[bold red] Recent Alerts [/bold red]',
            style='red',
            box=box.ROUNDED
        )
    )

    # ── Footer ─────────────────────────────────────────────────
    layout['footer'].update(
        Panel(
            Text(
                '  Press Ctrl+C to stop monitoring  |  '
                f'Log file: {LOG_FILE}',
                justify='center',
                style='dim white'
            ),
            style='dim',
            box=box.SIMPLE
        )
    )

    return layout


# ─────────────────────────────────────────────────────────────
#  MAIN — entry point
# ─────────────────────────────────────────────────────────────
def main():
    """
    Start the live IDS with Rich terminal dashboard.

    1. Load model + scaler
    2. Start Scapy packet capture in background thread
    3. Render Rich dashboard in main thread (updates every second)
    4. On Ctrl+C: stop capture, print session summary
    """

    # ── Load model ────────────────────────────────────────────
    model, scaler = load_model()

    console.print(
        f"\n[green][✓] Live IDS Starting...[/green]"
    )
    console.print(
        f"[cyan]    Capturing on: "
        f"{INTERFACE or 'Auto-detect'}[/cyan]"
    )
    console.print(
        f"[cyan]    Log file    : {LOG_FILE}[/cyan]"
    )
    console.print(
        f"[yellow]    Press Ctrl+C to stop[/yellow]\n"
    )

    # Small delay so user can read startup messages
    time.sleep(1)

    # ── Start packet capture in background thread ─────────────
    # Runs sniff() in a daemon thread so it stops when main exits
    def capture_thread():
        sniff(
            iface=INTERFACE,
            prn=lambda pkt: packet_handler(pkt, model, scaler),
            store=False,         # do not store packets in memory
            filter='ip',         # only IP packets
        )

    thread = threading.Thread(target=capture_thread, daemon=True)
    thread.start()

    # ── Rich Live Dashboard ───────────────────────────────────
    # Refreshes every second showing live stats and alerts
    try:
        with Live(
            build_dashboard(),
            refresh_per_second=1,
            screen=True
        ) as live:
            while True:
                time.sleep(1)
                live.update(build_dashboard())

    except KeyboardInterrupt:
        # ── Session summary on Ctrl+C ─────────────────────────
        console.print(
            f"\n\n[yellow][!] IDS stopped by user[/yellow]"
        )
        console.print(f"\n[cyan]Session Summary:[/cyan]")
        console.print(
            f"  Total packets analysed : "
            f"[white]{stats['packet_count']:,}[/white]"
        )
        console.print(
            f"  Normal packets         : "
            f"[green]{stats['normal_count']:,}[/green]"
        )
        console.print(
            f"  Attack packets         : "
            f"[red]{stats['attack_count']:,}[/red]"
        )
        console.print(
            f"  Total alerts raised    : "
            f"[bold red]{stats['alert_count']}[/bold red]"
        )
        console.print(
            f"  Active flows at stop   : "
            f"[white]{len(flow_stats):,}[/white]"
        )
        console.print(
            f"  Alert log saved to     : "
            f"[cyan]{LOG_FILE}[/cyan]"
        )
        console.print(
            f"\n[green][✓] Goodbye![/green]\n"
        )


# ── Entry point ───────────────────────────────────────────────
if __name__ == '__main__':
    main()