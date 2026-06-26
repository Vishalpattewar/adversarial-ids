# =============================================================
# src/visualise_results.py
# =============================================================
# PURPOSE:
#   Reads JSON results saved by run_experiment.py and generates
#   3 publication-ready plots saved to results/plots/
#
# PLOTS GENERATED:
#   1. evasion_curves.png
#      → Evasion rate vs epsilon for FGSM and PGD attacks
#      → Standard model (red) vs Adversarial model (green)
#      → Shows WHERE adversarial training helps
#
#   2. defence_gain.png
#      → Bar chart of evasion reduction per epsilon value
#      → defence_gain = standard_evasion% - adversarial_evasion%
#      → Positive bars = adversarial training helped
#
#   3. architecture.png
#      → Visual diagram of IDSNet neural network architecture
#      → Shows layer sizes and flow: Input → ... → Output
#
# HOW TO RUN:
#   cd ~/A-IDS/adversarial_ids/src
#   python3 visualise_results.py
#
# PRE-REQUISITE:
#   run_experiment.py must be run first to generate JSON files.
#
# BUG FIXED VS ORIGINAL:
#   Y-axis double multiplication bug:
#     WRONG: data stored as % (e.g. 99.8) then formatter
#            multiplied again → displayed as 9980%
#     FIX:   formatter uses y directly → displays 99.8%
#
# CDAC ITISS — Adversarial Input Attack on ML-based IDS
# =============================================================

import os
import sys
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch

# Use non-interactive backend — works in WSL without display
matplotlib.use('Agg')


# ─────────────────────────────────────────────────────────────
#  PATH CONFIGURATION
# ─────────────────────────────────────────────────────────────
RESULTS_DIR = '../results'
PLOTS_DIR   = '../results/plots'


# ─────────────────────────────────────────────────────────────
#  COLOUR SCHEME — dark theme matching project style
# ─────────────────────────────────────────────────────────────
COLORS = {
    'std_fgsm' : '#E74C3C',   # red    — standard model FGSM
    'std_pgd'  : '#C0392B',   # dark red — standard model PGD
    'adv_fgsm' : '#2ECC71',   # green  — adversarial model FGSM
    'adv_pgd'  : '#27AE60',   # dark green — adversarial model PGD
    'bg'       : '#0F1117',   # background
    'grid'     : '#2A2D3A',   # grid lines
    'text'     : '#ECF0F1',   # text
    'fill'     : '#3498DB',   # fill between curves (FGSM)
    'fill_pgd' : '#9B59B6',   # fill between curves (PGD)
}


# ─────────────────────────────────────────────────────────────
#  HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────
def load_json(path: str) -> list:
    """
    Load a JSON results file.

    Args:
        path: Full path to JSON file

    Returns:
        Parsed JSON content (list of dicts)
    """
    if not os.path.exists(path):
        print(f"  ERROR: File not found: {path}")
        print(f"  Run run_experiment.py first.")
        sys.exit(1)

    with open(path) as f:
        return json.load(f)


def style_axis(ax):
    """
    Apply consistent dark theme styling to a matplotlib axis.

    Args:
        ax: matplotlib Axes object to style
    """
    ax.set_facecolor(COLORS['bg'])

    # Style all 4 border lines
    for spine in ax.spines.values():
        spine.set_color(COLORS['grid'])

    # Style tick marks and labels
    ax.tick_params(colors=COLORS['text'], which='both')

    # Style axis labels
    ax.xaxis.label.set_color(COLORS['text'])
    ax.yaxis.label.set_color(COLORS['text'])

    # Style title
    ax.title.set_color(COLORS['text'])

    # Style grid
    ax.grid(True, color=COLORS['grid'], linestyle='--', alpha=0.5)


# ─────────────────────────────────────────────────────────────
#  PLOT 1 — Evasion Curves
#
#  Shows evasion rate (%) vs perturbation budget (ε).
#  Left panel  : FGSM attack
#  Right panel : PGD attack
#  Each panel  : Standard model vs Adversarial model
#
#  The GAP between red and green lines = defence gain.
#  Larger gap = adversarial training helped more.
# ─────────────────────────────────────────────────────────────
def plot_evasion_curves():
    """Generate evasion rate vs epsilon plot for FGSM and PGD."""

    # Load all 4 result files
    std_fgsm = load_json(f'{RESULTS_DIR}/standard/attack_results_fgsm.json')
    std_pgd  = load_json(f'{RESULTS_DIR}/standard/attack_results_pgd.json')
    adv_fgsm = load_json(f'{RESULTS_DIR}/adversarial/attack_results_fgsm.json')
    adv_pgd  = load_json(f'{RESULTS_DIR}/adversarial/attack_results_pgd.json')

    # Extract epsilon values (x-axis)
    epsilons = [r['epsilon'] for r in std_fgsm]

    # Extract evasion rates
    # NOTE: evasion_rate stored as % (e.g. 99.8) in JSON
    #       FIX: do NOT multiply by 100 again — original bug
    std_fgsm_evasion = [r['evasion_rate'] for r in std_fgsm]
    std_pgd_evasion  = [r['evasion_rate'] for r in std_pgd]
    adv_fgsm_evasion = [r['evasion_rate'] for r in adv_fgsm]
    adv_pgd_evasion  = [r['evasion_rate'] for r in adv_pgd]

    # Create figure with 2 side-by-side panels
    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        figsize=(14, 5.5),
        facecolor=COLORS['bg']
    )

    # ── Left panel: FGSM ──────────────────────────────────────
    style_axis(ax1)

    ax1.plot(
        epsilons, std_fgsm_evasion,
        'o-',
        color=COLORS['std_fgsm'],
        lw=2.5, ms=7,
        label='Standard IDS (No Defence)'
    )
    ax1.plot(
        epsilons, adv_fgsm_evasion,
        's-',
        color=COLORS['adv_fgsm'],
        lw=2.5, ms=7,
        label='Adversarially Hardened IDS'
    )

    # Shade gap between models — visualises defence benefit
    ax1.fill_between(
        epsilons,
        std_fgsm_evasion,
        adv_fgsm_evasion,
        alpha=0.15,
        color=COLORS['fill']
    )

    ax1.set_title('FGSM Attack — Evasion Rate vs ε',
                  fontsize=13, pad=10)
    ax1.set_xlabel('Perturbation Budget ε', fontsize=11)
    ax1.set_ylabel('Evasion Rate (%)', fontsize=11)

    # FIX: formatter uses y directly — data is already in %
    # WRONG was: lambda y, _: f'{y*100:.0f}%' → showed 9980%
    ax1.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda y, _: f'{y:.1f}%')
    )
    ax1.legend(
        facecolor='#1A1D2E',
        labelcolor=COLORS['text'],
        edgecolor=COLORS['grid'],
        fontsize=10
    )

    # ── Right panel: PGD ──────────────────────────────────────
    style_axis(ax2)

    ax2.plot(
        epsilons, std_pgd_evasion,
        'o-',
        color=COLORS['std_pgd'],
        lw=2.5, ms=7,
        label='Standard IDS (No Defence)'
    )
    ax2.plot(
        epsilons, adv_pgd_evasion,
        's-',
        color=COLORS['adv_pgd'],
        lw=2.5, ms=7,
        label='Adversarially Hardened IDS'
    )
    ax2.fill_between(
        epsilons,
        std_pgd_evasion,
        adv_pgd_evasion,
        alpha=0.15,
        color=COLORS['fill_pgd']
    )

    ax2.set_title('PGD Attack (10 steps) — Evasion Rate vs ε',
                  fontsize=13, pad=10)
    ax2.set_xlabel('Perturbation Budget ε', fontsize=11)
    ax2.set_ylabel('Evasion Rate (%)', fontsize=11)

    # FIX: formatter uses y directly — data is already in %
    ax2.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda y, _: f'{y:.1f}%')
    )
    ax2.legend(
        facecolor='#1A1D2E',
        labelcolor=COLORS['text'],
        edgecolor=COLORS['grid'],
        fontsize=10
    )

    # Overall figure title
    fig.suptitle(
        'Adversarial Attack Evasion Rate: Standard vs Adversarially Trained IDS\n'
        'CDAC ITISS — Adversarial Input Attack on ML-based IDS',
        color=COLORS['text'],
        fontsize=12,
        y=1.02
    )

    plt.tight_layout()
    output_path = f'{PLOTS_DIR}/evasion_curves.png'
    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches='tight',
        facecolor=COLORS['bg']
    )
    plt.close()
    print(f"  ✓ Saved: {output_path}")


# ─────────────────────────────────────────────────────────────
#  PLOT 2 — Defence Gain Bar Chart
#
#  Shows how much adversarial training reduced evasion rate
#  at each epsilon value.
#
#  defence_gain = standard_evasion% - adversarial_evasion%
#  Positive = adversarial training helped (good)
#  Negative = adversarial training hurt   (bad — epsilon mismatch)
# ─────────────────────────────────────────────────────────────
def plot_defence_gain():
    """Generate defence gain bar chart."""

    # Load summary report
    summary_path = f'{RESULTS_DIR}/summary_report.json'
    if not os.path.exists(summary_path):
        print(f"  ERROR: {summary_path} not found.")
        print(f"  Run run_experiment.py first.")
        sys.exit(1)

    with open(summary_path) as f:
        summary = json.load(f)

    data = summary['evasion_comparison']

    # Extract values for plotting
    epsilons   = [d['epsilon'] for d in data]
    gains_fgsm = [d['defence_gain_fgsm_pct'] for d in data]
    gains_pgd  = [
        d['standard_pgd_evasion_pct'] - d['adversarial_pgd_evasion_pct']
        for d in data
    ]

    # Create figure
    fig, ax = plt.subplots(figsize=(11, 5), facecolor=COLORS['bg'])
    style_axis(ax)

    # Bar positions
    x     = np.arange(len(epsilons))
    width = 0.35

    # Draw bars
    bars_fgsm = ax.bar(
        x - width / 2, gains_fgsm, width,
        label='FGSM Defence Gain',
        color='#3498DB', alpha=0.9
    )
    bars_pgd = ax.bar(
        x + width / 2, gains_pgd, width,
        label='PGD Defence Gain',
        color='#9B59B6', alpha=0.9
    )

    # X-axis labels
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f'ε={e}' for e in epsilons],
        color=COLORS['text']
    )

    ax.set_ylabel('Evasion Rate Reduction (%)',
                  color=COLORS['text'], fontsize=11)
    ax.set_title(
        'Defence Gain from Adversarial Training\n'
        '(Percentage Points of Evasion Reduction)',
        color=COLORS['text'], fontsize=12, pad=10
    )
    ax.legend(
        facecolor='#1A1D2E',
        labelcolor=COLORS['text'],
        edgecolor=COLORS['grid']
    )

    # Annotate each bar with its value
    for bar in ax.patches:
        h = bar.get_height()
        if abs(h) > 0.05:    # skip near-zero bars
            ax.annotate(
                f'{h:.1f}%',
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 4),
                textcoords='offset points',
                ha='center', va='bottom',
                color=COLORS['text'],
                fontsize=8
            )

    # Draw horizontal line at y=0 for reference
    ax.axhline(y=0, color=COLORS['text'], linewidth=0.8, alpha=0.5)

    plt.tight_layout()
    output_path = f'{PLOTS_DIR}/defence_gain.png'
    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches='tight',
        facecolor=COLORS['bg']
    )
    plt.close()
    print(f"  ✓ Saved: {output_path}")


# ─────────────────────────────────────────────────────────────
#  PLOT 3 — Architecture Diagram
#
#  Visual diagram of the IDSNet neural network.
#  Shows each layer as a labelled box with arrows between them.
# ─────────────────────────────────────────────────────────────
def plot_architecture():
    """Generate IDSNet architecture diagram."""

    fig, ax = plt.subplots(figsize=(13, 4), facecolor='#0A0E1A')
    ax.set_facecolor('#0A0E1A')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.5)
    ax.axis('off')

    # Define each layer box:
    # (label text, x-centre position, fill colour)
    layers = [
        ('Input\n38 features',         0.8,  '#2980B9'),
        ('BatchNorm\nDense(256)\nReLU', 2.3,  '#8E44AD'),
        ('Dense(128)\nReLU\nDropout',   4.0,  '#8E44AD'),
        ('Dense(64)\nReLU',             5.7,  '#8E44AD'),
        ('Dense(1)\nSigmoid',           7.4,  '#E74C3C'),
        ('Output\nNormal / Attack',     9.0,  '#27AE60'),
    ]

    # Draw each box
    for label, x, color in layers:
        box = FancyBboxPatch(
            (x - 0.6, 0.8), 1.2, 1.8,
            boxstyle='round,pad=0.1',
            facecolor=color,
            edgecolor='white',
            linewidth=1.5,
            alpha=0.88
        )
        ax.add_patch(box)
        ax.text(
            x, 1.72, label,
            ha='center', va='center',
            color='white', fontsize=8.5,
            fontweight='bold',
            multialignment='center'
        )

    # Draw arrows between boxes
    x_positions = [l[1] for l in layers]
    for i in range(len(x_positions) - 1):
        ax.annotate(
            '',
            xy     =(x_positions[i + 1] - 0.61, 1.7),
            xytext =(x_positions[i]     + 0.61, 1.7),
            arrowprops=dict(
                arrowstyle='->',
                color='#ECF0F1',
                lw=1.8
            )
        )

    # Title and subtitle
    ax.text(
        5, 3.15,
        'IDSNet — Neural Network Architecture',
        ha='center', va='center',
        color='white', fontsize=13, fontweight='bold'
    )
    ax.text(
        5, 0.35,
        'CDAC ITISS: Adversarial Input Attack on ML-based IDS',
        ha='center', va='center',
        color='#BDC3C7', fontsize=9
    )

    output_path = f'{PLOTS_DIR}/architecture.png'
    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches='tight',
        facecolor='#0A0E1A'
    )
    plt.close()
    print(f"  ✓ Saved: {output_path}")


# ─────────────────────────────────────────────────────────────
#  MAIN — run all 3 plots
# ─────────────────────────────────────────────────────────────
def main():
    """Generate all result visualisation plots."""

    # Ensure output directory exists
    os.makedirs(PLOTS_DIR, exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  Generating Result Visualisations")
    print(f"{'='*55}")
    print(f"  Output directory: {PLOTS_DIR}\n")

    try:
        plot_evasion_curves()
        plot_defence_gain()
        plot_architecture()

        print(f"\n  All 3 plots generated successfully.")
        print(f"  View them:")
        print(f"    explorer.exe ../results/plots/")
        print(f"{'='*55}\n")

    except SystemExit:
        # sys.exit() called inside plot functions (missing JSON)
        raise

    except Exception as e:
        print(f"\n  ERROR generating plots: {e}")
        print(f"  Make sure run_experiment.py ran successfully first.")
        raise


# ── Entry point ───────────────────────────────────────────────
if __name__ == '__main__':
    main()