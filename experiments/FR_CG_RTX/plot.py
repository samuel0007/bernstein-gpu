#!/usr/bin/env python3
"""
Stacked area (faint) + line/markers on top edge, one figure per p.
Config only (no CLI).
"""

# -------- CONFIG --------
LOG_PATTERNS = ["*.txt", "*.log"]
OUT_PREFIX   = "out"
LEVEL        = 3
EXCLUDE      = {""}      # set() for none
MAX_N        = 400_000            # or None
X_LOG        = False
Y_LOG        = False
SAVE_PDF     = True
DPI          = 300
FONT_SIZE    = 16
FILL_ALPHA   = 0.9              # area transparency
LINE_W       = 1.3
MARKER_SIZE  = 5
PALETTE_HEX  = ["#E88B7A", "#F6F0DB", "#9FC7C3", "#8A6C7D", "#253E48"]  # your colors
MARKERS      = ['o','s','^','D','v','P','X','>','<','h','*','+','x']   # cycle
# -------------------------------------

import glob, os, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

GPU_RE = re.compile(r"\[GPU\]\[(\d+)\]\s+(.+?):\s+([\d.]+)\s+ms")
N_RE   = re.compile(r"ndofs_global\s*=\s*(\d+)", re.IGNORECASE)
P_RE   = re.compile(r"_(\d+)\.(?:txt|log)$")

def poly_from_name(path):
    m = P_RE.search(os.path.basename(path))
    return int(m.group(1)) if m else None

def parse_file(path):
    txt = open(path, "r", errors="ignore").read()
    Nm = N_RE.search(txt)
    N  = int(Nm.group(1)) if Nm else None
    p  = poly_from_name(path)
    rows = []
    for lvl, tag, ms in GPU_RE.findall(txt):
        rows.append((os.path.basename(path), N, p, int(lvl), tag.strip(), float(ms)))
    return rows

def config_mpl():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE-2,
        "ytick.labelsize": FONT_SIZE-2,
        "legend.fontsize": FONT_SIZE-4,
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
    })

def main():
    config_mpl()

    files = [f for pat in LOG_PATTERNS for f in glob.glob(pat)]
    if not files: raise SystemExit("No files")

    rows = [r for f in files for r in parse_file(f)]
    df = pd.DataFrame(rows, columns=["file","N","p","level","tag","ms"])
    df = df[(df.level == LEVEL) & df.N.notna() & df.p.notna()]
    if EXCLUDE: df = df[~df.tag.isin(EXCLUDE)]
    if MAX_N is not None: df = df[(df.N <= MAX_N) & (df.N > 1000)]
    if df.empty: raise SystemExit("No data after filters")

    g = df.groupby(["p","N","tag"], as_index=False)["ms"].mean()
    g.to_csv(f"{OUT_PREFIX}_means.csv", index=False)

    for pdeg, sub in g.groupby("p"):
        pv = sub.pivot(index="N", columns="tag", values="ms").sort_index()
        pv = pv[pv.sum().sort_values(ascending=False).index]

        x = pv.index.values
        cum = np.zeros_like(x, dtype=float)

        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        handles, labels = [], []

        for i, tag in enumerate(pv.columns):
            y   = pv[tag].fillna(0.0).to_numpy()
            top = cum + y

            fill_col = mcolors.to_rgba(PALETTE_HEX[i % len(PALETTE_HEX)], FILL_ALPHA)
            line_col = mcolors.to_rgb(PALETTE_HEX[i % len(PALETTE_HEX)])

            poly = ax.fill_between(x, cum, top, color=fill_col, edgecolor="none", zorder=1)
            ax.plot(x, top, color=line_col, linewidth=LINE_W,
                    marker=MARKERS[i % len(MARKERS)], markersize=MARKER_SIZE,
                    linestyle='-', zorder=2, markeredgecolor="black")

            handles.append(poly); labels.append(tag)
            cum = top

        ax.set_xlabel(r"DOFs")
        ax.set_ylabel("Mean time [ms]")
        if X_LOG: ax.set_xscale("log")
        if Y_LOG: ax.set_yscale("log")

        ax.legend(handles, labels, loc="upper left", frameon=True)
        fig.tight_layout()

        base = f"{OUT_PREFIX}_p{pdeg}_stacked_area"
        fig.savefig(base + ".png")
        if SAVE_PDF: fig.savefig(base + ".pdf")
        plt.close(fig)
        print("saved", base + ".png")

if __name__ == "__main__":
    main()
