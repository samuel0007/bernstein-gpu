#!/usr/bin/env python3
"""
Parse logs named:  log_{block_size}_{cells_per_block}_{p}.txt
Extract Gdofs/s, build:
  1) Heatmap (block_size × cells_per_block) per p (raw Gdofs/s or speedup).
  2) Speedup bar plots vs best and vs a chosen baseline per p.

Adjust LOG_PATTERNS / GDOFS_RE if your line differs.
"""

import re, glob
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
LOG_PATTERNS = ["experiments/**/*/log_*.txt", "experiments/log_*.txt", "log_stiff*.txt"]
OUT_DIR      = "plots_bs_cpb_gdfos"
OUT_CSV      = "bs_cpb_gdfos_all.csv"

# Baseline config per p (block_size, cells_per_block); leave {} to auto-pick first entry
BASELINE_CFG = {2:(24,1), 3:(45,1), 4:(74, 1), 5:(122, 1), 6:(177, 1), 7:(729, 1), 8:(1000, 1)}  # e.g. {2:(64,1), 4:(128,1)}

# Heatmap uses raw Gdofs/s (True) or speedup vs best (False -> raw; True -> speedup)
USE_SPEEDUP_IN_HEATMAP = False

PALETTE_HEX = ["#E88B7A", "#F6F0DB", "#9FC7C3", "#8A6C7D", "#253E48"]

# Regexes
F_RE      = re.compile(r"log_stiff_(\d+)_(\d+)_(\d+)\.txt$")
P_RE      = re.compile(r"Polynomial degree:\s*(\d+)", re.IGNORECASE)
BLK_RE    = re.compile(r"Block size:\s*(\d+)", re.IGNORECASE)
CPB_RE    = re.compile(r"Cells per block:\s*(\d+)", re.IGNORECASE)
GDOFS_RE  = re.compile(r"Gdofs/s:\s*([\d.eE+-]+)", re.IGNORECASE)  # e.g. "SF Mat-free action Gdofs/s: 0.996124"

def _int_first(rx, s):
    m = rx.search(s)
    return int(m.group(1)) if m else None

def _float_first(rx, s):
    m = rx.search(s)
    return float(m.group(1)) if m else None

def parse_file(path):
    text = Path(path).read_text(errors="ignore")

    m = F_RE.search(Path(path).name)
    blk_f = int(m.group(1)) if m else None
    cpb_f = int(m.group(2)) if m else None
    p_f   = int(m.group(3)) if m else None

    p   = p_f   or _int_first(P_RE, text)
    blk = blk_f or _int_first(BLK_RE, text)
    cpb = cpb_f or _int_first(CPB_RE, text)
    gdf = _float_first(GDOFS_RE, text)

    if p is None or blk is None or cpb is None or gdf is None:
        return None

    return dict(file=path, p=p, block_size=blk, cells_per_block=cpb, gdfos=gdf)

def load_all():
    files = []
    for pat in LOG_PATTERNS:
        files.extend(glob.glob(pat, recursive=True))
    files = sorted(set(files))
    print("Found", len(files), "log files.")
    rows = [parse_file(f) for f in files]
    print("Parsed", len(rows), "log entries.")
    rows = [r for r in rows if r is not None]
    if not rows:
        raise SystemExit("No logs parsed.")
    return pd.DataFrame(rows)

def pick_baseline(df_p, p):
    if p in BASELINE_CFG:
        bs, cpb = BASELINE_CFG[p]
        row = df_p[(df_p.block_size == bs) & (df_p.cells_per_block == cpb)]
        if not row.empty:
            r = row.iloc[0]
            return r.gdfos, (bs, cpb)
    r = df_p.iloc[0]
    return r.gdfos, (int(r.block_size), int(r.cells_per_block))

def plot_heatmap(df_p, p, metric, fname):
    pv = df_p.pivot(index="cells_per_block", columns="block_size", values=metric)
    pv = pv.sort_index().sort_index(axis=1)

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    im = ax.imshow(pv.to_numpy(), origin="lower", aspect="auto", cmap="viridis")

    ax.set_xticks(range(len(pv.columns)))
    ax.set_xticklabels(pv.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(pv.index)))
    ax.set_yticklabels(pv.index)
    ax.set_xlabel("Stride (blockDim.x)")
    ax.set_ylabel("Cells per block (blockDim.y)")
    # ax.set_title(f"P={p}  {metric}")

    cb = fig.colorbar(im, ax=ax)
    cb.ax.set_ylabel("GDOFS/s" if metric == "gdfos" else "Speedup", rotation=270, labelpad=15)

    vals = pv.to_numpy()
    meanv = np.nanmean(vals)
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            v = vals[i, j]
            if np.isnan(v):
                continue
            txt = f"{v:.3f}" if metric == "gdfos" else f"{v:.2f}"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=8, color="white" if v < meanv else "black")

    fig.tight_layout()
    fig.savefig(fname, dpi=300)
    plt.close(fig)
    print("saved", fname)

def plot_speedup(df_p, p, metric, fname):
    sub = df_p.sort_values(metric, ascending=False)
    labels = [f"{b}x{c}" for b, c in zip(sub.block_size, sub.cells_per_block)]
    vals = sub[metric].to_numpy()

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.bar(labels, vals, color=PALETTE_HEX[0], edgecolor="black", linewidth=0.6)
    ax.axhline(1.0, color="black", ls=":", lw=0.9)
    ax.set_ylabel(metric)
    ax.set_xlabel("block_size x cells_per_block")
    ax.set_title(f"P={p}  {metric}")
    ax.set_ylim(0, max(vals)*1.1)
    ax.tick_params(axis='x', rotation=45)
    fig.tight_layout()
    fig.savefig(fname, dpi=300)
    plt.close(fig)
    print("saved", fname)

def main():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 13,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 9,
        "figure.dpi": 120
    })

    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    df = load_all()
    df.to_csv(OUT_CSV, index=False)

    out_all = []
    for p, sub in df.groupby("p"):
        sub = sub.copy()

        # Best = max gdfos
        best = sub.gdfos.max()
        sub["speedup_best"] = sub.gdfos / best

        # Baseline
        base_gdfos, base_cfg = pick_baseline(sub, p)
        sub["speedup_base"] = sub.gdfos / base_gdfos

        # Heatmap
        metric = "gdfos" if not USE_SPEEDUP_IN_HEATMAP else "speedup_best"
        plot_heatmap(sub, p, metric, Path(OUT_DIR) / f"heatmap_p{p}.png")

        # Speedup plots
        plot_speedup(sub, p, "speedup_best", Path(OUT_DIR) / f"speedup_best_p{p}.png")
        plot_speedup(sub, p, "speedup_base", Path(OUT_DIR) / f"speedup_base_p{p}.png")

        out_all.append(sub)

    pd.concat(out_all, ignore_index=True).to_csv(OUT_CSV, index=False)
    print("CSV ->", OUT_CSV)
    
    all_df = pd.concat(out_all, ignore_index=True)

    # per p: pick row with maximal speedup vs baseline
    grp = all_df.groupby("p", as_index=False).apply(
        lambda d: d.loc[d["speedup_base"].idxmax(), ["p","block_size","cells_per_block","speedup_base"]]
    ).reset_index(drop=True)

    # plot
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.plot(grp.p, grp.speedup_base, marker='o', linestyle='-', color=PALETTE_HEX[0], lw=2)
    ax.axhline(1.0, color='black', ls=':', lw=0.9)

    # annotate config "bxcpb"
    for _, r in grp.iterrows():
        lbl = f"{int(r.block_size)}x{int(r.cells_per_block)}"
        ax.text(r.p, r.speedup_base + 0.1, lbl, ha='center', va='bottom', fontsize=11, color="black")

    ax.set_xlabel("P")
    ax.set_ylabel("Max speedup vs baseline")
    # ax.set_title("Best config per P")
    ax.set_ylim(0, grp.speedup_base.max()*1.15)

    ax.grid(True, ls=":", lw=0.6, alpha=0.6)
    fig.tight_layout()
    fig.savefig(Path(OUT_DIR)/"max_speedup_vs_baseline.png", dpi=300)
    plt.close(fig)
    print("saved", Path(OUT_DIR)/"max_speedup_vs_baseline.png")
    
    
    best_rows   = []
    baseline_rows = []

    for p, sub in all_df.groupby("p"):
        # best (max gdfos)
        best_rows.append(sub.loc[sub.gdfos.idxmax()])

        # baseline: the one picked earlier in pick_baseline()
        base_gdf, (bs, cpb) = pick_baseline(sub, p)
        bl_row = sub[(sub.block_size == bs) & (sub.cells_per_block == cpb)].iloc[0]
        baseline_rows.append(bl_row)

    best_df   = pd.DataFrame(best_rows)
    base_df   = pd.DataFrame(baseline_rows)

    fig, ax = plt.subplots(figsize=(6.5, 4.0))

    ax.plot(base_df.p, base_df.gdfos,
            marker='s', linestyle='-', color=PALETTE_HEX[4],
            lw=2, label="Baseline")

    ax.plot(best_df.p, best_df.gdfos,
            marker='o', linestyle='-', color=PALETTE_HEX[0],
            lw=2, label="Best (tuned)")

    # annotate best with "blockxcells"
    for _, r in best_df.iterrows():
        lbl = f"{int(r.block_size)}x{int(r.cells_per_block)}\nx{r.speedup_base:.2f}"
        ax.text(r.p, r.gdfos + 0.07, lbl, ha='center', va='bottom',
                fontsize=10, color="black", bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2', linewidth=0.6))
      
    ax.set_xlabel("P")
    ax.set_ylabel("GDOF/s")
    ax.set_ylim(0, 1.1*best_df.gdfos.max())
    ax.grid(True, ls=":", lw=0.6, alpha=0.6)
    ax.legend(frameon=True)

    fig.tight_layout()
    fig.savefig(Path(OUT_DIR)/"gdfos_vs_p.png", dpi=300)
    plt.close(fig)
    print("saved", Path(OUT_DIR)/"gdfos_vs_p.png")


if __name__ == "__main__":
    main()
