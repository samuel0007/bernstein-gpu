#!/usr/bin/env python3
"""
Compare 4 Jacobian types: line plot of total kernel time vs DOFs (one fig per p).
Keeps original stacked-area code intact if you still need it.
"""

# -------- CONFIG --------
LOG_PATTERNS = ["*.txt"]
OUT_PREFIX   = "out"
LEVEL        = 9
EXCLUDE      = {"CG AXPY 1", "CG AXPY 2", "CG REDUCTION 1", "CG REDUCTION 2"}
# EXCLUDE      = {"C"}    
MAX_N        = None
X_LOG        = False
Y_LOG        = False
SAVE_PDF     = True
DPI          = 300
FONT_SIZE    = 16
LINE_W       = 1.8
MARKER_SIZE  = 6
PALETTE_HEX  = ["#E88B7A", "#F6F0DB", "#9FC7C3", "#8A6C7D", "#253E48",
                "#5E81AC", "#A3BE8C", "#BF616A"]  # extra just in case
MARKERS      = ['o','s','^','D','v','P','X','>','<','h','*','+','x']
# -------------------------------------

JAC_RENAME = {
    "1": "Baseline",
    "2": "Streams",
    "3": "Opti-Fused",
    "4": "Streams+Opti-Fused"
}

import glob, os, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch


GPU_RE = re.compile(r"\[GPU\]\[(\d+)\]\s+(.+?):\s+([\d.]+)\s+ms")
N_RE   = re.compile(r"ndofs_global\s*=\s*(\d+)", re.IGNORECASE)
F_RE   = re.compile(r"log_([^_]+)_(\d+)_(\d+)\.(?:txt|log)$")  # jac, N, p

def parse_file(path):
    m = F_RE.search(os.path.basename(path))
    if not m:  # fallback to old style if present
        jac = None
        N = p = None
    else:
        jac, N, p = m.group(1), int(m.group(2)), int(m.group(3))

    txt = open(path, "r", errors="ignore").read()
    Nm = N_RE.search(txt)
    N  = int(Nm.group(1)) if Nm else None
    rows = []
    for lvl, tag, ms in GPU_RE.findall(txt):
        rows.append((os.path.basename(path), jac, N, p, int(lvl), tag.strip(), float(ms)))
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
    df = pd.DataFrame(rows, columns=["file","jac","N","p","level","tag","ms"])
    df["jac"] = df["jac"].astype(str).map(JAC_RENAME).fillna(df["jac"])
    df = df[(df.level == LEVEL) & df.N.notna() & df.p.notna() & df.jac.notna()]
    if EXCLUDE: df = df[~df.tag.isin(EXCLUDE)]
    if MAX_N is not None: df = df[(df.N <= MAX_N) & (df.N > 1000)]
    if df.empty: raise SystemExit("No data after filters")

    # Mean per (jac,N,p,tag)
    g = df.groupby(["jac","p","N","tag"], as_index=False)["ms"].mean()
    g.to_csv(f"{OUT_PREFIX}_means_by_tag.csv", index=False)

    # Total per (jac,N,p)
    tot = g.groupby(["jac","p","N"], as_index=False)["ms"].sum().rename(columns={"ms":"ms_total"})
    tot.to_csv(f"{OUT_PREFIX}_totals.csv", index=False)

    # ---- Plot: one figure per p, lines = jac types ----
    for pdeg, sub in tot.groupby("p"):
        pv = sub.pivot(index="N", columns="jac", values="ms_total").sort_index()
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        for i, jac in enumerate(pv.columns):
            y = pv[jac].to_numpy()
            x = pv.index.values
            col = mcolors.to_rgb(PALETTE_HEX[i % len(PALETTE_HEX)])
            ax.plot(x, y, label=jac,
                    linewidth=LINE_W,
                    marker=MARKERS[i % len(MARKERS)],
                    markersize=MARKER_SIZE,
                    linestyle='-',
                    markeredgecolor="black",
                    color=col)

        ax.set_xlabel(r"DOFs")
        ax.set_ylabel("Mean time [ms]")
        if X_LOG: ax.set_xscale("log")
        if Y_LOG: ax.set_yscale("log")
        ax.legend(loc="upper left", frameon=True, ncol=2)
        fig.tight_layout()

        base = f"{OUT_PREFIX}_p{pdeg}_jac_compare"
        fig.savefig(base + ".png")
        if SAVE_PDF: fig.savefig(base + ".pdf")
        plt.close(fig)
        print("saved", base + ".png")

    # ---- Single figure: p=2 and p=4 together ----
    SELECT_P = {4, 6}
    sub = tot[tot.p.isin(SELECT_P)].copy()
    if sub.empty:
        raise SystemExit("No rows for p=2/4")

    # Sort for consistent drawing order
    sub = sub.sort_values(["jac", "p", "N"])

    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    p_styles = {6: "-", 4: "--"}          # solid for p2, dashed for p4
    p_markers = {6: "o", 4: "s"}           # optional: different markers
    jac_list = sorted(sub.jac.unique())

    for i, jac in enumerate(jac_list):
        col = mcolors.to_rgb(PALETTE_HEX[i % len(PALETTE_HEX)])
        for pdeg in sorted(SELECT_P):
            d = sub[(sub.jac == jac) & (sub.p == pdeg)]
            if d.empty: continue
            ax.plot(d.N.to_numpy(), d.ms_total.to_numpy(),
                    label=f"{jac} (p{pdeg})",
                    color=col,
                    linestyle=p_styles[pdeg],
                    marker=p_markers[pdeg],
                    markersize=MARKER_SIZE,
                    linewidth=LINE_W,
                    markeredgecolor="black")

    ax.set_xlabel(r"DOFs")
    ax.set_ylabel("Mean time [ms]")
    if X_LOG: ax.set_xscale("log")
    if Y_LOG: ax.set_yscale("log")
    ax.legend(loc="upper left", frameon=True, ncol=2)
    fig.tight_layout()

    base = f"{OUT_PREFIX}_p2_p4_jac_compare"
    fig.savefig(base + ".png")
    if SAVE_PDF: fig.savefig(base + ".pdf")
    plt.close(fig)
    print("saved", base + ".png")
    
    
    
    BASE_JAC   = "Baseline"
    SELECT_JAC = {"Opti-Fused"}

    base = tot[tot.jac == BASE_JAC][["p","N","ms_total"]].rename(columns={"ms_total":"ms_base"})
    spd  = tot.merge(base, on=["p","N"], how="inner")
    spd["speedup"] = spd["ms_base"] / spd["ms_total"]
    spd = spd[spd.jac.isin(SELECT_JAC)]

    fig, ax = plt.subplots(figsize=(8, 6))

    p_list    = sorted(spd.p.unique())
    p_styles  = {p: ("-","--","-.",":")[i % 4] for i,p in enumerate(p_list)}
    p_markers = {p: MARKERS[i % len(MARKERS)]   for i,p in enumerate(p_list)}

    for j, pdeg in enumerate(p_list):
        d = spd[spd.p == pdeg].sort_values("N")
        if d.empty: continue
        col = mcolors.to_rgb(PALETTE_HEX[j % len(PALETTE_HEX)])
        ax.plot(d.N.to_numpy(), d.speedup.to_numpy(),
                label=f"p{pdeg}",
                color=col,
                linestyle=p_styles[pdeg],
                marker=p_markers[pdeg],
                markersize=MARKER_SIZE,
                linewidth=LINE_W,
                markeredgecolor="black")

    ax.axhline(1.0, color="black", linewidth=1, linestyle=":", zorder=0)
    ax.set_xlabel(r"DOFs")
    ax.set_ylabel(f"Speedup vs. {BASE_JAC}")
    if X_LOG: ax.set_xscale("log")
    ax.legend(loc="best", frameon=True, ncol=2)
    fig.tight_layout()

    base_out = f"{OUT_PREFIX}_speedup_streams"
    fig.savefig(base_out + ".png")
    if SAVE_PDF: fig.savefig(base_out + ".pdf")
    plt.close(fig)
    print("saved", base_out + ".png")
    
    # -------- CONFIG --------
    JAC_ORDER   = ["Baseline", "Opti-Fused", "Streams+Opti-Fused", "Streams"]
    J_STACK     = {"Baseline", "Opti-Fused"}          # stacked
    J_SPLIT     = {"Streams+Opti-Fused", "Streams"}   # side-by-side
    SELECT_P    = {2, 3, 4, 5, 6, 7, 8}                         # or all: set(g.p.unique())
    # pick one N per p (largest); or set SELECT_N manually
    # -----------------------------------------------

    def pick_rows(df, pdeg):
        # choose largest N for that p
        Nmax = df[df.p == pdeg].N.max()
        return df[(df.p == pdeg) & (df.N == Nmax)], Nmax

    for pdeg in sorted(SELECT_P):
        sub, Nmax = pick_rows(g, pdeg)
        if sub.empty: 
            continue

        # ensure jac order
        sub = sub[sub.jac.isin(JAC_ORDER)]
        # operators sorted by baseline total
        base_ops_order = (sub[sub.jac == "Baseline"]
                        .groupby("tag")["ms"].sum()
                        .sort_values(ascending=False).index.tolist())
        # include any extra ops from others
        for t in sub.tag.unique():
            if t not in base_ops_order: base_ops_order.append(t)

        # color per operator
        cmap = plt.get_cmap("tab20")
        op2col = {op: cmap(i % 20) for i, op in enumerate(base_ops_order)}
        
        # PALETTE_HEX = ["#E88B7A", "#F6F0DB", "#9FC7C3", "#8A6C7D", "#253E48", "#F2C14E"]  # extend if needed

        ops = base_ops_order  # list of tags in desired order
        op2col = {op: PALETTE_HEX[i % len(PALETTE_HEX)] for i, op in enumerate(ops)}

        # pre-agg to one value per (jac, tag)
        agg = (sub.groupby(["jac","tag"], as_index=False)["ms"].mean())

        # x positions
        x = np.arange(len(JAC_ORDER), dtype=float)
        width = 0.6
        fig, ax = plt.subplots(figsize=(8, 4.5))

        # draw bars
        bottoms = {j: 0.0 for j in JAC_ORDER}   # for stacked
        split_counts = {j: 0 for j in J_SPLIT}  # side-by-side offset index

        for op in base_ops_order:
            for j_i, jac in enumerate(JAC_ORDER):
                val = agg[(agg.jac == jac) & (agg.tag == op)]["ms"]
                if val.empty:
                    continue
                y = float(val.values[0])

                if jac in J_STACK:
                    ax.bar(x[j_i], y, width=width, bottom=bottoms[jac],
                        color=op2col[op], edgecolor='black', label=op if bottoms[jac]==0 else None, linewidth=0.3)
                    bottoms[jac] += y
                else:  # split
                    nops = len(sub[sub.jac == jac].tag.unique())
                    idx  = split_counts[jac]
                    w    = width / nops
                    x0   = x[j_i] - width/2 + idx*w + w/2
                    ax.bar(x0, y, width=w*0.9, bottom=0.0,
                        color=op2col[op], edgecolor='black', linewidth=0.3,
                        label=None)  # legend handled separately
                    split_counts[jac] += 1

        # cosmetics
        ax.set_xticks(x)
        ax.set_xticklabels(JAC_ORDER, rotation=0)
        ax.set_ylabel("Mean time [ms]")
        ax.set_title(f"p{pdeg} @ {Nmax:,} DOFs")

        # legend for operators (one entry each)
        handles = [Patch(facecolor=op2col[o], edgecolor='none', label=o) for o in base_ops_order]
        ax.legend(handles=handles, loc="best", frameon=True, ncol=2)

        fig.tight_layout()
        base = f"{OUT_PREFIX}_p{pdeg}_jac_bar_mix"
        fig.savefig(base + ".png", dpi=DPI)
        if SAVE_PDF: fig.savefig(base + ".pdf")
        plt.close(fig)
        print("saved", base + ".png")






if __name__ == "__main__":
    main()
