#!/usr/bin/env python3
"""
Parse all ptxas logs, build a DF, derive FEM sizes + resource use,
compute occupancy limits (regs/shared/threads/hw) and plot:
  * CSV with all kernels
  * One figure per entry function: regs & shared per block, occupancy curves
Aesthetics use your PALETTE_HEX.
"""

import re, glob
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
LOG_PATTERNS   = ["*_lb_*.txt"]
OUT_CSV        = "ptxas_kernels_lb.csv"
OUT_DIR_PLOTS  = "ptxas_plots_lb"

PALETTE_HEX    = ["#E88B7A", "#F6F0DB", "#9FC7C3", "#8A6C7D", "#253E48"]
COL_ALL        = "#000000"   # actual occupancy (black)
COL_REGS       = PALETTE_HEX[0]
COL_SHARED     = PALETTE_HEX[4]
COL_THREADS    = PALETTE_HEX[2]
COL_HW         = PALETTE_HEX[3]

# RTX 6000 Ada caps
REGS_PER_SM        = 65536
SHARED_PER_SM      = 48 * 1024
MAX_THREADS_PER_SM = 2048
MAX_BLOCKS_PER_SM  = 32
WARP_SIZE          = 32
# ----------------------------------------

# Regexes
RE_COMPILE   = re.compile(r"Compiling entry function '([^']+)' for 'sm_(\d+)'")
RE_FUNC_HDR  = re.compile(r"Function properties for\s+(\S+)")
RE_PROPS     = re.compile(r"Used\s+(\d+)\s+registers,\s+used\s+(\d+)\s+barriers,\s+(\d+)\s+bytes\s+smem")
RE_CMEM_ALL  = re.compile(r"(\d+)\s+bytes\s+cmem\[(\d+)\]")
RE_STACK     = re.compile(r"(\d+)\s+bytes stack frame,\s+(\d+)\s+bytes spill stores,\s+(\d+)\s+bytes spill loads")
RE_CTIME     = re.compile(r"Compile time\s*=\s*([\d.]+)\s*ms")
RE_LAST_NUM  = re.compile(r"(\d+)(?!.*\d)")  # last number in filename stem

def collect_files(patterns):
    files = []
    for p in patterns: files.extend(glob.glob(p, recursive=True))
    return [f for f in sorted(set(files)) if Path(f).is_file()]

def flush(cur, out):
    if not cur: return
    for idx, val in cur["cmem"].items():
        cur[f"cmem[{idx}]"] = val
    cur.pop("cmem", None)
    out.append(cur)

def parse_logs(files):
    recs, cur = [], None
    for path in files:
        with open(path, "r", errors="ignore") as f:
            for line in f:
                m = RE_COMPILE.search(line)
                if m:
                    flush(cur, recs)
                    cur = dict(file=path, kernel=m.group(1), sm_arch=int(m.group(2)),
                               regs=None, barriers=None, smem_bytes=None,
                               stack_bytes=None, spill_stores=None, spill_loads=None,
                               compile_time_ms=None, cmem=defaultdict(int))
                    continue
                if cur is None:
                    mfh = RE_FUNC_HDR.search(line)
                    if mfh:
                        flush(cur, recs)
                        cur = dict(file=path, kernel=mfh.group(1), sm_arch=None,
                                   regs=None, barriers=None, smem_bytes=None,
                                   stack_bytes=None, spill_stores=None, spill_loads=None,
                                   compile_time_ms=None, cmem=defaultdict(int))
                    else:
                        continue
                if (m := RE_PROPS.search(line)):
                    cur["regs"]       = int(m.group(1))
                    cur["barriers"]   = int(m.group(2))
                    cur["smem_bytes"] = int(m.group(3))
                    for b, idx in RE_CMEM_ALL.findall(line):
                        cur["cmem"][int(idx)] = int(b)
                    continue
                if (m := RE_STACK.search(line)):
                    cur["stack_bytes"]  = int(m.group(1))
                    cur["spill_stores"] = int(m.group(2))
                    cur["spill_loads"]  = int(m.group(3))
                    continue
                if (m := RE_CTIME.search(line)):
                    cur["compile_time_ms"] = float(m.group(1))
                    continue
                if (m := RE_FUNC_HDR.search(line)):
                    cur["kernel"] = m.group(1)
    flush(cur, recs)
    return recs

def last_number_in_stem(path):
    m = RE_LAST_NUM.search(Path(path).stem)
    return int(m.group(1)) if m else None

def kernel_base(name):
    i = name.find('<'); j = name.find('IfL')
    cut = min([x for x in (i, j) if x != -1], default=len(name))
    return name[:cut]

def occ_from_blocks(blocks, threads_per_block):
    return (blocks * threads_per_block) / MAX_THREADS_PER_SM

def main():
    # matplotlib defaults
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 13,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 9,
        "figure.dpi": 120
    })

    Path(OUT_DIR_PLOTS).mkdir(parents=True, exist_ok=True)
    files = collect_files(LOG_PATTERNS)
    if not files: raise SystemExit("No log files found.")

    df = pd.DataFrame(parse_logs(files))

    # Drop huge cub kernels
    df = df[~df["kernel"].str.contains("cub17CUB")]

    # FEM sizes
    df["P"] = df["file"].map(last_number_in_stem)
    cmem_cols = [c for c in df.columns if c.startswith("cmem[")]
    df["cmem_total"] = df[cmem_cols].sum(axis=1) if cmem_cols else 0

    df["Q"] = df["P"] + 2
    df["N"]  = df["P"] + 1
    df["nd"] = df["N"]*(df["N"]+1)*(df["N"]+2)//6
    df["nq"] = (
        14*(df["Q"]==3)  + 24*(df["Q"]==4)  + 45*(df["Q"]==5)  + 74*(df["Q"]==6) +
        122*(df["Q"]==7) + 177*(df["Q"]==8) + 729*(df["Q"]==9) + 1000*(df["Q"]==10) +
        1331*(df["Q"]==11)
    )

    df["block_size"]        = df["nq"]
    df["shared_total"]      = df["smem_bytes"].fillna(0)
    df["threads_per_block"] = df["block_size"]
    df["warps_per_block"]   = (df["threads_per_block"] + WARP_SIZE - 1)//WARP_SIZE
    df["regs_per_block"]    = df["regs"] * df["threads_per_block"]

    # blocks limited by each resource
    # avoid /0
    safe = lambda a,b: np.where(b>0, a//b, 0)
    df["b_regs"]    = safe(REGS_PER_SM,        df["regs_per_block"])
    df["b_shared"]  = safe(SHARED_PER_SM,      df["shared_total"])
    df["b_threads"] = safe(MAX_THREADS_PER_SM, df["threads_per_block"])
    df["b_hw"]      = MAX_BLOCKS_PER_SM

    # occupancy per resource
    df["occ_regs"]    = occ_from_blocks(df["b_regs"],    df["threads_per_block"])
    df["occ_shared"]  = occ_from_blocks(df["b_shared"],  df["threads_per_block"])
    df["occ_threads"] = occ_from_blocks(df["b_threads"], df["threads_per_block"])
    df["occ_hw"]      = occ_from_blocks(df["b_hw"],      df["threads_per_block"])

    df["b_active"] = df[["b_regs","b_shared","b_threads","b_hw"]].min(axis=1)
    df["occ"]      = occ_from_blocks(df["b_active"], df["threads_per_block"])

    df.to_csv(OUT_CSV, index=False)

    df["entry_base"] = df["kernel"].map(kernel_base)

    for entry, sub in df.dropna(subset=["P"]).groupby("entry_base"):
        sub = sub.sort_values("P")

        fig, axes = plt.subplots(1, 1, figsize=(6, 4), sharex=True)

        # --- Regs / block ---
        # ax0 = axes[0]
        # ax0.plot(sub.P, sub.regs_per_block, 'o-', color=COL_REGS, linewidth=1.8)
        # ax0.axhline(REGS_PER_SM, ls='--', color='gray', lw=0.8)
        # ax0.set_ylabel("Regs / block")

        # --- Shared / block ---
        # ax1 = axes[1]
        # ax1.plot(sub.P, sub.shared_total, 's-', color=COL_SHARED, linewidth=1.8)
        # ax1.axhline(SHARED_PER_SM, ls='--', color='gray', lw=0.8)
        # ax1.set_ylabel("Shared / block [B]")

        # --- Occupancy curves ---
        ax2 = axes
        ax2.axhline(100, color='black', lw=1.0, ls='-', alpha=0.7, label="100%")
        ax2.plot(sub.P, sub.occ*100,        '-',  color=COL_ALL,    lw=2.2, label="min(all)")
        ax2.plot(sub.P, sub.occ_regs*100,   '--',  color=COL_REGS,   lw=1.6, label="regs")
        ax2.plot(sub.P, sub.occ_shared*100, '--',  color=COL_SHARED, lw=1.6, label="shared")
        ax2.plot(sub.P, sub.occ_threads*100,'--',  color=COL_THREADS,lw=1.6, label="threads")
        ax2.plot(sub.P, sub.occ_hw*100,     '--',  color=COL_HW,     lw=1.6, label="hw blocks")

        occ_vals = sub.occ * 100
        ax2.fill_between(sub.P, 0, occ_vals,
                 color=COL_ALL, alpha=0.15, zorder=1)
        ax2.set_ylabel("Occupancy [%]")
        ax2.set_xlabel("P")
        ax2.set_ylim(0, 210)
        ax2.legend(loc="best", frameon=True, ncol=1)

        # for ax in axes:
        axes.grid(True, ls=":", lw=0.6, alpha=0.6)

        # fig.suptitle(entry, fontsize=11)
        fig.tight_layout()  # make space for legend
        out = Path(OUT_DIR_PLOTS) / f"{entry.replace('/','_')}_vs_P.png"
        fig.savefig(out, dpi=300)
        plt.close(fig)
        print("saved", out)

    print(f"CSV -> {OUT_CSV}")

if __name__ == "__main__":
    main()
