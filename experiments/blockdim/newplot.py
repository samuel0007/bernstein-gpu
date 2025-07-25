#!/usr/bin/env python3
"""
Compare three families of logs:

    log_stiff_<bs>_<cpb>_<p>.txt
    log_mass_<bs>_<cpb>_<p>.txt
    log_<bs>_<cpb>_<p>.txt   (fused)

For each polynomial degree P:
   • best fused GDOF/s
   • best stiff+mass pair (harmonic combination)
   • stiff+mass baseline from BASELINE_CFG

Outputs a CSV and a line plot with three curves.
"""

import re, glob
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------ CONFIG
LOG_PATTERNS = ["experiments/**/*/log*.txt", "log*.txt"]
OUT_CSV      = "compare_fused_vs_split_gdfos.csv"
OUT_PLOT     = "gdfos_fused_vs_split.png"

COL_SPLIT  = "#E88B7A"
COL_FUSED  = "#253E48"
COL_BASE   = "grey"

BASELINE_CFG = {2:(24,1), 3:(45,1), 4:(74,1), 5:(122,1),
                6:(177,1), 7:(729,1), 8:(1000,1)}
# ---------------------------------------------------------

re_fused = re.compile(r"log_(\d+)_(\d+)_(\d+)\.txt$")
re_stiff = re.compile(r"log_stiff_(\d+)_(\d+)_(\d+)\.txt$")
re_mass  = re.compile(r"log_mass_(\d+)_(\d+)_(\d+)\.txt$")
g_re     = re.compile(r"Gdofs/s:\s*([\d.eE+-]+)", re.I)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 13,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 9,
    "figure.dpi": 120
})

# ---------------- helpers ----------------
def grab(path: str):
    name = Path(path).name
    if   (m := re_fused.match(name)): kind = "fused"
    elif (m := re_stiff.match(name)): kind = "stiff"
    elif (m := re_mass .match(name)): kind = "mass"
    else: return None
    bs, cpb, p = map(int, m.groups())
    txt = Path(path).read_text(errors="ignore")
    mt  = g_re.search(txt)
    if not mt:
        return None
    g = float(mt.group(1))
    return dict(kind=kind, p=p, block_size=bs, cells_per_block=cpb, g=g)

# ---------------- load logs --------------
records = []
for pat in LOG_PATTERNS:
    records.extend(filter(None, (grab(f) for f in glob.glob(pat, recursive=True))))
if not records:
    raise SystemExit("No logs parsed")

df = pd.DataFrame(records)
df.to_csv(OUT_CSV, index=False)

# ------------ best fused -----------------
best_fused = (df[df.kind == "fused"]
              .sort_values("g", ascending=False)
              .groupby("p", as_index=False)
              .first())        # row with max g per p

# ------------ best stiff+mass ------------
best_split_rows = []
for p, grp in df.groupby("p"):
    stiff = grp[grp.kind == "stiff"]
    mass  = grp[grp.kind == "mass"]
    if stiff.empty or mass.empty:
        continue
    best_g = 0.0
    for _, rs in stiff.iterrows():
        for _, rm in mass.iterrows():
            g_pair = 1.0 / (1.0 / rs.g + 1.0 / rm.g)   # harmonic mean
            if g_pair > best_g:
                best_g = g_pair
    best_split_rows.append(dict(p=p, g_split=best_g))
best_split = pd.DataFrame(best_split_rows)

# ------------ baseline stiff+mass --------
base_rows = []
for p, (bs, cpb) in BASELINE_CFG.items():
    s = df[(df.kind == "stiff") & (df.p == p) &
           (df.block_size == bs) & (df.cells_per_block == cpb)]
    m = df[(df.kind == "mass")  & (df.p == p) &
           (df.block_size == bs) & (df.cells_per_block == cpb)]
    if s.empty or m.empty:
        continue
    g_pair = 1.0 / (1.0 / s.iloc[0].g + 1.0 / m.iloc[0].g)
    base_rows.append(dict(p=p, g_base=g_pair))
baseline = pd.DataFrame(base_rows)

# ------------ merge & speedup ------------
merged = (best_split
          .merge(best_fused[["p", "g"]].rename(columns={"g": "g_fused"}), on="p")
          .merge(baseline, on="p", how="left"))
merged["speedup_vs_base"] = merged.g_fused / merged.g_base
merged.to_csv(OUT_CSV, index=False)
print("CSV ->", OUT_CSV)

# ---------------- plot -------------------
fig, ax = plt.subplots(figsize=(7, 4))

ax.plot(merged.p, merged.g_base,  "-o", color=COL_BASE,  lw=2, label="Stiff+Mass (baseline)")
ax.plot(merged.p, merged.g_split, "-o", color=COL_SPLIT, lw=2, label="Stiff+Mass (tuned)")
ax.plot(merged.p, merged.g_fused, "-o", color=COL_FUSED, lw=2, label="Fused (tuned)")

# annotate fused with speed‑up
for _, r in merged.iterrows():
    ax.text(r.p, r.g_fused + 0.08,
            f"×{r.speedup_vs_base:.2f}",
            ha="center", va="bottom",
            fontsize=10, color=COL_FUSED,  bbox=dict(facecolor="white", edgecolor=COL_FUSED, boxstyle="round,pad=0.2", linewidth=0.6))

ax.set_xlabel("Polynomial degree P")
ax.set_ylabel("GDOF/s")
ax.grid(True, ls=":", lw=0.6, alpha=0.6)
ax.legend(frameon=True)
fig.tight_layout()
fig.savefig(OUT_PLOT, dpi=300)
plt.close(fig)
print("saved", OUT_PLOT)
