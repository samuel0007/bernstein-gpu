#!/usr/bin/env python3
# Compare 2D amplitude fields: load, (optionally) resample, split-view plot, relative L2 error.

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

# --- Loading helpers ---
def _pick_2d_array(dct, prefer=("p_amp","amplitude_map")):
    cand = []
    for k,v in dct.items():
        if k.startswith("__"): continue
        a = np.asarray(v)
        if a.ndim != 2:
            # a = a.reshape(251, 251).T * 2.199 # BP1, BP2, BP4
            a = a.reshape(251, 251).T * 2.1999 # BP1, BP2, BP4
            # a = a.reshape(251, 251).T * 1.8 # BP3
            
            # a = a.reshape(251, 251).T
        cand.append((k,a))
    # prefer known keys
    print(dct.items())
    for name in prefer:
        for k,a in cand:
            if k == name: return a
    # else pick largest 2D array
    if cand:
        return max(cand, key=lambda kv: kv[1].size)[1]
    raise ValueError("No 2D arrays found.")

def load_mat_2d(path, prefer=()):
    # Try SciPy
    try:
        from scipy.io import loadmat
        d = loadmat(path, simplify_cells=True)
        return _pick_2d_array(d, prefer or ("p_amp","amplitude_map","amplitude","p_rms","p_max","u","p"))
    except NotImplementedError:
        pass  # v7.3 HDF5
    # Fallback: h5py (v7.3)
    import h5py
    def collect(ds_map, g, prefix=""):
        for k,v in g.items():
            name = f"{prefix}{k}"
            if isinstance(v, h5py.Dataset):
                ds_map[name] = v
            elif isinstance(v, h5py.Group):
                collect(ds_map, v, name+"/")
    with h5py.File(path, "r") as f:
        ds_map = {}
        collect(ds_map, f)
        # build dict of arrays (2D only)
        arrays = {}
        for name, ds in ds_map.items():
            a = np.array(ds)[...].squeeze()
            if a.ndim == 2 and np.isfinite(a).any():
                arrays[name] = a
        if not arrays:
            raise ValueError("No 2D datasets in HDF5 .mat.")
        # prefer keys
        for pref in (prefer or ("p_amp","amplitude_map","amplitude","p_rms","p_max","u","p")):
            if pref in arrays: return arrays[pref]
            # allow suffix match (e.g., '/p_amp')
            for k in arrays:
                if k.endswith("/"+pref): return arrays[k]
        # else largest
        return arrays[max(arrays, key=lambda k: arrays[k].size)]


# --- Paths ---
parser = argparse.ArgumentParser(description="Compare validation and simulation outputs.")
parser.add_argument("--validation-folder", type=Path, required=True,
                    help="Path to KWAVE benchmark data")
parser.add_argument("--comparison-folder", type=Path, required=True,
                    help="Path to FEniCSx output data")
parser.add_argument("--source", type=int, default=2,
                    help="Source index (default: 2)")
parser.add_argument("--bm", type=int, default=2,
                    help="Benchmark number (default: 2)")
args = parser.parse_args()
VALIDATION_FOLDER = args.validation_folder
COMPARISON_FOLDER = args.comparison_folder
SOURCE = args.source
BM = args.bm

validation_file = f"{VALIDATION_FOLDER}/PH1-BM{BM}-SC{SOURCE}_KWAVE.mat"
comparison_file = f"{COMPARISON_FOLDER}/amplitude.mat"

plt.rcParams.update({
    "font.family": "serif", "font.size": 10, "axes.labelsize": 10,
    "axes.titlesize": 11, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "lines.linewidth": 1.2, "axes.linewidth": 0.8, "xtick.direction": "in",
    "ytick.direction": "in", "figure.dpi": 300,
})
   

# --- Mask: upward spherical cap touching bottom ---
# def upward_cap_mask(shape, cx, a, Rcurv, Lcap):
#     H, W = shape
#     Y, X = np.ogrid[:H, :W]
#     a, R, L = float(a), float(Rcurv), float(Lcap)
#     s = R - np.sqrt(max(R**2 - a**2, 0.0))
#     cy = R
#     r = np.abs(X - cx)
#     y_surf = cy - np.sqrt(np.clip(R**2 - r**2, 0.0, None))
#     y_cap = np.minimum(y_surf, L)
#     mask_sph = (r <= a) & (Y >= 0) & (Y <= y_cap)
#     mask_cyl = (r <= a) & (Y > s) & (Y <= L)
#     return mask_sph | mask_cyl

# --- Load fields ---
val = load_mat_2d(validation_file)
cmp_ = load_mat_2d(comparison_file)


# Optional resample
if cmp_.shape != val.shape:
    from scipy.ndimage import zoom
    scale = (val.shape[0] / cmp_.shape[0], val.shape[1] / cmp_.shape[1])
    cmp_ = zoom(cmp_, scale, order=1)

# Transpose
val_t = val.T
cmp_t = cmp_.T


L_val = 0.07     # m
L_cmp = 0.064    # m

if L_val > L_cmp:
    nx_val = val_t.shape[1]
    remove_frac = (L_val - L_cmp) / L_val
    remove_cols = int(round(remove_frac * nx_val / 2))
    val_t = val_t[:, remove_cols:nx_val - remove_cols]
    # Resample comparison to match trimmed validation
    from scipy.ndimage import zoom
    scale_x = val_t.shape[1] / cmp_t.shape[1]
    cmp_t = zoom(cmp_t, (1, scale_x), order=1)

# mask where cmp_ is == 0
mask_t = ~np.isclose(cmp_t, 0, atol=1e-12)
# 5 bottom are also masked
mask_t[:5, :] = False  # Mask last 5 rows (bottom)
plt.imshow(mask_t, origin="lower", cmap="gray")
plt.savefig("mask.png")


# Mask
# mask_cap = ~upward_cap_mask(val_t.shape, cx=val_t.shape[1] // 2, a=90, Rcurv=132, Lcap=20)

# Relative L2 (global)
rel_l2_global = np.linalg.norm((cmp_ - val).ravel()) / np.linalg.norm(val.ravel())
print(f"Global relative L2 error: {rel_l2_global*100:.3f}%")

# --- Masked Global Relative L2 ---
val_masked = val_t[mask_t]
cmp_masked = cmp_t[mask_t]
num = np.linalg.norm(cmp_masked - val_masked)
den = np.linalg.norm(val_masked)
rel_l2_global = num / den if den != 0 else np.inf
print(f"Masked global relative L2 error (inside dome): {rel_l2_global*100:.3f}%")

# --- Save plots ---
out_dir = Path(COMPARISON_FOLDER)
out_dir.mkdir(parents=True, exist_ok=True)

def save_plot(data, title, fname, cmap="viridis", vmin=None, vmax=None):
    plt.figure(figsize=(7, 5))
    plt.imshow(data, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.colorbar(pad=0.02)
    plt.tight_layout()
    plt.savefig(out_dir / fname, dpi=300)
    plt.close()

# Difference (masked)
diff_masked = np.where(mask_t, cmp_t - val_t, np.nan)
lim = np.nanmax(np.abs(diff_masked))
save_plot(diff_masked, "Difference (masked, transposed)", f"BM{BM}_SC{SOURCE}_diff_masked.png",
          cmap="seismic", vmin=-lim, vmax=lim)

# Absolute L2 (masked)
abs_l2_masked = np.where(mask_t, np.abs(cmp_t - val_t), np.nan)
save_plot(abs_l2_masked, "Absolute L2 Error (masked, transposed)",
          f"BM{BM}_SC{SOURCE}_absL2_masked.png", cmap="viridis")

# Relative L2 (masked)
eps = 1e-12
rel_l2_masked = np.where(mask_t, np.abs(cmp_t - val_t) / (np.abs(val_t) + eps), np.nan)
save_plot(rel_l2_masked, "Relative L2 Error (masked, transposed)",
          f"BM{BM}_SC{SOURCE}_relL2_masked.png", cmap="viridis")

print(f"Plots saved to {out_dir}")


domain_length = 0.12  # y direction [m]
domain_width  = 0.064  # x direction [m]

ny, nx = val_t.shape
y = np.linspace(0, domain_length, ny)
x = np.linspace(-domain_width/2,  domain_width/2,  nx)


# Flip comparison vertically
cmp_flipped = np.flipud(cmp_t)
val_flipped = np.flipud(val_t)
mask_flipped = np.flipud(mask_t)

# Create split image
mid = nx // 2
split_img = np.zeros_like(val_t)
split_img[:, :mid] = val_flipped[:, :mid]
split_img[:, mid:] = cmp_flipped[:, mid:]
split_img[~mask_flipped] = 0  # Apply mask

# Color limits
vmin = min(np.nanmin(val_t), np.nanmin(cmp_t))
vmax = max(np.nanmax(val_t), np.nanmax(cmp_t))

# Plot
plt.figure(figsize=(7, 5))
plt.imshow(
    split_img,
    origin="lower",
    aspect="auto",
    vmin=vmin,
    vmax=vmax,
    cmap="turbo",
    extent=[x[0], x[-1], y[0], y[-1]]
)
plt.axvline(0, color="white", linestyle="--", linewidth=1)  # divider at x=0
plt.xlabel("x [m]")
plt.ylabel("z [m]")

# Symmetric ticks
plt.xticks(np.linspace(x[0], x[-1], 5))
plt.yticks(np.linspace(y[-1], y[0], 5))

plt.colorbar(pad=0.02, label="Amplitude")
plt.tight_layout()
plt.savefig(out_dir / f"BM{BM}_SC{SOURCE}_split_view.png", dpi=300)
plt.close()



# Vertical midline
mid_x = nx // 2
plt.figure(figsize=(6, 4))
plt.plot(y[:-2], val_t[:-2, mid_x], label="KWAVE")
plt.plot(y[:-2], cmp_t[:-2, mid_x], linestyle="-.", label="FreeFUS")
# plt.scatter(y[:-2], cmp_t[:-2, mid_x], marker="x", s=5, label="FreeFUS", color="black", zorder=3)
plt.xlabel("z [m]")
plt.ylabel("Amplitude")
plt.legend()
plt.tight_layout()
plt.savefig(out_dir / f"BM{BM}_SC{SOURCE}_midline_vertical_plot.png", dpi=300)
plt.close()

# Horizontal midline
mid_y = ny // 2
plt.figure(figsize=(6, 4))
plt.plot(x, val_t[mid_y, :], label="KWAVE")
plt.scatter(x, cmp_t[mid_y, :], marker="x", s=5, color="black", label="FreeFUS", zorder=3)
plt.xlabel("x [m]")
plt.ylabel("Amplitude")
plt.legend()
plt.tight_layout()
plt.savefig(out_dir / f"BM{BM}_SC{SOURCE}_midline_horizontal_plot.png", dpi=300)
plt.close()