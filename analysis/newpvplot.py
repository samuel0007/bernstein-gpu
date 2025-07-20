import argparse
import os
from paraview.simple import ADIOS2VTXReader, servermanager, Slice, ResampleToImage, Transform
from vtk.util.numpy_support import vtk_to_numpy
print("imported paraview")
import numpy as np
print("imported numpy")
import matplotlib.pyplot as plt
import multiprocessing as mp
print("imported multiprocessing")
import time

# --- Configuration ---
n = 1000
field        = "u"              
axis         = "X"              # "X", "Y", or "Z"
value        = 0.0              # Slice location
dims         = [1, n, n]    # Sampling dimensions [nx, ny, nz] for the slice
f0           = 0.1e6            # Center frequency for FFT analysis (Hz)
# ---

print("number of points:", n)

# This function is designed to be run in a separate process. It remains unchanged.
def load_one_timestep(t, input_file, field_name, sb_config, dims_config):
    """
    Loads and processes a single timestep in an isolated process.
    This is critical because ParaView/VTK objects are not thread/process-safe.
    """
    from paraview.simple import ADIOS2VTXReader, Slice, ResampleToImage, servermanager
    from vtk.util.numpy_support import vtk_to_numpy

    # Each worker creates its own pipeline
    local_reader = ADIOS2VTXReader(FileName=[input_file])
    local_slicer = Slice(Input=local_reader)
    local_resampler = ResampleToImage(Input=local_slicer)
    local_resampler.SamplingBounds = sb_config
    local_resampler.SamplingDimensions = dims_config
    
    # Update the pipeline for this specific timestep
    local_resampler.UpdatePipeline(t)
    
    # Fetch the data from the server
    d = servermanager.Fetch(local_resampler)
    u = vtk_to_numpy(d.GetPointData().GetArray(field_name))
    
    print(f"  ...Worker finished loading timestep t={t:.6f}")
    return u

# This guard is CRUCIAL for multiprocessing to work correctly and safely.
if __name__ == "__main__":
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(
        description="Process a single ADIOS2 BP file to generate FFT amplitude plots."
    )
    parser.add_argument(
        "input_file", 
        type=str, 
        help="Path to the input .bp file"
    )
    parser.add_argument(
        "N", 
        type=int, 
        default=5,
        nargs="?",
        help="Number of timesteps to analyze",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=-1,
        help="Number of parallel jobs to use for data loading (-1 means use all available cores)",
    )
    args = parser.parse_args()

    # --- Setup Output Directory ---
    os.makedirs('output', exist_ok=True)

    # --- Matplotlib Setup ---
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "lines.linewidth": 1.2,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "figure.dpi": 300,
    })

    # --- Data Loading and Preparation ---
    # We only need a reader in the main process to get metadata (times, bounds)
    print("Loading metadata from:", args.input_file)
    meta_reader = ADIOS2VTXReader(FileName=[args.input_file])
    meta_reader.UpdatePipeline()
    print("Metadata loaded successfully.")

    # Use the last N timesteps for analysis
    if isinstance(meta_reader.TimestepValues, float):
        times = [meta_reader.TimestepValues]
    else:
        times = meta_reader.TimestepValues[-args.N:]
    print(f"Analyzing {len(times)} timesteps: {np.asarray(times)}")

    # Get data bounds and define sampling bounds for the slice
    s, b = Slice(Input=meta_reader), meta_reader.GetDataInformation().GetBounds()
    b = [0.0 if abs(val) < 1e-15 else val for val in b]
    print("Data Bounds:", b)

    if axis == "X":
        sb = [value, value, b[2], b[3], b[4], b[5]]
    elif axis == "Y":
        sb = [b[0], b[1], value, value, b[4], b[5]]
    else: # "Z"
        sb = [b[0], b[1], b[2], b[3], value, value]
    print("Sampling Bounds:", sb)

    # --- Parallel Data Loading using multiprocessing ---
    
    # Use 'spawn' to create clean worker processes. Highly recommended for complex libraries.
    mp.set_start_method('spawn', force=True)

    start_time = time.time()
    n_jobs = min(os.cpu_count(), len(times)) if args.jobs == -1 else args.jobs
    print(f"Loading data in parallel using {n_jobs} jobs...")

    # Prepare arguments for each task. starmap needs a list of tuples.
    task_args = [(t, args.input_file, field, sb, dims) for t in times]
    
    # Create a pool of worker processes and distribute the tasks
    with mp.Pool(processes=n_jobs) as pool:
        results_list = pool.starmap(load_one_timestep, task_args)
    
    # Combine the list of 1D arrays into a single 2D numpy array
    u_stack = np.vstack(results_list)
    
    end_time = time.time()
    print(f"Parallel data loading finished in {end_time - start_time:.2f} seconds.")

    # --- The rest of the script remains identical ---

    # Find the global min and max across all loaded timesteps
    vmin = u_stack.min()
    vmax = u_stack.max()
    print(f"Global data range determined: vmin={vmin:.4f}, vmax={vmax:.4f}")

    # --- Main Plotting Loop ---
    print("Generating timestep images with fixed color scale...")
    for i, t in enumerate(times):
        plt.imshow(u_stack[i].reshape(n, n), cmap='seismic', aspect='auto', vmin=vmin, vmax=vmax)
        plt.colorbar(label=f'Field "{field}"')
        plt.title(f'Timestep {i+1} (t={t:.6f})')
        plt.savefig(f'output/field_timestep_{i+1}.png')
        plt.close()

    # --- FFT Analysis ---
    nt, npts = u_stack.shape
    if len(times) > 1:
        dt = np.mean(np.diff(times))
    else:
        dt = 1.0
        print("Warning: Only one timestep. FFT results may not be meaningful.")
    fft_result = np.fft.rfft(u_stack, axis=0)
    freqs = np.fft.rfftfreq(nt, dt)
    idx = np.argmin(np.abs(freqs - f0))
    print(f"Target frequency f0={f0/1e6:.2f} MHz, closest FFT frequency={freqs[idx]/1e6:.2f} MHz")
    amp_fft = 2 * np.abs(fft_result[idx, :]) / nt
    amp = amp_fft.reshape(dims[2], dims[1])

    # --- Final Amplitude Plot ---
    fig, ax = plt.subplots(figsize=(6, 5))
    amp_flipped = np.flip(amp, axis=0)
    if axis == 'X':
        extent = [sb[2], sb[3], sb[4], sb[5]]
        xlabel, ylabel = 'y (m)', 'z (m)'
    elif axis == 'Y':
        extent = [sb[0], sb[1], sb[4], sb[5]]
        xlabel, ylabel = 'x (m)', 'z (m)'
    else: # 'Z'
        extent = [sb[0], sb[1], sb[2], sb[3]]
        xlabel, ylabel = 'x (m)', 'y (m)'
    im = ax.imshow(amp_flipped, origin='lower', aspect='auto', cmap='turbo', extent=extent)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f'FFT Amplitude at f={freqs[idx]/1e6:.2f} MHz', pad=10)
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label('Amplitude')
    plt.tight_layout()
    plt.savefig('output/amplitude.png')
    plt.close()
    print("\nFinal amplitude plot saved to 'output/amplitude.png'.")

    # --- Line Plot along Center Line ---
    print("Generating line plot along the center of the slice...")
    center_col_index = amp.shape[1] // 2
    line_data = amp[:, center_col_index]
    vertical_axis_coords = np.linspace(extent[2], extent[3], amp.shape[0])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(vertical_axis_coords, line_data, color='b')
    ax.set_xlabel(ylabel)
    ax.set_ylabel('Amplitude')
    ax.set_title(f'Amplitude along Vertical Center Line')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig('output/amplitude_line_plot.png')
    plt.close()
    print("Line plot saved to 'output/amplitude_line_plot.png'.")
    print("\nProcessing complete.")