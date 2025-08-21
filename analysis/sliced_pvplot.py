import argparse
import multiprocessing as mp
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from paraview.simple import ADIOS2VTXReader, ResampleToImage, servermanager
from vtk.util.numpy_support import vtk_to_numpy
from scipy.io import savemat
from scipy.signal import get_window


# --- Default Configuration ---
DEFAULT_FIELD_NAME = "u"
DEFAULT_SAMPLING_DIMS = [251, 1, 251]
DEFAULT_TARGET_FREQ_HZ = 0.5e6
DEFAULT_OUTPUT_DIR = Path("./output")

# --- Matplotlib Plotting Style ---
def setup_plotting_style():
    """Sets a consistent, publication-quality style for all generated plots."""
    plt.rcParams.update({
        "font.family": "serif", "font.size": 10, "axes.labelsize": 10,
        "axes.titlesize": 11, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "lines.linewidth": 1.2, "axes.linewidth": 0.8, "xtick.direction": "in",
        "ytick.direction": "in", "figure.dpi": 300,
    })

def parse_arguments():
    """Parses command-line arguments for the script."""
    parser = argparse.ArgumentParser(
        description="Process an ADIOS2 BP file, perform FFT, and generate plots in parallel.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # ... (rest of the argument parser is unchanged) ...
    parser.add_argument("input_file", type=Path, help="Path to the input .bp file.")
    parser.add_argument("n_timesteps", type=int, nargs="?", default=5, help="Number of final timesteps to analyze.")
    parser.add_argument("--field", type=str, default=DEFAULT_FIELD_NAME, help="Name of the data field to process.")
    parser.add_argument("--freq", type=float, default=DEFAULT_TARGET_FREQ_HZ, help="Target frequency for FFT in Hz.")
    parser.add_argument("--dims", nargs=3, type=int, default=DEFAULT_SAMPLING_DIMS, metavar=('NX', 'NY', 'NZ'),
                        help="Sampling dimensions for the slice. One dimension must be 1.")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory to save output images.")
    parser.add_argument("--jobs", type=int, default=-1,
                        help="Number of parallel jobs for data loading and plotting. -1 uses all available cores.")
    
    parser.add_argument(
        "--window",
        type=str,
        default="hann",
        choices=["rect", "hann", "hamming", "blackman", "blackmanharris", "flattop", "tukey"],
        help="Time-domain window for FFT amplitude extraction."
    )
    parser.add_argument(
        "--tukey-alpha",
        type=float,
        default=0.5,
        help="Alpha parameter for Tukey window (ignored unless --window=tukey)."
    )
    args = parser.parse_args()

    if 1 not in args.dims:
        parser.error("--dims must contain exactly one '1' to define a 2D slice.")

    return args


def get_slice_plane_info(sampling_dims, bounds):
    """Determines the slice orientation, axis labels, and extent from sampling dimensions."""
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    if sampling_dims[1] == 1:  # X-Z plane
        return {"orientation": "XZ", "shape": (sampling_dims[2], sampling_dims[0]),
                "extent": [xmin, xmax, zmin, zmax], "xlabel": "x (m)", "ylabel": "z (m)",
                "line_plot_axis": "z", "line_plot_xlabel": "z (m)"}
    elif sampling_dims[0] == 1:  # Y-Z plane
        return {"orientation": "YZ", "shape": (sampling_dims[2], sampling_dims[1]),
                "extent": [ymin, ymax, zmin, zmax], "xlabel": "y (m)", "ylabel": "z (m)",
                "line_plot_axis": "z", "line_plot_xlabel": "z (m)"}
    else:  # X-Y plane
        return {"orientation": "XY", "shape": (sampling_dims[1], sampling_dims[0]),
                "extent": [xmin, xmax, ymin, ymax], "xlabel": "x (m)", "ylabel": "y (m)",
                "line_plot_axis": "y", "line_plot_xlabel": "y (m)"}

def load_one_timestep(t, input_file, field_name, sampling_dims):
    """Loads and processes a single timestep in an isolated process."""
    from paraview.simple import ADIOS2VTXReader, ResampleToImage, servermanager
    from vtk.util.numpy_support import vtk_to_numpy

    reader = ADIOS2VTXReader(FileName=[str(input_file)])
    resampler = ResampleToImage(Input=reader)
    resampler.SamplingDimensions = sampling_dims
    resampler.UpdatePipeline(t)
    vtk_data = servermanager.Fetch(resampler)
    numpy_data = vtk_to_numpy(vtk_data.GetPointData().GetArray(field_name))
    print(f"  ...Worker finished loading timestep t={t:.6f}")
    return numpy_data

def load_data_in_parallel(input_file, times, field_name, sampling_dims, n_jobs):
    """Manages a pool of workers to load timestep data in parallel."""
    # ... (function is unchanged) ...
    start_time = time.time()
    if n_jobs == -1:
        n_jobs = min(os.cpu_count(), len(times))
    print(f"Loading {len(times)} timesteps in parallel using {n_jobs} jobs...")
    
    task_args = [(t, input_file, field_name, sampling_dims) for t in times]
    
    with mp.Pool(processes=n_jobs) as pool:
        results_list = pool.starmap(load_one_timestep, task_args)
    
    data_stack = np.vstack(results_list)
    
    end_time = time.time()
    print(f"Parallel data loading finished in {end_time - start_time:.2f} seconds.")
    return data_stack


def perform_fft_analysis(data_stack, times, target_freq, window_name="hann", tukey_alpha=0.5):
    """FFT amplitude at target_freq with optional windowing.
       Amplitude normalization uses coherent gain sum(window)."""
    print("Performing FFT analysis...")
    nt, npts = data_stack.shape

    if len(times) > 1:
        dt = float(np.mean(np.diff(times)))
    else:
        dt = 1.0
        print("Warning: Only one timestep. FFT results may not be meaningful.")

    # Build window
    name_map = {
        "rect": "boxcar",
        "hann": "hann",
        "hamming": "hamming",
        "blackman": "blackman",
        "blackmanharris": "blackmanharris",
        "flattop": "flattop",
        "tukey": ("tukey", tukey_alpha),
    }
    gw = name_map[window_name]
    w = get_window(gw, nt, fftbins=True).astype(data_stack.dtype, copy=False)
    wsum = float(w.sum())

    # Apply window and FFT
    X = np.fft.rfft(w[:, None] * data_stack, axis=0)
    freqs = np.fft.rfftfreq(nt, dt)

    idx = int(np.argmin(np.abs(freqs - target_freq)))
    closest_freq = float(freqs[idx])
    print(f"Window: {window_name}  (sum={wsum:.6g})")
    print(f"Target f0={target_freq/1e6:.2f} MHz, closest FFT bin={closest_freq/1e6:.2f} MHz")

    # Amplitude with coherent-gain correction: for a unit sinusoid at-bin,
    # |X[k0]| = (A/2)*sum(w)  =>  A = 2*|X[k0]|/sum(w)
    amplitude = 2.0 * np.abs(X[idx, :]) / wsum
    return amplitude, closest_freq


# --- NEW: Worker function for parallel plotting ---
def generate_single_timestep_plot(i, t, timestep_data, vmin, vmax, field_name, plane_info, output_dir):
    """
    Generates and saves a plot for a single timestep. Designed to be run in a worker process.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    data = timestep_data.reshape(plane_info["shape"])
    im = ax.imshow(
        data,
        cmap='seismic', aspect='auto', origin='lower',
        vmin=-vmax, vmax=vmax, extent=plane_info["extent"]
    )
    ax.set_xlabel(plane_info["xlabel"])
    ax.set_ylabel(plane_info["ylabel"])
    ax.set_title(f'Field: "{field_name}" at Timestep {i+1} (t={t:.6f})')
    fig.colorbar(im, ax=ax, label='Field Value')
    plt.tight_layout()
    output_path = output_dir / f"field_timestep_{i+1:03d}.png"
    plt.savefig(output_path)
    plt.close(fig)
    print(f"  ...Worker finished plotting timestep {i+1}")
    
    # plot line in the middle
    center_line = data[:, data.shape[1] // 2]
    plt.plot(center_line, color='k', lw=2)
    plt.xlim(right=100)
    
    plt.tight_layout()
    plt.savefig(output_dir / f"field_timestep_{i+1:03d}_line.png")

# --- NEW: Manager function for parallel plotting ---
def plot_timesteps_in_parallel(data_stack, times, field_name, plane_info, output_dir, n_jobs):
    """Manages a pool of workers to generate timestep plots in parallel."""
    start_time = time.time()
    if n_jobs == -1:
        n_jobs = min(os.cpu_count(), len(times))
    print(f"Generating {len(times)} timestep images in parallel using {n_jobs} jobs...")
    
    # Calculate color limits once to ensure consistency across all plots
    vmin, vmax = data_stack.min(), data_stack.max()

    # Prepare arguments for each plotting task
    tasks = []
    for i, t in enumerate(times):
        task_args = (
            i, t, data_stack[i], vmin, vmax, field_name, plane_info, output_dir
        )
        tasks.append(task_args)

    with mp.Pool(processes=n_jobs) as pool:
        pool.starmap(generate_single_timestep_plot, tasks)
        
    end_time = time.time()
    print(f"Parallel plotting finished in {end_time - start_time:.2f} seconds.")

def plot_fft_amplitude(amplitude_map, plane_info, freq, output_dir):
    """Saves the final 2D FFT amplitude map."""
    # ... (function is unchanged) ...
    print("Generating final FFT amplitude plot...")
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(
        amplitude_map.reshape(plane_info["shape"]),
        origin='lower', aspect='auto', cmap='turbo', extent=plane_info["extent"]
    )
    ax.set_xlabel(plane_info["xlabel"])
    ax.set_ylabel(plane_info["ylabel"])
    ax.set_title(f'FFT Amplitude at f={freq/1e6:.2f} MHz', pad=10)
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label('Amplitude')
    plt.tight_layout()
    plt.savefig(output_dir / 'amplitude_map.png')
    plt.close(fig)
    print(f"Final amplitude map saved to '{output_dir / 'amplitude_map.png'}'")

def plot_centerline(amplitude_map, plane_info, output_dir):
    """Plots the amplitude along the vertical center line of the slice."""
    # ... (function is unchanged) ...
    print("Generating line plot along the center of the slice...")
    amp_reshaped = amplitude_map.reshape(plane_info["shape"])
    
    center_col_index = amp_reshaped.shape[1] // 2
    line_data = amp_reshaped[:, center_col_index]

    _, _, ymin, ymax = plane_info["extent"]
    vertical_coords = np.linspace(ymin, ymax, amp_reshaped.shape[0])
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(vertical_coords, line_data, color='b')
    ax.set_xlabel(plane_info["line_plot_xlabel"])
    ax.set_ylabel('Amplitude')
    ax.set_title('Amplitude along Vertical Center Line')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / 'amplitude_line_plot.png')
    plt.close(fig)
    print(f"Line plot saved to '{output_dir / 'amplitude_line_plot.png'}'")

def main():
    """Main execution function."""
    args = parse_arguments()
    setup_plotting_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"--- Starting Analysis ---")
    print(f"Input File: {args.input_file}")
    print(f"Timesteps:  Last {args.n_timesteps}")
    print(f"Field:      '{args.field}'")
    print(f"Frequency:  {args.freq / 1e6} MHz")
    print(f"Output Dir: {args.output_dir}")
    print(f"Jobs:       {args.jobs if args.jobs != -1 else 'All Cores'}")
    print("-" * 25)

    # 1. Load Metadata
    print("Loading metadata...")
    meta_reader = ADIOS2VTXReader(FileName=[str(args.input_file)])
    meta_reader.UpdatePipeline()
    
    all_times = meta_reader.TimestepValues
    times_to_process = [all_times] if isinstance(all_times, float) else all_times[-args.n_timesteps:]
    data_bounds = meta_reader.GetDataInformation().GetBounds()
    plane_info = get_slice_plane_info(args.dims, data_bounds)

    print(f"Found {len(all_times) if not isinstance(all_times, float) else 1} total timesteps. Analyzing the last {len(times_to_process)}.")
    print(f"Data Bounds: [xmin, xmax, ymin, ymax, zmin, zmax] = {data_bounds}")
    print(f"Slice Info: Plane={plane_info['orientation']}, Shape={plane_info['shape']}")
    
    # 2. Load Data in Parallel
    data_stack = load_data_in_parallel(
        args.input_file, times_to_process, args.field, args.dims, args.jobs
    )
    
    amplitude_map, actual_freq = perform_fft_analysis(
        data_stack, times_to_process, args.freq, args.window, args.tukey_alpha
    )
    
    # Save
    savemat(args.output_dir / 'amplitude.mat', {'amplitude_map': amplitude_map})
    print(f"Amplitude map saved to '{args.output_dir / 'amplitude.mat'}'")


    # 4. Generate Plots (Timestep plots are now parallel)
    plot_timesteps_in_parallel(
        data_stack, times_to_process, args.field, plane_info, args.output_dir, args.jobs
    )
    plot_fft_amplitude(amplitude_map, plane_info, actual_freq, args.output_dir)
    plot_centerline(amplitude_map, plane_info, args.output_dir)

    print("\n--- Processing complete. ---")

if __name__ == "__main__":
    # 'spawn' is crucial for both VTK and Matplotlib when using multiprocessing.
    mp.set_start_method("spawn", force=True)
    main()