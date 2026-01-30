"""
Widefield-HDF5 Alignment Pipeline
==================================
This script performs trigger-based temporal mapping between Analog.dat files and Stimulus HDF5 clock,
then aligns and synchronizes widefield frames with offset corrections to create a consolidated
global timeline matrix.

Key Steps:
1. Load and normalize HDF5 synchronization data (master clock at 10kHz)
2. Identify trial boundaries from HDF5 trigger signals (Ch3=Start, Ch4=Stop)
3. Load Analog.dat files (1kHz local recordings from widefield computer)
4. Perform trigger-based temporal mapping between HDF5 and Analog files
5. Align widefield frames with offset/baseline corrections from frameTimes.mat
6. Upsample and consolidate all data into single global timeline matrix (10kHz)
7. Generate verification plots

Author: Matthew Loukine
Date: January 2026
"""

import os
import glob
import struct
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import scipy.io
from datetime import timedelta


# ============================================================================
# GLOBAL CONFIGURATION
# ============================================================================

# Set your data directory here
DATA_DIR = r"D:\Vanessa_test_data\Tests_Jan23\23-Jan-2026_ledTTL_10random"
OUTPUT_DIR = None  # If None, will default to DATA_DIR/alignment


# ============================================================================
# CONFIGURATION
# ============================================================================

class AlignmentConfig:
    """Configuration parameters for alignment pipeline."""
    
    # HDF5 Channel Mapping
    CHANNEL_MAP = {
        0: 'water_valve',
        1: 'block_timing',
        2: 'led_timing',
        3: 'acquisition_start',
        4: 'stop_signal',
        5: 'eye_camera',
        6: 'photodiode',
        7: 'rotary_z',
        8: 'rotary_a',
        9: 'rotary_b',
        10: 'lickometer',
        11: 'wrong_choice',
        12: 'mhc_timing',
        13: 'behav_cam_2'
    }
    
    # Sampling Rates
    HDF5_SAMPLING_RATE = 10000  # Hz
    ANALOG_SAMPLING_RATE = 1000  # Hz
    FRAME_RATE = 20  # Hz (approximate)
    
    # Detection Thresholds
    TRIGGER_THRESHOLD = 0.5  # For normalized signals
    
    # Frame Dimensions (default)
    DEFAULT_FRAME_SHAPE = (2, 640, 540)  # (channels, height, width)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def normalize_arr(arr):
    """Normalize array to 0-1 range."""
    arr = arr.astype(float)
    min_v = np.nanmin(arr)
    max_v = np.nanmax(arr)
    if max_v - min_v == 0:
        return arr
    return (arr - min_v) / (max_v - min_v)


def find_edges(signal, threshold=0.5):
    """Find rising and falling edges in a signal."""
    binary = signal > threshold
    rising = np.where(np.diff(binary.astype(int)) == 1)[0]
    falling = np.where(np.diff(binary.astype(int)) == -1)[0]
    return rising, falling


def get_file_index(fname):
    """Extract numerical index from filename."""
    base = os.path.splitext(os.path.basename(fname))[0]
    try:
        parts = base.split('_')
        for part in reversed(parts):
            if part.isdigit():
                return int(part)
        return 0
    except:
        return 0


# ============================================================================
# HDF5 DATA LOADING
# ============================================================================

def load_hdf5_data(hdf5_path, verbose=True):
    """
    Load and normalize HDF5 synchronization data.
    
    Args:
        hdf5_path: Path to HDF5 file
        verbose: Print loading information
        
    Returns:
        Normalized data array (n_samples, n_channels)
    """
    if verbose:
        print(f"Loading HDF5 file: {hdf5_path}")
    
    with h5py.File(hdf5_path, 'r') as f:
        data = None
        
        def find_sync_data(name, obj):
            nonlocal data
            if isinstance(obj, h5py.Dataset) and 'sync' in name:
                data = obj[:]
                if verbose:
                    print(f"  Found sync data in: {name} with shape {obj.shape}")
            elif isinstance(obj, h5py.Dataset) and data is None and len(obj.shape) > 1 and obj.shape[1] >= 14:
                data = obj[:]
                if verbose:
                    print(f"  Found analog data in: {name} with shape {obj.shape}")
        
        f.visititems(find_sync_data)
        
        if data is None:
            raise ValueError("Could not find sync/analog channel data in HDF5 file")
    
    # Normalize all channels
    if verbose:
        print(f"  Normalizing {data.shape[1]} channels...")
    
    data = data.astype(float)
    for i in range(data.shape[1]):
        data[:, i] = normalize_arr(data[:, i])
    
    return data


def identify_hdf5_trials(data, config=AlignmentConfig(), verbose=True):
    """
    Identify trial boundaries from HDF5 trigger signals.
    
    Args:
        data: Normalized HDF5 data array
        config: Configuration object
        verbose: Print information
        
    Returns:
        List of trial dictionaries with timing information
    """
    if verbose:
        print("\nIdentifying HDF5 trial boundaries...")
    
    c3 = data[:, 3]  # Acquisition Start
    c4 = data[:, 4]  # Stop Signal
    
    start_rising, _ = find_edges(c3, threshold=config.TRIGGER_THRESHOLD)
    stop_rising, _ = find_edges(c4, threshold=config.TRIGGER_THRESHOLD)
    
    if verbose:
        print(f"  Found {len(start_rising)} start triggers")
        if len(start_rising) > 1:
            gaps = np.diff(start_rising)
            print(f"  Average gap: {np.mean(gaps):.2f} samples ({np.mean(gaps)/config.HDF5_SAMPLING_RATE:.2f} s)")
    
    trials_hdf5 = []
    for i in range(len(start_rising)):
        start_idx = start_rising[i]
        
        # Find corresponding stop trigger
        relevant_stops = stop_rising[stop_rising > start_idx]
        if len(relevant_stops) > 0:
            stop_idx = relevant_stops[0]
        else:
            stop_idx = start_idx + int(10 * config.HDF5_SAMPLING_RATE)  # Default 10s
        
        trials_hdf5.append({
            'trial_idx': i,
            'start_sample': start_idx,
            'stop_sample': stop_idx,
            'start_time_s': start_idx / config.HDF5_SAMPLING_RATE
        })
    
    if verbose:
        print(f"  Identified {len(trials_hdf5)} trials")
    
    return trials_hdf5


# ============================================================================
# ANALOG DATA LOADING
# ============================================================================

def load_single_analog(file_path, verbose=False):
    """
    Load a single Analog_*.dat file.
    
    File format:
        - Header: 4 doubles (32 bytes)
        - Data: uint16 array, reshaped to (n_channels, n_samples)
    
    Args:
        file_path: Path to Analog file
        verbose: Print loading info
        
    Returns:
        (data, metadata) tuple or (None, None) on error
    """
    try:
        with open(file_path, 'rb') as fd:
            # Read header
            header_bytes = fd.read(32)
            if len(header_bytes) < 32:
                return None, None
            
            onset = struct.unpack("d", header_bytes[8:16])[0]
            nchannels = int(struct.unpack("<d", header_bytes[16:24])[0])
            
            # Read data
            dat = np.fromfile(fd, dtype='uint16')
            
            # Reshape
            if len(dat) % nchannels != 0:
                if verbose:
                    print(f"  Warning: {os.path.basename(file_path)} data not divisible by {nchannels}")
                new_len = (len(dat) // nchannels) * nchannels
                dat = dat[:new_len]
            
            dat = dat.reshape((-1, nchannels)).T
            return dat, {'onset': onset, 'nchannels': nchannels, 'path': file_path}
            
    except Exception as e:
        if verbose:
            print(f"  Error reading {file_path}: {e}")
        return None, None


def load_all_analog_files(data_dir, config=AlignmentConfig(), verbose=True):
    """
    Load all Analog_*.dat files and identify local triggers.
    
    Args:
        data_dir: Directory containing Analog files
        config: Configuration object
        verbose: Print information
        
    Returns:
        List of analog trial dictionaries
    """
    if verbose:
        print("\nLoading Analog.dat files...")
    
    analog_files = glob.glob(os.path.join(data_dir, "Analog_*.dat"))
    analog_files.sort(key=get_file_index)
    
    if not analog_files:
        raise ValueError("No Analog_*.dat files found")
    
    if verbose:
        print(f"  Found {len(analog_files)} Analog files")
    
    analog_trials = []
    for i, f in enumerate(analog_files):
        dat, meta = load_single_analog(f, verbose=verbose)
        if dat is not None:
            # Find local start trigger (Ch3 = received from HDF5)
            if dat.shape[0] > 3:
                ch3_normalized = normalize_arr(dat[3].astype(float))
                local_starts, _ = find_edges(ch3_normalized, config.TRIGGER_THRESHOLD)
                local_start_sample = local_starts[0] if len(local_starts) > 0 else 0
            else:
                local_start_sample = 0
            
            analog_trials.append({
                'file_idx': i,
                'filename': os.path.basename(f),
                'analog_data_shape': dat.shape,
                'local_start_sample': local_start_sample,
                'data': dat
            })
            
            if verbose:
                print(f"  Trial {i}: {os.path.basename(f)} | Shape: {dat.shape} | Start: {local_start_sample}")
    
    return analog_trials


# ============================================================================
# FRAME DATA LOADING AND PROCESSING
# ============================================================================

def check_blue_is_first_via_intensity(dat_file_path, H, W, dtype=np.uint16):
    """
    Check if Blue channel is first by comparing frame intensities.
    Violet (hemodynamic) is typically brighter than Blue (functional).
    
    Args:
        dat_file_path: Path to Frames file
        H, W: Frame dimensions
        dtype: Data type
        
    Returns:
        True if Blue is first channel
    """
    try:
        pixels_per_frame = H * W
        bytes_per_pixel = np.dtype(dtype).itemsize
        bytes_to_read = 2 * pixels_per_frame * bytes_per_pixel
        
        with open(dat_file_path, 'rb') as f:
            data_bytes = f.read(bytes_to_read)
        
        if len(data_bytes) < bytes_to_read:
            return True
        
        data = np.frombuffer(data_bytes, dtype=dtype)
        if len(data) < pixels_per_frame * 2:
            return True
        
        m0 = np.mean(data[:pixels_per_frame])
        m1 = np.mean(data[pixels_per_frame:])
        
        # Violet is brighter, so if m0 < m1, Blue is first
        return m0 < m1
    except:
        return True


def load_frame_offset(frameTimes_path, verbose=False):
    """
    Load offset information from frameTimes.mat file.
    
    Priority:
        1. removedFrames (if > 0)
        2. preStim
        
    Args:
        frameTimes_path: Path to frameTimes.mat file
        verbose: Print information
        
    Returns:
        Number of offset frames
    """
    try:
        ft_mat = scipy.io.loadmat(frameTimes_path)
        
        # Check removedFrames first
        if 'removedFrames' in ft_mat:
            removed = int(ft_mat['removedFrames'].flatten()[0])
            if removed > 0:
                if verbose:
                    print(f"    Using removedFrames: {removed}")
                return removed
        
        # Fall back to preStim
        if 'preStim' in ft_mat:
            prestim = int(ft_mat['preStim'].flatten()[0])
            if verbose:
                print(f"    Using preStim: {prestim}")
            return prestim
        
        return 0
        
    except Exception as e:
        if verbose:
            print(f"    Error loading frameTimes.mat: {e}")
        return 0


def process_frames_with_offset(frames_files, frameTimes_files, analog_trials, trials_hdf5, 
                                config=AlignmentConfig(), verbose=True):
    """
    Process frame files with offset correction and align to global timeline.
    
    This is the critical "Step 6" that performs frame synchronization with offset handling.
    
    Args:
        frames_files: List of Frames_*.dat files
        frameTimes_files: List of frameTimes*.mat files
        analog_trials: List of analog trial dictionaries
        trials_hdf5: List of HDF5 trial dictionaries
        config: Configuration object
        verbose: Print information
        
    Returns:
        (global_blue_times, global_blue_vals) - Lists of aligned frame times and values
    """
    if verbose:
        print("\n" + "="*70)
        print("STEP 6: Frame Synchronization with Offset Correction")
        print("="*70)
        print("Interpreting 'preStim'/'removedFrames' as START DELAY.")
        print("Action: Keep ALL frames. Align Frame[0] to TTL[Offset].")
        print("Result: Frames are shifted LATER in time (rightward).")
        print("-"*70)
    
    global_blue_times = []
    global_blue_vals = []
    
    n_proc = min(len(analog_trials), len(frames_files))
    
    for i in range(n_proc):
        try:
            a_trial = analog_trials[i]
            h_trial = trials_hdf5[i]
            f_path = frames_files[i]
            
            # Load offset from frameTimes.mat
            offset_frames = 0
            if i < len(frameTimes_files):
                offset_frames = load_frame_offset(frameTimes_files[i], verbose=verbose)
                offset_time_s = offset_frames * (1.0 / config.FRAME_RATE)
                if verbose:
                    print(f"\nTrial {i}:")
                    print(f"  Offset: {offset_frames} frames ~ {offset_time_s:.2f}s delay")
                    print(f"  Mapping: Frame[0] -> TTL[{offset_frames}]")
            
            # Determine frame dimensions
            channels, h_dim, w_dim = config.DEFAULT_FRAME_SHAPE
            parts = os.path.basename(f_path).split('_')
            nums = [int(p) for p in parts if p.isdigit()]
            if len(nums) >= 3:
                channels, h_dim, w_dim = nums[:3]
            
            # Check frame order (Blue first vs Violet first)
            is_blue_first = check_blue_is_first_via_intensity(f_path, h_dim, w_dim)
            if not is_blue_first and verbose:
                print(f"  Detected: Violet first (swapped sequence)")
            
            # Get TTLs from Analog (Ch1=Blue, Ch2=Violet)
            raw_ch1 = a_trial['data'][1].astype(float)
            raw_ch2 = a_trial['data'][2].astype(float)
            
            r_blue, _ = find_edges(normalize_arr(raw_ch1), config.TRIGGER_THRESHOLD)
            r_violet, _ = find_edges(normalize_arr(raw_ch2), config.TRIGGER_THRESHOLD)
            
            # Create sorted list of all TTL events
            ttls = [(t, 'blue') for t in r_blue] + [(t, 'violet') for t in r_violet]
            ttls.sort(key=lambda x: x[0])
            
            if not ttls:
                if verbose:
                    print(f"  No TTLs found. Skipping.")
                continue
            
            if offset_frames >= len(ttls):
                if verbose:
                    print(f"  Offset {offset_frames} >= TTLs {len(ttls)}. Skipping.")
                continue
            
            # Load all frames
            with open(f_path, 'rb') as f:
                raw_dat = np.fromfile(f, dtype='uint16')
            
            n_pix = h_dim * w_dim
            n_total_frames = len(raw_dat) // n_pix
            
            # Sync Logic: Frame[0] -> TTL[offset_frames]
            ttl_idx = offset_frames
            frame_idx = 0
            
            # Verify color match at alignment point
            frame0_color = 'blue' if is_blue_first else 'violet'
            ttl_offset_color = ttls[ttl_idx][1]
            
            if frame0_color != ttl_offset_color:
                if verbose:
                    print(f"  Color mismatch: Frame[0]={frame0_color}, TTL[{ttl_idx}]={ttl_offset_color}")
                    print(f"  Adjusting: +1 TTL")
                ttl_idx += 1
                if ttl_idx >= len(ttls):
                    if verbose:
                        print(f"  Adjustment failed. Skipping.")
                    continue
            
            # Calculate global time offset
            t_trig_glob = h_trial['start_time_s']
            t_trig_loc_s = a_trial['local_start_sample'] / config.ANALOG_SAMPLING_RATE
            t_start_glob = t_trig_glob - t_trig_loc_s
            
            # Extract Blue frame data
            trial_times = []
            trial_vals = []
            
            while frame_idx < n_total_frames and ttl_idx < len(ttls):
                # Determine frame color
                frame_is_blue = (frame0_color == 'blue' and frame_idx % 2 == 0) or \
                               (frame0_color == 'violet' and frame_idx % 2 != 0)
                
                if frame_is_blue:
                    # Extract frame pixels
                    start_ptr = frame_idx * n_pix
                    end_ptr = start_ptr + n_pix
                    frame_pixels = raw_dat[start_ptr:end_ptr]
                    val = np.mean(frame_pixels)
                    
                    # Get global time from TTL
                    ttl_sample = ttls[ttl_idx][0]
                    t_glob = t_start_glob + (ttl_sample / config.ANALOG_SAMPLING_RATE)
                    
                    trial_times.append(t_glob)
                    trial_vals.append(val)
                
                frame_idx += 1
                ttl_idx += 1
            
            if verbose:
                print(f"  Extracted {len(trial_vals)} Blue frames")
            
            global_blue_times.extend(trial_times)
            global_blue_vals.extend(trial_vals)
            
        except Exception as e:
            if verbose:
                print(f"  Error processing trial {i}: {e}")
                import traceback
                traceback.print_exc()
    
    return global_blue_times, global_blue_vals


# ============================================================================
# MATRIX CONSTRUCTION
# ============================================================================

def create_consolidated_matrix(data, analog_trials, trials_hdf5, config=AlignmentConfig(), 
                                verbose=True):
    """
    Create consolidated matrix with upsampled analog data.
    
    Args:
        data: HDF5 data array
        analog_trials: List of analog trial dictionaries
        trials_hdf5: List of HDF5 trial dictionaries
        config: Configuration object
        verbose: Print information
        
    Returns:
        Consolidated matrix (n_samples, n_hdf5_channels + n_analog_channels)
    """
    if verbose:
        print("\nConstructing consolidated matrix (upsampling analog to 10kHz)...")
    
    n_samples_total = data.shape[0]
    n_hdf5_ch = 14
    n_analog_ch = 5
    n_total_ch = n_hdf5_ch + n_analog_ch
    
    # Initialize with HDF5 data
    full_matrix = np.zeros((n_samples_total, n_total_ch), dtype=np.float32)
    full_matrix[:, 0:n_hdf5_ch] = data.astype(np.float32)
    
    # Fill analog data (upsampling from 1kHz to 10kHz)
    mapped_count = min(len(trials_hdf5), len(analog_trials))
    
    for i in range(mapped_count):
        h_trial = trials_hdf5[i]
        a_trial = analog_trials[i]
        
        # Calculate start time
        t_trig_glob = h_trial['start_time_s']
        t_trig_loc_s = a_trial['local_start_sample'] / config.ANALOG_SAMPLING_RATE
        t_start_glob = t_trig_glob - t_trig_loc_s
        
        start_sample_idx = int(t_start_glob * config.HDF5_SAMPLING_RATE)
        
        # Get analog data
        analog_dat = a_trial['data']
        n_samples_1k = analog_dat.shape[1]
        
        # Upsample 10x using linear interpolation
        t_1k = np.arange(n_samples_1k) / config.ANALOG_SAMPLING_RATE
        n_samples_10k = int(n_samples_1k * 10)
        t_10k = np.linspace(0, t_1k[-1], n_samples_10k)
        
        # Handle bounds
        if start_sample_idx < 0:
            start_sample_idx = 0
        
        end_sample_idx = start_sample_idx + n_samples_10k
        if end_sample_idx > n_samples_total:
            n_samples_10k = n_samples_total - start_sample_idx
            t_10k = t_10k[:n_samples_10k]
            end_sample_idx = n_samples_total
        
        # Interpolate each analog channel
        for ch in range(n_analog_ch):
            col_idx = n_hdf5_ch + ch
            val_1k = analog_dat[ch].astype(np.float32)
            val_10k = np.interp(t_10k, t_1k, val_1k)
            full_matrix[start_sample_idx:end_sample_idx, col_idx] = val_10k
    
    if verbose:
        print(f"  Matrix shape: {full_matrix.shape}")
    
    return full_matrix


def add_blue_trace_to_matrix(full_matrix, global_blue_times, global_blue_vals, 
                             config=AlignmentConfig(), verbose=True):
    """
    Add interpolated blue trace to consolidated matrix.
    
    Args:
        full_matrix: Existing consolidated matrix
        global_blue_times: List of frame times (seconds)
        global_blue_vals: List of frame values
        config: Configuration object
        verbose: Print information
        
    Returns:
        Extended matrix with blue trace as last column
    """
    if verbose:
        print("\nAdding raw Blue trace to matrix...")
    
    # Create extended matrix
    n_existing_ch = full_matrix.shape[1]
    new_full_matrix = np.zeros((full_matrix.shape[0], n_existing_ch + 1), dtype=np.float32)
    new_full_matrix[:, :n_existing_ch] = full_matrix
    
    # Sort blue trace data
    srt = np.argsort(global_blue_times)
    t_source = np.array(global_blue_times)[srt]
    v_source = np.array(global_blue_vals)[srt]
    
    # Target timeline
    n_samples = full_matrix.shape[0]
    t_target = np.arange(n_samples) / config.HDF5_SAMPLING_RATE
    
    # Interpolate
    blue_trace = np.interp(t_target, t_source, v_source, left=0, right=0)
    
    # Assign to last column
    new_full_matrix[:, n_existing_ch] = blue_trace.astype(np.float32)
    
    if verbose:
        print(f"  Extended matrix shape: {new_full_matrix.shape}")
    
    return new_full_matrix


# ============================================================================
# VISUALIZATION
# ============================================================================

# Define consistent color scheme for channels
CHANNEL_COLORS = {
    # HDF5 channels (0-13)
    0: '#1f77b4',  # water_valve - blue
    1: '#ff7f0e',  # block_timing - orange
    2: '#2ca02c',  # led_timing - green
    3: '#d62728',  # acquisition_start - red
    4: '#9467bd',  # stop_signal - purple
    5: '#8c564b',  # eye_camera - brown
    6: '#e377c2',  # photodiode - pink
    7: '#7f7f7f',  # rotary_z - gray
    8: '#bcbd22',  # rotary_a - yellow-green
    9: '#17becf',  # rotary_b - cyan
    10: '#ff9896', # lickometer - light red
    11: '#aec7e8', # wrong_choice - light blue
    12: '#c5b0d5', # mhc_timing - light purple
    13: '#c49c94', # behav_cam_2 - light brown
    # Analog channels (14-18)
    14: '#f7b6d2', # a0_timing - light pink
    15: '#0000ff', # a1_blue_led - pure blue
    16: '#9400d3', # a2_violet_led - dark violet
    17: '#00ff00', # a3_start - lime
    18: "#ff00d9", # a4_stim - vivid pink
    # Blue trace
    19: '#000080', # blue_trace - navy blue
}

def plot_trigger_alignment(data, analog_trials, trials_hdf5, output_dir, 
                           config=AlignmentConfig(), n_plot=5):
    """Plot alignment verification for first N trials in combined graphs."""
    print("\nGenerating combined trigger alignment plot...")
    
    mapped_count = min(len(trials_hdf5), len(analog_trials), n_plot)
    
    # Create figure with 2 subplots: Ch3 verification and Ch1/Ch4 stimulus verification
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 10))
    
    # Get channel colors
    hdf5_ch3_color = CHANNEL_COLORS.get(3, '#d62728')  # acquisition_start - red
    analog_ch3_color = CHANNEL_COLORS.get(17, '#00ff00')  # a3_start - lime
    hdf5_ch1_color = CHANNEL_COLORS.get(1, '#ff7f0e')  # block_timing - orange
    analog_ch4_color = CHANNEL_COLORS.get(18, '#ff00d9')  # a4_stim - vivid pink
    
    yticks_locs_1 = []
    yticks_labels_1 = []
    yticks_locs_2 = []
    yticks_labels_2 = []
    
    # --- Subplot 1: HDF5 Ch3 vs Analog Ch3 (Acquisition Start Verification) ---
    for i in range(mapped_count):
        h_trial = trials_hdf5[i]
        a_trial = analog_trials[i]
        
        # HDF5 Ch3 window
        h_start = h_trial['start_sample']
        h_idx_start = max(0, h_start - int(1 * config.HDF5_SAMPLING_RATE))
        h_idx_end = min(len(data), h_start + int(5 * config.HDF5_SAMPLING_RATE))
        
        hdf5_segment = data[h_idx_start:h_idx_end, 3]
        hdf5_time = (np.arange(h_idx_start, h_idx_end) - h_start) / config.HDF5_SAMPLING_RATE
        
        # Analog Ch3
        a_dat = a_trial['data']
        a_sig = normalize_arr(a_dat[3].astype(float))
        a_trig_idx = a_trial['local_start_sample']
        analog_time = (np.arange(len(a_sig)) - a_trig_idx) / config.ANALOG_SAMPLING_RATE
        
        # Plot with offset for visibility
        offset = i * 1.2
        ax1.plot(hdf5_time, hdf5_segment + offset, 
                color=hdf5_ch3_color, linewidth=1.5, 
                label='HDF5 Ch3 (Acq Start)' if i == 0 else "")
        ax1.plot(analog_time, a_sig + offset, 
                color=analog_ch3_color, linestyle='--', linewidth=1.2, alpha=0.8,
                label='Analog Ch3 (Start Rx)' if i == 0 else "")
        
        # Store y-tick positions and labels
        yticks_locs_1.append(offset + 0.5)
        yticks_labels_1.append(f'Trial {i}')
    
    ax1.set_title('Acquisition Start Trigger Verification (HDF5 Ch3 vs Analog Ch3)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Time relative to Trigger (s)')
    ax1.set_ylabel('Trial')
    ax1.set_xlim(-1, 5)
    ax1.set_yticks(yticks_locs_1)
    ax1.set_yticklabels(yticks_labels_1)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # --- Subplot 2: HDF5 Ch1 vs Analog Ch4 (Stimulus Timing Verification) ---
    for i in range(mapped_count):
        h_trial = trials_hdf5[i]
        a_trial = analog_trials[i]
        
        # HDF5 Ch1 window
        h_start = h_trial['start_sample']
        h_idx_start = max(0, h_start - int(1 * config.HDF5_SAMPLING_RATE))
        h_idx_end = min(len(data), h_start + int(5 * config.HDF5_SAMPLING_RATE))
        
        hdf5_ch1_segment = data[h_idx_start:h_idx_end, 1]  # Ch1: Block timing
        hdf5_time = (np.arange(h_idx_start, h_idx_end) - h_start) / config.HDF5_SAMPLING_RATE
        
        # Analog Ch4 (Stimulus copy)
        a_dat = a_trial['data']
        a_ch4 = normalize_arr(a_dat[4].astype(float))
        a_trig_idx = a_trial['local_start_sample']
        analog_time = (np.arange(len(a_ch4)) - a_trig_idx) / config.ANALOG_SAMPLING_RATE
        
        # Plot with offset
        offset = i * 1.2
        ax2.plot(hdf5_time, normalize_arr(hdf5_ch1_segment) + offset, 
                color=hdf5_ch1_color, linewidth=1.5,
                label='HDF5 Ch1 (Block Timing)' if i == 0 else "")
        ax2.plot(analog_time, a_ch4 + offset, 
                color=analog_ch4_color, linestyle='--', linewidth=1.2, alpha=0.8,
                label='Analog Ch4 (Stim)' if i == 0 else "")
        
        # Store y-tick positions and labels
        yticks_locs_2.append(offset + 0.5)
        yticks_labels_2.append(f'Trial {i}')
    
    ax2.set_title('Stimulus Timing Verification (HDF5 Ch1 vs Analog Ch4)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Time relative to Trigger (s)')
    ax2.set_ylabel('Trial')
    ax2.set_xlim(-1, 5)
    ax2.set_yticks(yticks_locs_2)
    ax2.set_yticklabels(yticks_labels_2)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, "alignment_verification_combined.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {os.path.basename(save_path)}")


def plot_full_reconstruction(data, analog_trials, trials_hdf5, output_dir, config=AlignmentConfig()):
    """Plot full timeline reconstruction with all trials."""
    print("\nGenerating full reconstruction plot...")
    
    plt.figure(figsize=(20, 8))
    
    hdf5_time_axis = np.arange(len(data)) / config.HDF5_SAMPLING_RATE
    plt.plot(hdf5_time_axis, normalize_arr(data[:, 3]), label='HDF5 Start (Ch3)', 
             color='black', alpha=0.3, linewidth=1)
    plt.plot(hdf5_time_axis, normalize_arr(data[:, 4]), label='HDF5 Stop (Ch4)', 
             color='red', alpha=0.3, linewidth=1)
    
    mapped_count = min(len(trials_hdf5), len(analog_trials))
    
    for i in range(mapped_count):
        h_trial = trials_hdf5[i]
        a_trial = analog_trials[i]
        
        t_trig_glob = h_trial['start_time_s']
        t_trig_loc_s = a_trial['local_start_sample'] / config.ANALOG_SAMPLING_RATE
        t_start_glob = t_trig_glob - t_trig_loc_s
        
        n_samples = a_trial['analog_data_shape'][1]
        analog_global_time = t_start_glob + (np.arange(n_samples) / config.ANALOG_SAMPLING_RATE)
        
        a_dat = a_trial['data']
        ch1_frames = normalize_arr(a_dat[1].astype(float))
        
        plt.plot(analog_global_time, ch1_frames + 1.2, color='blue', linewidth=0.5,
                label='Analog Frames (Ch1)' if i == 0 else "")
        
        ch3_start = normalize_arr(a_dat[3].astype(float))
        plt.plot(analog_global_time, ch3_start, color='lime', linestyle='--', linewidth=0.8,
                label='Analog Start (Ch3)' if i == 0 else "")
    
    plt.title("Full Alignment: Analog Frames vs HDF5 Start/Stop")
    plt.xlabel("Global Time (s)")
    plt.yticks([0.5, 1.7], ['Triggers', 'Frames'])
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')
    
    save_path = os.path.join(output_dir, "full_reconstruction.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {os.path.basename(save_path)}")


def plot_consolidated_matrix(full_matrix, output_dir, config=AlignmentConfig()):
    """Plot the complete consolidated matrix with color-coded channels."""
    print("\nGenerating consolidated matrix plot with color-coded channels...")
    
    plt.figure(figsize=(20, 12))
    
    t_global = np.arange(len(full_matrix)) / config.HDF5_SAMPLING_RATE
    n_channels = full_matrix.shape[1]
    
    yticks_locs = []
    yticks_labels = []
    
    # Plot all channels with consistent colors
    for i in range(min(n_channels, 19)):  # HDF5 (14) + Analog (5)
        sig = normalize_arr(full_matrix[:, i]) * 0.8
        color = CHANNEL_COLORS.get(i, 'grey')
        plt.plot(t_global, sig + i, linewidth=0.8, color=color, alpha=0.8)
        
        yticks_locs.append(i)
        if i < 14:
            yticks_labels.append(f"{i}: {config.CHANNEL_MAP.get(i, 'unknown')}")
        else:
            analog_ch = i - 14
            analog_names = ['timing', 'blue_led', 'violet_led', 'start_rx', 'stim']
            yticks_labels.append(f"A{analog_ch}: {analog_names[analog_ch]}")
    
    # Plot Blue trace if present
    if n_channels > 19:
        sig_blue = normalize_arr(full_matrix[:, 19]) * 1.5
        color = CHANNEL_COLORS.get(19, 'blue')
        plt.plot(t_global, sig_blue + 19, color=color, linewidth=1.2, label='Raw Blue Mean')
        yticks_locs.append(19)
        yticks_labels.append('Blue Trace')
    
    plt.yticks(yticks_locs, yticks_labels, fontsize=8)
    plt.xlabel('Global Time (s)')
    plt.title('Consolidated Alignment Matrix (All Channels - Color Coded)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, "consolidated_matrix.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {os.path.basename(save_path)}")


def plot_blue_behavior_overlay(full_matrix, trials_hdf5, output_dir, config=AlignmentConfig()):
    """Plot overlay of blue trace with behavior signals - separate plot per trial."""
    print("\nGenerating blue-behavior overlay plots (one per trial)...")
    
    if full_matrix.shape[1] < 20:
        print("  Skipping: Blue trace not available")
        return
    
    t_global = np.arange(len(full_matrix)) / config.HDF5_SAMPLING_RATE
    
    # Get channel colors
    blue_color = CHANNEL_COLORS.get(19, '#000080')
    photodiode_color = CHANNEL_COLORS.get(6, '#e377c2')
    rotary_color = CHANNEL_COLORS.get(7, '#7f7f7f')
    
    # Plot each trial separately
    for trial_idx, trial in enumerate(trials_hdf5):
        fig, ax = plt.subplots(figsize=(15, 5))
        
        # Get trial time window
        trial_start = trial['start_time_s']
        trial_stop = trial['stop_sample'] / config.HDF5_SAMPLING_RATE
        
        # Add some padding
        window_start = max(0, trial_start - 2)  # 2s before
        window_stop = min(t_global[-1], trial_stop + 2)  # 2s after
        
        # Find indices for this window
        idx_start = int(window_start * config.HDF5_SAMPLING_RATE)
        idx_stop = int(window_stop * config.HDF5_SAMPLING_RATE)
        
        t_window = t_global[idx_start:idx_stop]
        
        # Blue trace
        blue_trace = normalize_arr(full_matrix[idx_start:idx_stop, 19])
        ax.plot(t_window, blue_trace, color=blue_color, alpha=0.9, 
                label='Raw Blue Mean', linewidth=1.5)
        
        # Photodiode (Ch6)
        ch6 = normalize_arr(full_matrix[idx_start:idx_stop, 6])
        ax.plot(t_window, ch6, color=photodiode_color, alpha=0.7, 
                label='Photodiode (Ch6)', linewidth=1)
        
        # Rotary Z (Ch7)
        ch7 = normalize_arr(full_matrix[idx_start:idx_stop, 7])
        ax.plot(t_window, ch7, color=rotary_color, alpha=0.7, 
                label='Rotary Z (Ch7)', linewidth=1)
        
        # Mark trial boundaries
        ax.axvline(trial_start, color='green', linestyle='--', alpha=0.5, 
                  linewidth=2, label='Trial Start')
        ax.axvline(trial_stop, color='red', linestyle='--', alpha=0.5, 
                  linewidth=2, label='Trial Stop')
        
        ax.set_title(f"Trial {trial_idx}: Blue Frame Intensity vs Behavior Signals", 
                    fontsize=12, fontweight='bold')
        ax.set_xlabel("Global Time (s)")
        ax.set_ylabel("Normalized Amplitude")
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        save_path = os.path.join(output_dir, f"blue_behavior_trial_{trial_idx}.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"  Saved: {os.path.basename(save_path)}")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_alignment_pipeline(data_dir, output_dir=None, verbose=True):
    """
    Run the complete alignment pipeline.
    
    Args:
        data_dir: Directory containing all data files
        output_dir: Output directory (default: data_dir/alignment)
        verbose: Print progress information
        
    Returns:
        Dictionary with results including final matrix
    """
    config = AlignmentConfig()
    
    # Setup output directory
    if output_dir is None:
        output_dir = os.path.join(data_dir, "alignment")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        if verbose:
            print(f"Created output directory: {output_dir}")
    
    print("\n" + "="*70)
    print("WIDEFIELD-HDF5 ALIGNMENT PIPELINE")
    print("="*70)
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    
    # Step 1: Load HDF5 data
    print("\n" + "="*70)
    print("STEP 1: Loading HDF5 Synchronization Data")
    print("="*70)
    
    hdf5_files = glob.glob(os.path.join(data_dir, "*.hdf5")) + \
                 glob.glob(os.path.join(data_dir, "*.h5"))
    
    if not hdf5_files:
        raise ValueError("No HDF5 file found in directory")
    
    hdf5_path = hdf5_files[0]
    data = load_hdf5_data(hdf5_path, verbose=verbose)
    
    # Step 2: Identify HDF5 trials
    print("\n" + "="*70)
    print("STEP 2: Identifying HDF5 Trial Boundaries")
    print("="*70)
    
    trials_hdf5 = identify_hdf5_trials(data, config=config, verbose=verbose)
    
    # Step 3: Load Analog files
    print("\n" + "="*70)
    print("STEP 3: Loading Analog.dat Files")
    print("="*70)
    
    analog_trials = load_all_analog_files(data_dir, config=config, verbose=verbose)
    
    # Step 4: Create consolidated matrix (HDF5 + Analog)
    print("\n" + "="*70)
    print("STEP 4: Creating Consolidated Matrix (HDF5 + Analog)")
    print("="*70)
    
    full_matrix = create_consolidated_matrix(data, analog_trials, trials_hdf5, 
                                             config=config, verbose=verbose)
    
    # Step 5: Load and process frame files
    print("\n" + "="*70)
    print("STEP 5: Loading Frame Files")
    print("="*70)
    
    frames_files = glob.glob(os.path.join(data_dir, "Frames_*.dat"))
    frames_files.sort(key=get_file_index)
    
    frameTimes_files = glob.glob(os.path.join(data_dir, "*frameTimes*.mat"))
    frameTimes_files.sort(key=get_file_index)
    
    if verbose:
        print(f"  Found {len(frames_files)} Frames files")
        print(f"  Found {len(frameTimes_files)} frameTimes files")
    
    # Step 6: Process frames with offset (THE KEY STEP)
    global_blue_times, global_blue_vals = process_frames_with_offset(
        frames_files, frameTimes_files, analog_trials, trials_hdf5, 
        config=config, verbose=verbose
    )
    
    # Step 7: Add blue trace to matrix
    print("\n" + "="*70)
    print("STEP 7: Adding Blue Trace to Matrix")
    print("="*70)
    
    if global_blue_times and global_blue_vals:
        final_matrix = add_blue_trace_to_matrix(full_matrix, global_blue_times, 
                                                global_blue_vals, config=config, verbose=verbose)
    else:
        if verbose:
            print("  Warning: No blue trace data available")
        final_matrix = full_matrix
    
    # Step 8: Save matrix
    print("\n" + "="*70)
    print("STEP 8: Saving Results")
    print("="*70)
    
    npy_path = os.path.join(output_dir, "aligned_full_matrix.npy")
    np.save(npy_path, final_matrix)
    if verbose:
        print(f"  Saved matrix: {npy_path}")
        print(f"  Matrix shape: {final_matrix.shape}")
    
    # Save channel names
    col_names = [config.CHANNEL_MAP.get(i, f"h{i}") for i in range(14)] + \
                ["a0_timing", "a1_blue_led", "a2_violet_led", "a3_start", "a4_stim"]
    if final_matrix.shape[1] > 19:
        col_names.append("blue_trace_mean")
    
    names_path = os.path.join(output_dir, "channel_names.txt")
    with open(names_path, 'w') as f:
        for idx, name in enumerate(col_names):
            f.write(f"{idx}: {name}\n")
    if verbose:
        print(f"  Saved channel names: {names_path}")
    
    # Step 9: Generate visualizations
    print("\n" + "="*70)
    print("STEP 9: Generating Visualizations")
    print("="*70)
    
    plot_trigger_alignment(data, analog_trials, trials_hdf5, output_dir, config=config)
    plot_full_reconstruction(data, analog_trials, trials_hdf5, output_dir, config=config)
    plot_consolidated_matrix(final_matrix, output_dir, config=config)
    
    if final_matrix.shape[1] > 19:
        plot_blue_behavior_overlay(final_matrix, trials_hdf5, output_dir, config=config)
    
    # Summary
    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print("="*70)
    print(f"Final matrix shape: {final_matrix.shape}")
    print(f"  - Channels: {final_matrix.shape[1]}")
    print(f"  - Samples: {final_matrix.shape[0]}")
    print(f"  - Duration: {final_matrix.shape[0] / config.HDF5_SAMPLING_RATE:.2f} seconds")
    print(f"Output directory: {output_dir}")
    print("="*70)
    
    return {
        'matrix': final_matrix,
        'hdf5_data': data,
        'trials_hdf5': trials_hdf5,
        'analog_trials': analog_trials,
        'channel_names': col_names,
        'config': config,
        'output_dir': output_dir
    }


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Widefield-HDF5 Alignment Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    python alignment_pipeline.py  # Uses global DATA_DIR variable
    python alignment_pipeline.py --data_dir "D:\\Data\\experiment_folder"
    python alignment_pipeline.py --data_dir "D:\\Data\\experiment_folder" --output_dir "D:\\Output"
        """
    )
    
    parser.add_argument('--data_dir', type=str, default=DATA_DIR,
                       help=f'Directory containing HDF5, Analog, and Frames files (default: {DATA_DIR})')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR,
                       help='Output directory (default: data_dir/alignment)')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress verbose output')
    
    args = parser.parse_args()
    
    try:
        results = run_alignment_pipeline(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            verbose=not args.quiet
        )
        print("\nSuccess! Check the output directory for results.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
