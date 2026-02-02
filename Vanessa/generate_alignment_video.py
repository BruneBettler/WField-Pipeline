
"""
Alignment Video Generator
=========================
Generates a side-by-side video verification of the Widefield-Eye synchronization.
It overlays:
- Preprocessed Widefield Frames (dF/F)
- Eye Camera Video
- Trial Information
- Synchronization Metrics

Usage:
    python generate_alignment_video.py --data_dir "D:/path/to/data" --output_dir "D:/path/to/output"
"""

import os
import argparse
import numpy as np
import h5py
import cv2
import tqdm
import glob
import bisect

# ============================================================================
# CONFIGURATION
# ============================================================================

# Default logic constants
COL_EYE_CAMERA = 5
COL_FRAME_INDEX = 21
COL_TRIAL_INDEX = 20

# Frame size for raw validation (uint16, 2 channels, 640x540)
FRAME_SIZE_BYTES = 2 * 640 * 540 * 2  

DEFAULT_DATA_DIR = r"D:\Vanessa_test_data\Tests_Jan23\23-Jan-2026_ledTTL_10random"


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def normalize_frame(frame, p_min=1, p_max=99):
    """Normalize frame to 0-255 range based on percentiles."""
    frame = np.nan_to_num(frame)
    vmin, vmax = np.percentile(frame, [p_min, p_max])
    if vmax == vmin:
        return np.zeros_like(frame, dtype=np.uint8)
    norm = (frame - vmin) / (vmax - vmin)
    norm = np.clip(norm, 0, 1)
    return (norm * 255).astype(np.uint8)

def get_trial_frame_counts(data_dir):
    """Scan Frames_*.dat files to determine frames per trial."""
    dat_files = glob.glob(os.path.join(data_dir, "Frames_*.dat"))
    dat_files.sort(key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0]))
    
    counts = []
    print(f"Scanning {len(dat_files)} .dat files...")
    for fpath in dat_files:
        size = os.path.getsize(fpath)
        # File contains N timepoints (each timepoint = 2 channels * 640 * 540 * uint16)
        n_raw = size // FRAME_SIZE_BYTES
        counts.append(n_raw)
    return counts, dat_files


# ============================================================================
# MAIN
# ============================================================================

def generate_video(data_dir, output_dir=None, output_filename="alignment_side_by_side.mp4", 
                   cmap_name='jet', p_min=1, p_max=99):
    
    # Map string names to OpenCV Colormaps
    CMAPS = {
        'jet': cv2.COLORMAP_JET,
        'viridis': cv2.COLORMAP_VIRIDIS,
        'plasma': cv2.COLORMAP_PLASMA,
        'inferno': cv2.COLORMAP_INFERNO,
        'magma': cv2.COLORMAP_MAGMA,
        'hot': cv2.COLORMAP_HOT,
        'bone': cv2.COLORMAP_BONE,
        'ocean': cv2.COLORMAP_OCEAN,
        'cool': cv2.COLORMAP_COOL,
        'gray': None # Special case
    }
    
    if cmap_name.lower() not in CMAPS:
        print(f"Warning: Colormap '{cmap_name}' not found. Using 'jet'. Options: {list(CMAPS.keys())}")
        cmap_name = 'jet'
    
    selected_cmap = CMAPS[cmap_name.lower()]

    alignment_dir = os.path.join(data_dir, "alignment")
    preprocessed_dir = os.path.join(data_dir, "preprocessed_data")
    matrix_path = os.path.join(alignment_dir, "aligned_full_matrix.npy")
    
    # Load Brain Mask
    brain_mask = None
    mask_candidates = glob.glob(os.path.join(preprocessed_dir, "*_full_mask.npy")) + \
                      glob.glob(os.path.join(preprocessed_dir, "brain_mask.npy"))
    if mask_candidates:
        mask_path = mask_candidates[0]
        try:
            brain_mask = np.load(mask_path).astype(bool)
            print(f"Loaded mask/region for background suppression: {os.path.basename(mask_path)}")
        except Exception as e:
            print(f"Warning: Failed to load mask {mask_path}: {e}")
    else:
        print("Warning: No mask file found. Background will not be suppressed.")
    
    # Output path logic
    if output_dir is None:
        # Default to 'alignment_verification' subfolder inside data_dir
        output_dir = os.path.join(data_dir, "alignment_verification")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    output_path = os.path.join(output_dir, output_filename)

    if os.path.exists(output_path):
        print(f"\nAlignment verification video already exists at: {output_path}")
        print("Skipping video generation step.")
        return

    # 1. Validation
    if not os.path.exists(matrix_path):
        raise FileNotFoundError(f"Alignment matrix not found at {matrix_path}")
        
    print(f"\n--- Generating Alignment Verification Video ---")
    print(f"Data Directory: {data_dir}")
    print(f"Output: {output_path}")

    # 2. Find Files
    # HDF5
    h5_files = [f for f in os.listdir(preprocessed_dir) if f.endswith('.h5py') or f.endswith('.h5')]
    if not h5_files:
        raise FileNotFoundError("No widefield HDF5 file found in preprocessed_data")
    wf_h5_path = os.path.join(preprocessed_dir, h5_files[0])
    
    # MP4
    mp4_files = [f for f in os.listdir(data_dir) if f.startswith("eyecam") and f.endswith('.mp4')]
    if not mp4_files:
        raise FileNotFoundError("No eyecam MP4 video found in data directory")
    eye_video_path = os.path.join(data_dir, mp4_files[0])

    # 3. Load Matrix
    print("Loading aligned matrix...")
    matrix = np.load(matrix_path, mmap_mode='r')
    eye_camera_signal = matrix[:, COL_EYE_CAMERA]
    frame_indices = matrix[:, COL_FRAME_INDEX]
    trial_indices = matrix[:, COL_TRIAL_INDEX]

    # 4. Eye Camera Offset Calculation
    print("Detecting eye camera triggers...")
    thresh = 2.5 if eye_camera_signal.max() > 1.5 else 0.5
    triggers = (eye_camera_signal > thresh).astype(int)
    rising_edges = np.where(np.diff(triggers) == 1)[0]
    n_triggers = len(rising_edges)
    
    print(f"Opening eye video: {os.path.basename(eye_video_path)}")
    eye_cap = cv2.VideoCapture(eye_video_path)
    eye_n_frames = int(eye_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    eye_width = int(eye_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    eye_height = int(eye_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"  Triggers: {n_triggers}, Frames: {eye_n_frames}")
    # BACKWARD OFFSET: Last Trigger <-> Last Frame
    eye_offset = eye_n_frames - n_triggers
    print(f"  Calculated Offset (Frames - Triggers): {eye_offset}")

    # 5. Trial Mapping
    trial_counts, _ = get_trial_frame_counts(data_dir)
    print(f"  Trials found: {len(trial_counts)}")

    # 6. Open Widefield HDF5
    print(f"Opening widefield data: {os.path.basename(wf_h5_path)}")
    hf = h5py.File(wf_h5_path, 'r')
    
    # Try multiple possible dataset paths in order of preference
    possible_paths = [
        'led/deltaF',
        'led/hemo_corrected',
        'led/motion_corrected',
        'led/raw_frames',
        'led/data',
        'widefield/deltaF',
        'widefield/data'
    ]
    
    wf_dset = None
    dataset_name = None
    
    for p in possible_paths:
        if p in hf:
            wf_dset = hf[p]
            dataset_name = p
            print(f"  Using dataset: {p}")
            break
            
    if wf_dset is None:
        available = []
        hf.visit(lambda name: available.append(name) if isinstance(hf[name], h5py.Dataset) else None)
        raise ValueError(f"Could not find valid widefield dataset. Checked {possible_paths}. Available: {available}")
        
    n_wf_frames, wf_height, wf_width = wf_dset.shape
    print(f"  Processed Frames: {n_wf_frames}")

    # 7. Setup Output
    out_width = wf_width + eye_width
    out_height = max(wf_height, eye_height)
    print(f"Output Resolution: {out_width}x{out_height}")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(output_path, fourcc, 20.0, (out_width, out_height))
    
    # 8. Build Alignment Map
    print("Building alignment index map...")
    trial_start_indices = [0]
    cumulative = 0
    for c in trial_counts:
        cumulative += c
        trial_start_indices.append(cumulative)
        
    valid_idx_mask = ~np.isnan(frame_indices) & ~np.isnan(trial_indices)
    valid_samples = np.where(valid_idx_mask)[0]
    t_vals = trial_indices[valid_samples].astype(int)
    f_vals = frame_indices[valid_samples].astype(int)
    
    # Key: Trial * 100000 + Frame
    keys = t_vals.astype(np.uint64) * 100000 + f_vals.astype(np.uint64)
    u_keys, u_indices = np.unique(keys, return_index=True)
    global_u_indices = valid_samples[u_indices]
    
    print("Map built. Starting video generation...")
    
    last_eye_frame = None
    current_eye_pos = -1

    alignment_indices = []

    for global_idx in tqdm.tqdm(range(n_wf_frames)):
        
        # Determine Trial
        t_idx = bisect.bisect_right(trial_start_indices, global_idx) - 1
        if t_idx >= len(trial_counts): break
        
        local_idx = global_idx - trial_start_indices[t_idx]
        target_raw = local_idx * 2  # Stride correction (10Hz vs 20Hz)
        
        # Lookup Timestamp
        search_key = int(t_idx) * 100000 + int(target_raw)
        k_idx = np.searchsorted(u_keys, search_key)
        
        has_sync = False
        timestamp_sample = None
        
        if k_idx < len(u_keys) and u_keys[k_idx] == search_key:
            timestamp_sample = global_u_indices[k_idx]
            has_sync = True
            
        # Get Widefield
        raw_frame = wf_dset[global_idx]
        
        # Capture NaNs for fallback masking
        nan_mask = np.isnan(raw_frame)
        
        w_img = normalize_frame(raw_frame, p_min=p_min, p_max=p_max)
        
        # Apply Colormap
        if selected_cmap is not None:
            w_color = cv2.applyColorMap(w_img, selected_cmap)
        else:
            w_color = cv2.cvtColor(w_img, cv2.COLOR_GRAY2BGR)

        # Apply Brain Mask (force background to black)
        if brain_mask is not None:
             # Handle shape mismatch if any
             if brain_mask.shape == w_color.shape[:2]:
                 w_color[~brain_mask] = 0
             else:
                 # Try resizing mask? Or just skip
                 pass
        
        # Fallback: Apply NaN mask from data itself
        if np.any(nan_mask):
            w_color[nan_mask] = 0

        # Get Eye Frame
        e_img = None
        eye_idx_display = -1
        
        if has_sync:
            n_trigues_before = np.searchsorted(rising_edges, timestamp_sample)
            idx_in_trig = n_trigues_before - 1
            target_eye = idx_in_trig + eye_offset if idx_in_trig >= 0 else -1
            eye_idx_display = target_eye
            
            if target_eye < 0:
                 e_img = np.zeros((eye_height, eye_width, 3), dtype=np.uint8)
            elif target_eye >= eye_n_frames:
                 e_img = last_eye_frame if last_eye_frame is not None else np.zeros((eye_height, eye_width, 3), dtype=np.uint8)
            else:
                if target_eye != current_eye_pos:
                    eye_cap.set(cv2.CAP_PROP_POS_FRAMES, target_eye)
                    ret, fr = eye_cap.read()
                    if ret:
                        e_img = fr
                        last_eye_frame = fr
                        current_eye_pos = target_eye
                    else:
                        e_img = np.zeros((eye_height, eye_width, 3), dtype=np.uint8)
                else:
                    e_img = last_eye_frame
        else:
            e_img = np.zeros((eye_height, eye_width, 3), dtype=np.uint8)
            cv2.putText(e_img, "NO SYNC", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            
        # Composite
        canvas = np.zeros((out_height, out_width, 3), dtype=np.uint8)
        canvas[:wf_height, :wf_width] = w_color
        if e_img is not None:
             h_e, w_e = e_img.shape[:2]
             canvas[:h_e, wf_width:wf_width+w_e] = e_img
             
        # Text
        txt1 = f"Trial: {t_idx} | WF: {local_idx}"
        txt2 = f"Eye: {eye_idx_display}" if has_sync else "Eye: --"
        cv2.putText(canvas, txt1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(canvas, txt2, (wf_width+10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        out_writer.write(canvas)
        alignment_indices.append([global_idx, eye_idx_display])
        
    out_writer.release()
    eye_cap.release()
    hf.close()

    # Save alignment indices
    alignment_arr = np.array(alignment_indices).T
    base_name = os.path.splitext(output_filename)[0]
    idx_path = os.path.join(output_dir, f"{base_name}_indices.npy")
    np.save(idx_path, alignment_arr)
    print(f"Alignment indices saved to {idx_path}")

    # Save text description
    desc_path = os.path.join(output_dir, f"{base_name}_indices_README.txt")
    with open(desc_path, 'w') as f:
        f.write("Row 0: Widefield Frame Index\n")
        f.write("Row 1: Eye Camera Frame Index (or -1 if no sync)\n")
    print(f"Indices description saved to {desc_path}")

    print(f"\nDone. Saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Alignment Video")
    parser.add_argument('--data_dir', type=str, required=True, help="Path to data directory")
    parser.add_argument('--output_dir', type=str, default=None, help="Directory to save video (defaults to data_dir/alignment_verification)")
    parser.add_argument('--filename', type=str, default="alignment_side_by_side.mp4", help="Output filename")
    parser.add_argument('--cmap', type=str, default='jet', help="Colormap (jet, viridis, plasma, etc.)")
    parser.add_argument('--pmin', type=float, default=1.0, help="Min percentile for normalization")
    parser.add_argument('--pmax', type=float, default=99.0, help="Max percentile for normalization")
    
    args = parser.parse_args()
    try:
        generate_video(args.data_dir, args.output_dir, args.filename, 
                       cmap_name=args.cmap, p_min=args.pmin, p_max=args.pmax)
    except Exception as e:
        print(f"Error: {e}")
