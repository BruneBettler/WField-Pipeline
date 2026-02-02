
import sys
import os
import glob
import h5py
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from tqdm import tqdm
# from PyQt5.QtWidgets import QApplication
# from PyQt5.QtCore import Qt

# ==========================================
# Setup Paths
# ==========================================

def setup_paths():
    """Add Nicole directory to sys.path"""
    # Robustly find the location of this script
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        # Fallback for interactive environments
        current_dir = os.getcwd()

    # Look for Nicole folder relative to this script
    # 1. Hardcoded workspace path (User's specific path)
    workspace_nicole = r"c:\Users\mlouki1\Desktop\WField-Pipeline\Nicole"
    
    # 2. Sibling?
    sibling_nicole = os.path.abspath(os.path.join(current_dir, '..', 'Nicole'))
    
    nicole_path = None
    if os.path.exists(workspace_nicole):
        nicole_path = workspace_nicole
    elif os.path.exists(sibling_nicole):
        nicole_path = sibling_nicole
    
    if nicole_path and os.path.exists(nicole_path):
        if nicole_path not in sys.path:
            sys.path.insert(0, nicole_path)
    else:
        raise FileNotFoundError(f"Could not locate 'Nicole' library folder. Checked: {sibling_nicole} and {workspace_nicole}")
setup_paths()

from pipeline_utils import create_hdf5, normalize_arr, get_trial_index, check_blue_is_first_via_intensity
from pipeline_processing import hdf5_motion_correct, hemodynamic_correction, get_deltaF
# from GUIs.registration_gui import RegistrationGUI
from visualization_utils import save_video_of_stack, plot_global_trace, load_dat_analog

# ==========================================
# Pipeline Steps
# ==========================================

def step1_load_raw(hdf5_file_path, exp_path, dataset_type, overwrite_raw):
    print("\n--- Step 1: Load and Concatenate Raw Frames ---")
    
    with h5py.File(hdf5_file_path, 'a') as f:
        if f"{dataset_type}" in f:
            print(f"Group {dataset_type} already exists.")
            hdf5_group = f[f"{dataset_type}"]
        else:
            print(f"Creating group {dataset_type}.")
            hdf5_group = f.create_group(f"{dataset_type}")

        if 'raw_frames' in hdf5_group and not overwrite_raw:
            print("raw_frames dataset already exists. Skipping load.")
            return

        if 'raw_frames' in hdf5_group and overwrite_raw:
             print("Overwriting existing raw_frames...")
             del hdf5_group['raw_frames']

        # Find .dat files
        dat_pattern = os.path.join(exp_path, "Frames_*.dat")
        dat_files = sorted(glob.glob(dat_pattern))
        
        if not dat_files:
            raise FileNotFoundError(f"No .dat files found in {exp_path}")
        
        print(f"Found {len(dat_files)} trial files.")
        
        # Parse metadata
        first_file = dat_files[0]
        dat_info = os.path.basename(first_file).split('_')
        try:
            channels = int(dat_info[1])
            H = int(dat_info[2])
            W = int(dat_info[3])
            dtype_str = dat_info[4]
            dtype = np.dtype(dtype_str)
        except IndexError:
             print("Filename format unexpected, using defaults.")
             channels = 2
             H = 640
             W = 540
             dtype = np.uint16

        frame_size_bytes = H * W * channels * dtype.itemsize
        
        dset = hdf5_group.create_dataset(
            'raw_frames', 
            shape=(0, channels, H, W), 
            maxshape=(None, channels, H, W), 
            dtype=dtype, 
            chunks=(min(100, 500), channels, H, W)
        )
        
        total_frames = 0
        for dat_file in tqdm(dat_files, desc="Loading trials"):
            file_size = os.path.getsize(dat_file)
            n_frames_in_file = file_size // frame_size_bytes
            
            data = np.memmap(dat_file, dtype=dtype, mode='r', shape=(n_frames_in_file, channels, H, W))
            dset.resize((total_frames + n_frames_in_file, channels, H, W))
            
            # Check frame order using Image Intensity (More robust than Analog sync)
            try:
                 is_blue_first = check_blue_is_first_via_intensity(dat_file, H, W, dtype, channels)
                 if not is_blue_first:
                      print(f"  [Trial {get_trial_index(dat_file)}] Detected Violet-First via intensity check. Swapping channels.")
            except Exception as e:
                 print(f"Warning: Intensity check failed: {e}. Defaulting to Blue First.")
                 is_blue_first = True
            
            if is_blue_first:
                dset[total_frames : total_frames + n_frames_in_file] = data
            else:
                # Swap channels: Channel 1 (Violet) is currently at index 0
                dset[total_frames : total_frames + n_frames_in_file] = data[:, ::-1, :, :]
                
            total_frames += n_frames_in_file
            del data
            
        print(f"Total frames loaded: {total_frames}")

def step2_motion_correction(hdf5_file_path, dataset_type, mc_nreference, mc_chunksize):
    print("\n--- Step 2: Motion Correction ---")
    
    with h5py.File(hdf5_file_path, 'a') as f:
        hdf5_group = f[f"{dataset_type}"]
        
        if 'motion_corrected' in hdf5_group:
            print("Motion corrected frames already exist.")
            return

        print("Starting motion correction...")
        num_to_remove = 0 
        input_dset = hdf5_group["raw_frames"]
        output_shape = (input_dset.shape[0] - num_to_remove, *input_dset.shape[1:])
        
        motion_corrected_dataset = hdf5_group.create_dataset(
            'motion_corrected', 
            shape=output_shape, 
            dtype=input_dset.dtype
        )
        
        hdf5_motion_correct(
            input_dset, 
            motion_corrected_dataset, 
            nreference=mc_nreference, 
            chunksize=mc_chunksize, 
            dark_frames_to_remove=num_to_remove
        )
        print("Done motion correction")

def step3_and_4_masking(hdf5_file_path, hdf5_creation_folder_path, dataset_type):
    print("\n--- Step 3 & 4: Masking ---")
    
    # Check for existing masks
    # 1. Search in preprocessed_data folder
    mask_files = glob.glob(os.path.join(hdf5_creation_folder_path, "*_full_mask.npy"))
    mask_files += glob.glob(os.path.join(hdf5_creation_folder_path, "brain_mask.npy"))
    
    # 2. Search in parent/experiment directory
    exp_dir = os.path.dirname(hdf5_creation_folder_path)
    mask_files += glob.glob(os.path.join(exp_dir, "*_full_mask.npy"))
    mask_files += glob.glob(os.path.join(exp_dir, "brain_mask.npy"))
    
    # Remove duplicates and sort by newest
    mask_files = sorted(list(set(mask_files)), key=os.path.getmtime, reverse=True)
    
    if mask_files:
        path_to_full_mask = mask_files[0]
        print(f"Found existing mask: {path_to_full_mask}")
    else:
        print("No mask found. Stopping pipeline for manual masking.")
        print(f"Checked: {hdf5_creation_folder_path} and {exp_dir}")
        print(f"Please use the Launcher's 'Draw Mask' button to create: brain_mask.npy")
        # Exit with a specific status or raise error to stop pipeline
        raise RuntimeError("MASK_MISSING")

    return path_to_full_mask

def step5_hemo_correction(hdf5_file_path, path_to_full_mask, dataset_type, hm_highpass):
    print("\n--- Step 5: Hemodynamic Correction ---")
    
    with h5py.File(hdf5_file_path, 'a') as f:
        hdf5_group = f[f"{dataset_type}"]
        if 'hemo_corrected' in hdf5_group:
            print("Hemodynamic corrected frames already exist.")
            return

        print("Starting hemodynamic correction...")
        full_mask = np.load(path_to_full_mask)
        input_dset = hdf5_group['motion_corrected']
        new_shape = (input_dset.shape[0], *input_dset.shape[2:])
        
        # Initialize output dataset with NaNs (so gaps are automatically handled)
        # Note: We must create it, then fill with NaNs. 
        hemoCorrected_dataset = hdf5_group.create_dataset(
            'hemo_corrected', 
            shape=new_shape, 
            dtype='float64', 
            chunks=(200, *input_dset.shape[2:]), 
            compression=None
        )
        
        # Fill with NaNs (by chunks for memory)
        print("Initializing output with NaNs...")
        for i in range(0, new_shape[0], 1000):
            hemoCorrected_dataset[i:i+1000] = np.nan

        # 1. SCAN FOR VALID TRIALS
        print("Scanning for valid trials to avoid filter artifacts...")
        n_frames = input_dset.shape[0]
        blue_means = []
        scan_chunk = 500
        for i in range(0, n_frames, scan_chunk):
            chunk = input_dset[i:min(i+scan_chunk, n_frames), 0, :, :]
            blue_means.append(np.mean(chunk, axis=(1,2)))
        blue_means = np.concatenate(blue_means)
        
        # Define valid frames
        DARK_THRESHOLD = 9000
        is_valid = blue_means > DARK_THRESHOLD
        
        # Group contiguous indices into (start, end) tuples
        valid_indices = np.where(is_valid)[0]
        if len(valid_indices) == 0:
            print("Error: No valid data found above threshold.")
            return

        from itertools import groupby
        from operator import itemgetter
        
        ranges = []
        for k, g in groupby(enumerate(valid_indices), lambda ix: ix[0] - ix[1]):
            group = list(map(itemgetter(1), g))
            # Tuple: (start_index, end_index_exclusive)
            ranges.append((group[0], group[-1] + 1))
            
        print(f"Found {len(ranges)} valid segments/trials.")
        
        # 2. PROCESS EACH TRIAL INDIVIDUALLY
        for start, end in tqdm(ranges, desc="Processing Trials"):
            # A. Extract Trial Data (NumPy Array)
            trial_data = input_dset[start:end]
            
            if trial_data.shape[0] < 10:
                print(f"Skipping tiny fragment at {start}-{end}")
                continue

            # B. Run Hemo Correction
            try:
                corrected_trial = hemodynamic_correction(trial_data, output_dataset=None, binary_npy_mask=full_mask, highpass=hm_highpass, verbose=False)
                # C. Write back
                hemoCorrected_dataset[start:end] = corrected_trial
            except Exception as e:
                print(f"Error processing trial {start}-{end}: {e}")

        print(f"Done hemodynamic correction. Processed {len(ranges)} valid intervals.")

def step6_deltaF(hdf5_file_path, dataset_type):
    print("\n--- Step 6: Delta F / F ---")
    
    with h5py.File(hdf5_file_path, 'a') as f:
        hdf5_group = f[f"{dataset_type}"]
        if 'deltaF' in hdf5_group:
             print("deltaF frames already exist.")
             return

        print("Calculating deltaF...")
        
        if 'hemo_corrected' not in hdf5_group:
            print("Error: hemo_corrected dataset required for deltaF step.")
            return
            
        dF_dset = hdf5_group['hemo_corrected']
        F0_source_dset = hdf5_group['motion_corrected']
        n_frames = dF_dset.shape[0]
        
        # 1. Calculate Baseline F0
        print("Calculating baseline F0 (using valid frames)...")
        
        sum_F0 = np.zeros(dF_dset.shape[1:], dtype=np.float64)
        valid_frame_count = 0
        chunk_size = 500
        
        for i in tqdm(range(0, n_frames, chunk_size), desc="Baseline F0"):
            chunk_end = min(i + chunk_size, n_frames)
            
            dF_chunk = dF_dset[i:chunk_end]
            middle_pixel = dF_chunk[:, dF_chunk.shape[1]//2, dF_chunk.shape[2]//2]
            valid_mask = ~np.isnan(middle_pixel)
            
            if np.any(valid_mask):
                raw_chunk = F0_source_dset[i:chunk_end, 0, :, :]
                sum_F0 += np.sum(raw_chunk[valid_mask], axis=0)
                valid_frame_count += np.sum(valid_mask)
            
        if valid_frame_count == 0:
            print("Warning: No valid frames found for F0!")
            mean_F0 = np.ones(dF_dset.shape[1:], dtype=np.float64)
        else:
            mean_F0 = sum_F0 / valid_frame_count
        
        mean_F0[mean_F0 == 0] = 1.0 
        
        # 2. Calculate dF/F
        deltaF_dataset = hdf5_group.create_dataset(
            'deltaF', 
            shape=dF_dset.shape, 
            dtype='float64',
            chunks=dF_dset.chunks
        )
        
        print("Computing dF/F ...")
        for i in tqdm(range(0, n_frames, chunk_size), desc="Writing deltaF"):
            chunk_end = min(i + chunk_size, n_frames)
            
            dF_chunk = dF_dset[i:chunk_end]
            output_chunk = dF_chunk / mean_F0
            deltaF_dataset[i:chunk_end] = output_chunk
        
        print("Done deltaF")

def step7_visualization(hdf5_file_path, output_folder, dataset_type):
    print("\n--- Step 7: Visualization ---")
    
    with h5py.File(hdf5_file_path, 'r') as f:
        datasets_to_plot = []
        
        if 'raw_frames' in f[dataset_type]:
            datasets_to_plot.append('raw_frames')
            
        if 'deltaF' in f[dataset_type]:
            processed_dset_name = 'deltaF'
            datasets_to_plot.append('deltaF')
        elif 'hemo_corrected' in f[dataset_type]:
            processed_dset_name = 'hemo_corrected'
            datasets_to_plot.append('hemo_corrected')
        else:
            processed_dset_name = None
            
        if not datasets_to_plot:
            print("No data found for visualization.")
            return

        for dset_name in datasets_to_plot:
            dset = f[dataset_type][dset_name]
            print(f"Calculating global mean trace for {dset_name}...")
            
            n_frames = dset.shape[0]
            mean_trace = np.zeros(n_frames)
            chunk_size = 1000
            nan_warning_count = 0
            
            for i in tqdm(range(0, n_frames, chunk_size), desc=f"Trace ({dset_name})"):
                chunk_end = min(i + chunk_size, n_frames)
                
                if dset.ndim == 3:
                    chunk = dset[i:chunk_end, :, :]
                    batch_means = np.nanmean(chunk, axis=(1, 2))
                elif dset.ndim == 4:
                    chunk = dset[i:chunk_end, 0, :, :]
                    batch_means = np.nanmean(chunk, axis=(1, 2))
                    
                mean_trace[i:chunk_end] = batch_means
                if np.isnan(batch_means).any():
                    nan_warning_count += np.sum(np.isnan(batch_means))

            if nan_warning_count > 0:
                print(f"Warning: {nan_warning_count} frames had NaN means in {dset_name}.")

            trace_path = os.path.join(output_folder, f"global_trace_{dset_name}.png")
            ylabel = "Pixel Value" if dset_name == 'raw_frames' else "dF/F"
            plot_global_trace(mean_trace, trace_path, title=f"Global Average Trace ({dset_name})", ylabel=ylabel)

        if 'motion_corrected' in f[dataset_type]:
            print("Comparing Blue vs Violet traces...")
            mc_dset = f[dataset_type]['motion_corrected']
            n_frames = mc_dset.shape[0]
            blue_trace = np.zeros(n_frames)
            violet_trace = np.zeros(n_frames)
            chunk_size = 1000
            for i in tqdm(range(0, n_frames, chunk_size), desc="Trace (Blue/Violet)"):
                chunk_end = min(i + chunk_size, n_frames)
                chunk = mc_dset[i:chunk_end]
                means = np.nanmean(chunk, axis=(2, 3))
                blue_trace[i:chunk_end] = means[:, 0]
                violet_trace[i:chunk_end] = means[:, 1]

            plt.figure(figsize=(12, 6))
            plt.plot(blue_trace, label='Blue (Functional)', color='cyan', alpha=0.8)
            plt.plot(violet_trace, label='Violet (Hemo/Ref)', color='magenta', alpha=0.6)
            plt.title("Blue vs Violet Channel Comparison")
            plt.xlabel("Frame")
            plt.ylabel("Raw Intensity")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_folder, "Blue_vs_Violet_Trace.png"))
            plt.close()
            print(f"Saved: {os.path.join(output_folder, 'Blue_vs_Violet_Trace.png')}")

        if processed_dset_name:
            dset = f[dataset_type][processed_dset_name]
            print(f"Generating video preview for {processed_dset_name}...")
            preview_frames = dset.shape[0] 
            if dset.ndim == 3:
                data_sample = dset[:preview_frames, :, :]
            else:
                data_sample = dset[:preview_frames, 0, :, :]

            video_path = os.path.join(output_folder, "preview_video.mp4")
            save_video_of_stack(data_sample, video_path, fps=20, title=processed_dset_name)

# ==========================================
# Main
# ==========================================

def get_or_create_hdf5(parent_path):
    """
    Finds existing HDF5 file or creates a new one if none exists.
    Matches naming convention from Nicole/pipeline_utils.py
    """
    # Naming convention based on Nicole/pipeline_utils.py:
    # parent_prev_base + '_' + parent_base + '_processedData' + '.h5py'
    
    parent_base = os.path.basename(parent_path)
    # parent_path is .../preprocessed_data
    # so parent_prev_base is the session folder name
    # e.g. D:/.../Session1/preprocessed_data -> Session1
    
    # Actually, create_hdf5 takes 'hdf5_creation_folder_path' which is 'preprocessed_data'
    # os.path.dirname(parent_path) gives the session folder path.
    # os.path.basename(os.path.dirname(parent_path)) gives 'Session1'.
    
    session_dir_path = os.path.dirname(parent_path)
    session_name = os.path.basename(session_dir_path)
    
    # Nicole's create_hdf5 uses:
    # parent_base = os.path.basename(parent_path)  -> "preprocessed_data"
    # parent_prev_base = os.path.basename(os.path.dirname(parent_path)) -> "SessionName"
    # h5py_fileName = "SessionName_preprocessed_data_processedData"
    
    # Wait, let's verify exact logic from create_hdf5 read above:
    # parent_base = os.path.basename(parent_path) (this is the folder passed in)
    # parent_prev_base = os.path.basename(os.path.dirname(parent_path))
    # h5py_fileName = parent_prev_base + '_' + parent_base + '_processedData'
    
    # If parent_path is ".../Session1/preprocessed_data":
    # parent_base = "preprocessed_data"
    # parent_prev_base = "Session1"
    # Filename: "Session1_preprocessed_data_processedData.h5py"
    
    # Let's search for this pattern
    search_pattern = os.path.join(parent_path, f"{session_name}_{parent_base}_processedData*.h5py")
    existing_files = glob.glob(search_pattern)
    
    if existing_files:
        # Sort by modification time to get the most recent one (or creation time if reliable)
        # Using modification time seems safest to pick up where we left off
        existing_files.sort(key=os.path.getmtime, reverse=True)
        chosen_file = existing_files[0]
        print(f"Found existing HDF5 file: {chosen_file}")
        return chosen_file
    else:
        # Create new using the original function
        print("No existing HDF5 file found. Creating new...")
        return create_hdf5(parent_path)

def run_pipeline(exp_path, dataset_type, overwrite_raw):
    # Parameters (could be args if needed)
    MC_NREFERENCE = 60
    MC_CHUNKSIZE = 512
    HM_HIGHPASS = False
    
    try:
        # Create output location
        hdf5_creation_folder_path = os.path.join(exp_path, "preprocessed_data")
        if not os.path.exists(hdf5_creation_folder_path):
            os.makedirs(hdf5_creation_folder_path)
        
        # Use get_or_create instead of always create
        hdf5_file_path = get_or_create_hdf5(hdf5_creation_folder_path)
        print(f"HDF5 File: {hdf5_file_path}")
        
        step1_load_raw(hdf5_file_path, exp_path, dataset_type, overwrite_raw)
        step2_motion_correction(hdf5_file_path, dataset_type, MC_NREFERENCE, MC_CHUNKSIZE)
        path_to_mask = step3_and_4_masking(hdf5_file_path, hdf5_creation_folder_path, dataset_type)
        step5_hemo_correction(hdf5_file_path, path_to_mask, dataset_type, HM_HIGHPASS)
        step6_deltaF(hdf5_file_path, dataset_type)
        step7_visualization(hdf5_file_path, hdf5_creation_folder_path, dataset_type)
        
        print("\nPreprocessing pipeline completed successfully!")
        
    except Exception as e:
        print(f"\nPipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WField Preprocessing Pipeline")
    parser.add_argument('--data_dir', type=str, required=True, help="Path to experiment data (folder containing .dat files)")
    parser.add_argument('--dataset_type', type=str, default='led', help="Dataset type (led/widefield)")
    parser.add_argument('--overwrite', action='store_true', help="Overwrite raw frames if they exist")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory not found: {args.data_dir}")
        sys.exit(1)
        
    run_pipeline(args.data_dir, args.dataset_type, args.overwrite)
