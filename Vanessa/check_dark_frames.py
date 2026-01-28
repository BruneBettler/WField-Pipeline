
import sys
import os
import glob
import h5py
import numpy as np
import scipy.io
from tqdm import tqdm

# Setup path to import Nicole modules
current_dir = os.getcwd()
nicole_path = os.path.join(current_dir, 'Nicole')
if nicole_path not in sys.path:
    sys.path.insert(0, nicole_path)

from pipeline_utils import get_exp_file_path, get_darkFrame_num, get_hdf5
from data_extraction import get_frameTime_data

EXP_PATH = r"D:\Vanessa_test_data\Tests_Jan23\23-Jan-2026_ledTTL_10random"

def check_dark_frames():
    print(f"Checking data in: {EXP_PATH}")
    
    # 1. Check frameTimes.mat metadata
    print("\n--- 1. Checking Metadata (frameTimes.mat) ---")
    ft_path = get_exp_file_path(EXP_PATH, 'T', dig=True)
    if ft_path:
        print(f"Found frameTimes file: {os.path.basename(ft_path)}")
        try:
            # We just want removedFrames, so we can use loadmat directly or the helper
            mat_data = scipy.io.loadmat(ft_path)
            if 'removedFrames' in mat_data:
                rem_frames = mat_data['removedFrames'][0][0]
                print(f"Metadata 'removedFrames': {rem_frames}")
            else:
                print("'removedFrames' key not found in .mat file.")
        except Exception as e:
            print(f"Error reading .mat file: {e}")
    else:
        print("No frameTimes_*.mat file found.")

    # 2. Check HDF5 raw_frames
    print("\n--- 2. Checking Raw Data (HDF5) ---")
    hdf5_folder = os.path.join(EXP_PATH, "preprocessed_data")
    if not os.path.exists(hdf5_folder):
        print("No preprocessed_data folder found. Run Step 1 of pipeline first.")
        return

    hdf5_path = get_hdf5(hdf5_folder, verbose=False)
    if not hdf5_path:
        print("No HDF5 file found.")
        return
        
    print(f"Analyzing HDF5: {os.path.basename(hdf5_path)}")
    
    with h5py.File(hdf5_path, 'r') as f:
        if 'led' not in f or 'raw_frames' not in f['led']:
            print("Dataset 'led/raw_frames' not found.")
            return
            
        dset = f['led']['raw_frames']
        n_frames, channels, H, W = dset.shape
        print(f"Raw Frames Shape: {dset.shape}")
        
        # A. Run Nicole's get_darkFrame_num (Checks END of file)
        print("Running get_darkFrame_num (checks end of file)...")
        # Need to be careful about verify context of get_darkFrame_num, 
        # it expects a dataset.
        try:
            blue_dark, violet_dark = get_darkFrame_num(dset)
            print(f"Dark frames detected at END of recording: Blue={blue_dark}, Violet={violet_dark}")
        except Exception as e:
            print(f"Error running get_darkFrame_num: {e}")
            
        # B. Scan for internal dark frames (The "dips")
        print("Scanning entire dataset for internal dark frames (frames < 100 mean intensity)...")
        # We'll use a chunked approach
        chunk_size = 1000
        internal_dark_count = 0
        dark_indices = []
        
        for i in tqdm(range(0, n_frames, chunk_size)):
            chunk_end = min(i + chunk_size, n_frames)
            # Read Blue channel
            chunk = dset[i:chunk_end, 0, :, :] 
            
            # Simple mean per frame
            means = np.mean(chunk, axis=(1, 2))
            
            # Threshold check (100 is very low for uint16 0-65535 data, usually means empty)
            # Raw data trace showed dips to ~0-2000 range? 
            # In trace image, user said "large dip". 
            # Trace image y-axis goes to 50000, dips go to < 5000 approx.
            # Let's use a conservative threshold like 1000 or find the min.
            
            mask = means < 4000 # Based on the user's plot looking like it dips way down
            if np.any(mask):
                local_indices = np.where(mask)[0] + i
                dark_indices.extend(local_indices)
                internal_dark_count += len(local_indices)
                
        print(f"Total frames below threshold (4000): {internal_dark_count}")
        if dark_indices:
            print(f"First 10 dark indices: {dark_indices[:10]}")
            # Check periodicity
            if len(dark_indices) > 1:
                diffs = np.diff(dark_indices)
                print(f"Sample spacing between dark frames: {diffs[:10]}")

if __name__ == "__main__":
    check_dark_frames()
