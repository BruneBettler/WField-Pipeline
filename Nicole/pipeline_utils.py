# Author: Brune Bettler
# Last Modified: 2025-03-17
# ------------------------------
# NOTE: utils.py contains all necessary helper functions for the modular processing pipeline/workflow. 

import h5py
import os
import json
import glob
from tqdm import tqdm
import numpy as np
from multiprocessing import Pool, cpu_count
from functools import partial
import scipy
from pathlib import Path
import re
import cv2
from datetime import datetime, timedelta
import shutil
from scipy.ndimage import binary_dilation, binary_erosion
import sys

# HDF5 FUNCTIONS ------------------------------------
def _visit(name, obj):
    indent = "  " * (name.count("/") * 2)  # Indent based on depth
    short_name = os.path.basename(name)  # Get only the last part of the path
    if isinstance(obj, h5py.Group):
        print(f"{indent}*group*   {short_name}/")  # Group
    elif isinstance(obj, h5py.Dataset):
        print(f"{indent}*dataset*   {short_name} -> shape: {obj.shape}, dtype: {obj.dtype}")  # Dataset info
    return None

def print_hdf5_structure(file_path):
    """
    Recursively prints the hierarchical structure of an HDF5 file.
    """
    with h5py.File(file_path, "r") as hdf5_file:
        print(f"Contents of HDF5 File: {file_path}")
        hdf5_file.visititems(_visit)  # Recursively visit all groups and datasets
    
    return None

def create_hdf5(parent_path, verbose=True):
    '''
    Function creates an hdf5 file that will store all subsequent analysis datasets for inner paths and experiments. 
    Returns path of created hdf5 file. 
    '''
    parent_base = os.path.basename(parent_path)
    parent_prev_base = os.path.basename(os.path.dirname(parent_path))

    h5py_fileName = parent_prev_base + '_' + parent_base + '_processedData'
    h5py_extension = '.h5py'
    final_savePath = os.path.join(parent_path, (h5py_fileName + h5py_extension))

    # increment the file number if file already exists in the folder (ie. avoid overwriting data) 
    counter = 1
    while True:
        try:
            # Creates file if it does not exist and throws error if one exists 
            with h5py.File(final_savePath, 'x') as f:
                f.attrs['file_creation_date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S") # store file creation data 
                if verbose: print(f"H5py file successfully created: {final_savePath}.")
            break # exist loop

        except FileExistsError:
            # increment counter if file already exists 
            counter_addition = "_" + str(counter) 
            final_savePath = os.path.join(parent_path, (h5py_fileName + counter_addition + h5py_extension))
            counter += 1  # if necessary 

    return final_savePath

def get_hdf5(parent_path, verbose=True):
    '''
    Returns the path to the most recently updated HDF5 file in the given directory.
    If no HDF5 file exists, creates a new one.

    Inputs:
        - parent_path: Path where the HDF5 files are stored.
        - verbose: If True, prints messages about file selection/creation.

    Returns:
        - Path to the most recently updated HDF5 file (or a new one if none exist).
    '''
    # Extract the expected HDF5 file name pattern
    parent_base = os.path.basename(parent_path)
    parent_prev_base = os.path.basename(os.path.dirname(parent_path))
    hdf5_base_name = f"{parent_prev_base}_{parent_base}_processedData"

    # Find all matching HDF5 files in parent_path
    hdf5_files = [
        os.path.join(parent_path, f) 
        for f in os.listdir(parent_path) 
        if f.startswith(hdf5_base_name) and f.endswith('.h5py')
    ]

    # If HDF5 files exist, return the most recently modified one
    if hdf5_files:
        most_recent_hdf5 = max(hdf5_files, key=os.path.getmtime)  # Get file with latest modification time
        if verbose: print(f"Using existing HDF5 file: {most_recent_hdf5}")
        return most_recent_hdf5

    # If no HDF5 file exists, create a new one
    if verbose: print("No existing HDF5 file found. Creating a new one...")
    return create_hdf5(parent_path, verbose)


# PROGRESS LOG FUNCTIONS ----------------------------
def create_progressLog(parent_path, verbose=True):
    '''
    Function creates a .json file that will store all analysis progress for a given experimental run in order to follow and return back to processing. 
    Returns path of created .json file. 
    '''
    parent_base = os.path.basename(parent_path)
    parent_prev_base = os.path.basename(os.path.dirname(parent_path))

    json_fileName = parent_prev_base + '_' + parent_base + '_processingLog'
    json_extension = '.json'
    final_savePath = os.path.join(parent_path, (json_fileName + json_extension))

    # increment the file number if file already exists in the folder (ie. avoid overwriting data) 
    counter = 1
    while True:
        try:
            # Creates file if it does not exist and throws error if one exists 
            with open(final_savePath, "x") as f:
                progress_log_data = {}
                progress_log_data['log_creation_date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S") # store file creation data
                progress_log_data['last_updated'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                json.dump(progress_log_data, f, indent=4)
                if verbose: print(f"Progress .json file successfully created: {final_savePath}.")
            break # exist loop

        except FileExistsError:
            # increment counter if file already exists 
            counter_addition = "_" + str(counter) 
            final_savePath = os.path.join(parent_path, (json_fileName + counter_addition + json_extension))
            counter += 1  # if necessary 

    return final_savePath

def get_progressLog(parent_path, verbose=True):
    '''
    Returns the path to the most recently updated progress log in the given directory.
    If no progress log exists, creates a new one.

    Inputs:
        - parent_path: Path where the progress logs are stored.
        - verbose: If True, prints messages about file selection/creation.

    Returns:
        - Path to the most recently updated progress log (or a new one if none exist).
    '''
    # Extract the expected progress log name pattern
    parent_base = os.path.basename(parent_path)
    parent_prev_base = os.path.basename(os.path.dirname(parent_path))
    json_base_name = f"{parent_prev_base}_{parent_base}_processingLog"
    
    # Find all matching progress log files in parent_path
    json_files = [
        os.path.join(parent_path, f) 
        for f in os.listdir(parent_path) 
        if f.startswith(json_base_name) and f.endswith('.json')
    ]

    # If progress logs exist, return the most recently modified one
    if json_files:
        most_recent_log = max(json_files, key=os.path.getmtime)  # Get file with latest modification time
        if verbose: print(f"Using existing progress log: {most_recent_log}")
        return most_recent_log

    # If no log exists, create a new one
    if verbose: print("No existing progress log found. Creating a new one...")
    return create_progressLog(parent_path, verbose)
 
def load_progress(progressLog_path):
    '''
    Helper function for progress logging 
    Returns dictionary with Log data if present or an empty dictionary if not. 
    '''
    if os.path.exists(progressLog_path):
        with open(progressLog_path, "r") as f:
            return json.load(f)
    else: return {}  # If no file exists, start fresh

def save_progress(progressLog_path, progress):
    '''
    Helper function for progress logging 
    Returns 0. 
    '''
    with open(progressLog_path, "w") as f:
        json.dump(progress, f, indent=4)
    return 0

def update_progress(progress, exp_path, step, status, progressLog_path=None, saveProgress=False):
    '''
    Updates the progress log for a specific step within an experiment.

    Inputs:
        - progress: The dictionary tracking experiment progress.
        - exp_path: The experiment path as a string.
        - step: The name of the step being updated (e.g., 'motion_correction').
        - status: Status of the step ('completed', 'in progress', or an error message). 

    Returns:
        - 0 on success.
    '''
    progress['last_updated'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    parts = exp_path.strip(os.sep).split(os.sep)
    ref = progress
    for part in parts[:-1]:  # Traverse nested dict structure
        ref = ref.setdefault(part, {})

    # Ensure the last part of the experiment path exists
    if parts[-1] not in ref:
        ref[parts[-1]] = {}

    # Clear any previous errors if they exist
    if "error" in ref[parts[-1]]:
        del ref[parts[-1]]["error"]

    # Update specific step status
    ref[parts[-1]][step] = status

    # Save progress if required
    if saveProgress and progressLog_path:
        with open(progressLog_path, "w") as f:
            json.dump(progress, f, indent=4)

    return 0

def check_progress(progress, exp_path, step=None):
    '''
    Checks the progress of an experiment or a specific step.

    Inputs:
        - progress: The dictionary tracking experiment progress.
        - exp_path: The experiment path as a string.
        - step: Optional, the specific step to check.

    Returns:
        - If step is provided, returns the progress of that step (or None if not found).
        - If no step is provided, returns the full dictionary of steps for the experiment (or None if not found).
    '''
    parts = exp_path.strip(os.sep).split(os.sep)
    ref = progress
    for part in parts[:-1]:
        ref = ref.get(part, {})

    exp_data = ref.get(parts[-1], None)
    
    if step:
        return exp_data.get(step, None) if exp_data else None
    return exp_data  # Return all steps if step is not specified


# DATA LOADING AND PATH EXTRACTION FUNCTIONS ----------------------------
def match_stimMat_file(exp_path, path_to_mats, copy_files=False):
    '''
    Function identifies which stim mat file belongs to the experiment located in the exp_path
    Returns path to matfile corresponding to the experiment at exp_path

    copy_files (bool): If True, copies the matching .mat and .hdf5 files to exp_path.
    '''
    # identify the times associated to the exp at exp_path
    # load the frameTimes.mat file and identify the first and last time in this file
    frameTimes_path = get_exp_file_path(exp_path, 'T')
    fTime_mat_data = scipy.io.loadmat(frameTimes_path)
    fTimes = fTime_mat_data['frameTimes']
    firstTime = mattime_to_hour(fTimes[0][0])
    lastTime = mattime_to_hour(fTimes[-1][0])

    mat_files = []
    
    # Extract datetime from filenames
    pattern = re.compile(r"_(\d{15})")
    path_to_mats = Path(path_to_mats)
    for file in path_to_mats.glob("*.mat"):
        match = pattern.search(file.name)
        if match:
            raw_timestamp = match.group(1)  # Extracted 15-digit timestamp
            trimmed_timestamp = raw_timestamp[:14]  # Take only the first 14 digits
            file_time = datetime.strptime(trimmed_timestamp, "%Y%m%d%H%M%S")
            mat_files.append((file_time, file, raw_timestamp))
    
    # Sort files by extracted datetime
    mat_files.sort()
    
    prev_mat, post_mat = None, None
    prev_num, post_num = None, None
    for file_time, file_path, raw_timestamp in mat_files:
        if file_time < firstTime:
            prev_mat, prev_num = file_path, raw_timestamp
        elif file_time > lastTime:
            post_mat, post_num = file_path, raw_timestamp
            break
    
    # Return the best available option
    matching_mat, matching_num = (prev_mat, prev_num) if prev_mat else (post_mat, post_num)
    
    # Copy files if required
    if copy_files and matching_mat:
        shutil.copy(matching_mat, os.path.join(exp_path, matching_mat.name))
        print(f"the .mat file was successfully copied to the exp path: {os.path.join(exp_path, matching_mat.name)}")
        # Search for the corresponding .hdf5 file with the same timestamp
        corresponding_hdf5 = None
        for hdf5_file in path_to_mats.glob("*.hdf5"):
            if matching_num in hdf5_file.name:
                corresponding_hdf5 = hdf5_file
                shutil.copy(corresponding_hdf5, os.path.join(exp_path, corresponding_hdf5.name))
                print(f"the .hdf5 file was successfully copied to the exp path: {os.path.join(exp_path, corresponding_hdf5.name)}")

                break 
    return_path = os.path.join(exp_path, matching_mat.name) if copy_files and matching_mat else str(matching_mat)
    return str(matching_mat) if matching_mat else None

def get_exp_file_path(exp_path, file_type, dig=False):
    '''
    Helper function to extract the path of a given experiment. 
    file_type either "A" for analog, "F" for frames.dat, "T" for frameTimes.mat, or "M" for data.mat
    Returns the file path or None if file is not found

    - dig: allows the exploration into nested folders
    '''
    patterns = {
        'A': 'Analog_*.dat',
        'F': 'Frames_*.dat',
        'T': 'frameTimes_*.mat',
        'M': 'data*.mat'
    }
    # Try finding the file directly in exp_path
    files = glob.glob(os.path.join(exp_path, patterns[file_type]))
    
    if dig and not files:
        # Search recursively in subdirectories (only one level deep for efficiency)
        files = glob.glob(os.path.join(exp_path, '*', patterns[file_type]))

    return files[0] if files else None  # Return first match or None if no match

def datFrames_to_hdf5(dat_filepath, hdf5, exp_str='', compress=False, chunk_size=512, verbose=False):
    '''
    Function efficiently loads .dat frame data and saves them into an HDF5 file.

    - hdf5 = filepath or HDF5 group 
    '''
    # Extract metadata from filename (assumes format: Frames_2_640_540_uint16_0001.dat)
    dat_info = os.path.basename(dat_filepath).split('_')
    channels = int(dat_info[1])
    H = int(dat_info[2])
    W = int(dat_info[3])
    dtype = np.dtype(dat_info[4])

    frame_size = H * W * channels
    file_size = os.path.getsize(dat_filepath)
    n_frames = file_size // (frame_size * dtype.itemsize)

    # **Check if hdf5 is a file path or an HDF5 group**
    file_opened = False  # Track whether we opened an HDF5 file
    hdf5_group = None
    if isinstance(hdf5, str):  # If `hdf5` is a file path
        h5file = h5py.File(hdf5, 'a')  # Open file in append mode
        hdf5_group = h5file  # Use root group
        file_opened = True  # Mark that we opened the file
    elif isinstance(hdf5, h5py.Group):  # If `hdf5` is an HDF5 group
        hdf5_group = hdf5  # Use the provided group
    else:
        raise ValueError("Invalid HDF5 input. Must be a file path (str) or an h5py.Group.")

    # Memory-map the data file for efficient loading
    frame_data = np.memmap(dat_filepath, dtype=dtype, mode='r', shape=(n_frames, channels, H, W))

    # create dataset name (remove dataset if it already exists)
    if 'raw_frames' in hdf5_group: del hdf5_group['raw_frames']
    if f"{exp_str}_raw_frames" in hdf5_group: del hdf5_group[f"{exp_str}_raw_frames"]

    dataset_name = f"{exp_str}_raw_frames" if exp_str else "raw_frames"
    
    try:
        if verbose: 
            print(f"{datetime.now()}: Saving frame data into HDF5 at {dataset_name}... ")

        # Determine chunk size for efficient writing
        chunk_shape = (min(chunk_size, n_frames), channels, H, W)

        # Create dataset in the correct HDF5 location
        dataset = None
        if compress:
            dataset = hdf5_group.create_dataset(dataset_name, shape=(n_frames, channels, H, W), 
                                                dtype=dtype, compression='lzf', chunks=chunk_shape)
        else:
            dataset = hdf5_group.create_dataset(dataset_name, shape=(n_frames, channels, H, W), 
                                                dtype=dtype, chunks=chunk_shape)

        # Write data in chunks
        for i in tqdm(range(0, n_frames, chunk_size), desc='.dat --> .hdf5'):
            end = min(i + chunk_size, n_frames)
            dataset[i:end] = frame_data[i:end]  

        if verbose: 
            print(f"{datetime.now()}: Done saving frames to HDF5!")

    finally:
        if file_opened:
            hdf5_group.file.flush()
            h5file.close()  # Close the file only if we opened it
        else:
            hdf5_group.file.flush()
        del frame_data  # Clean up memory-mapped file

    #'raw_blueFrames' = frame_data[:, 0, :, :], 'raw_violetFrames' = frame_data[:, 1, :, :]
    return dataset_name

def get_darkFrame_num(hdf5_dataset, chunk_size=500, threshold=0.01):
    """
    Function counts the number of dark frames at the end of an HDF5-stored 1D frame array.
    
    Parameters:
        hdf5_dataset (str): Path to the HDF5 file.
        chunk_size (int): Number of frames to read at a time from the end.
        threshold (float): Frame intensity threshold below which frames are considered dark.
    
    Returns:
        int: Total number of consecutive dark frames at the end of the hdf5 dataset.
    """
    
    total_frames = hdf5_dataset.shape[0]  # Get total number of frames
    num_dark_blue = 0
    num_dark_violet = 0
    
    # Read in reverse, chunk by chunk
    for i in range(total_frames, 0, -chunk_size):
        start = max(0, i - chunk_size)  # Ensure we don’t go below index 0
        chunk = normalize_arr(hdf5_dataset[start:i])  # Read chunk (shape: (chunk_size, 2, H, W))

        # Compute mean intensity per frame for each channel
        mean_blue = np.mean(chunk[:, 0, :, :], axis=(1, 2))  # Channel 0 (Blue)
        mean_violet = np.mean(chunk[:, 1, :, :], axis=(1, 2))  # Channel 1 (Violet)

        # Find where frames are dark
        dark_mask_blue = mean_blue <= threshold
        dark_mask_violet = mean_violet <= threshold
        
        # Count dark frames starting from the end of the chunk
        num_dark_in_chunk_blue = np.sum(dark_mask_blue)  
        num_dark_in_chunk_violet = np.sum(dark_mask_violet)

        # If all frames in chunk are dark, add count and continue
        num_dark_blue += num_dark_in_chunk_blue
        num_dark_violet += num_dark_in_chunk_violet
        if num_dark_in_chunk_blue == len(chunk) or num_dark_in_chunk_violet == len(chunk):
            continue
        else: break  

    return int(num_dark_blue), int(num_dark_violet)


# MOTION CORRECTION FUNCTIONS
# from churchland code
def chunk_indices(nframes, chunksize = 512, min_chunk_size = 16):
    '''
    Gets chunk indices for iterating over an array in evenly sized chunks
    '''
    chunks = np.arange(0,nframes,chunksize,dtype = int)
    if (nframes - chunks[-1]) < min_chunk_size:
        chunks[-1] = nframes
    if not chunks[-1] == nframes:
        chunks = np.hstack([chunks,nframes])
    return [[chunks[i],chunks[i+1]] for i in range(len(chunks)-1)]

def parinit():
    os.environ['MKL_NUM_THREADS'] = "1"
    os.environ['OMP_NUM_THREADS'] = "1"

def runpar(f,X,nprocesses = None,**kwargs):
    '''
    res = runpar(function,          # function to execute
                 data,              # data to be passed to the function
                 nprocesses = None, # defaults to the number of cores on the machine
                 **kwargs)          # additional arguments passed to the function (dictionary)

    '''

    if nprocesses is None:
        nprocesses = cpu_count()
    with Pool(initializer = parinit, processes=nprocesses) as pool:
        res = pool.map(partial(f,**kwargs),X)
    pool.join()
    return res

# FINAL FRAME STACK FUNCTIONS
def filter_by_neighbors(stack, size=2):
    if len(stack.shape) != 3:
        stack = np.expand_dims(stack, axis=0)  # Ensure stack is 3D (frames, height, width)
    # Binary mask of active pixels (1 = valid, 0 = NaN)
    active_mask = np.isfinite(stack).astype(int)

    # Count active neighbors per pixel (frame by frame)
    final = np.full(active_mask.shape, np.nan)
    mask_struc = np.ones((size,size))
    for i in range(stack.shape[0]):
        eroded = binary_erosion(active_mask[i], mask_struc)
        dilated = binary_dilation(eroded, structure=mask_struc)
        # If any sizexsize window has all 1s, its avg will be 1.0
        final[i] = np.where(dilated, stack[i], np.nan)

    if final.shape[0] == 1:
        final = final[0]
    return final


# OTHER / TODO: group these better
def normalize_arr(arr):
    '''
    Function normalizes an array such that the returned values are between values 0 and 1 
    '''
    min_v = np.nanmin(arr)
    max_v = np.nanmax(arr)

    return np.array((arr - min_v) / (max_v - min_v))

def mattime_to_hour(mat_time):
    """
    Convert the fractional part of a day to hours, minutes, and seconds.

    Parameters:
        fractional_day (float): The fractional part of a day.

    Returns:
        list: [hours, minutes, seconds]
    """
    """fractional_day = mat_time - int(mat_time)
    # Total hours in a day
    hours = fractional_day * 24
    # Extract the integer part as hours
    hour = int(hours)
    
    # Calculate remaining fraction and convert to minutes
    minutes = (hours - hour) * 60
    minute = int(minutes)
    
    # Calculate remaining fraction and convert to seconds
    seconds = (minutes - minute) * 60
    second = round(seconds, 10)  

    # Return as a list
    return hours #[hour, minute, second]"""

    python_datetime = datetime.fromordinal(int(mat_time) - 366) + timedelta(days=mat_time%1)
    return python_datetime

def flip_image_about_x(image: np.ndarray, x_val: int) -> np.ndarray:
    """
    Flips the left part of an image about a given x-value (vertical line)
    and places the flipped portion on the right side.
    
    Parameters:
    - image (np.ndarray): Input image (grayscale or RGB).
    - x_val (int): X-coordinate (column index) about which to flip.

    Returns:
    - np.ndarray: Image with the left part reflected to the right side.
    """
    # Ensure x_val is within bounds
    if x_val < 0 or x_val >= image.shape[1]:
        raise ValueError("x_val must be within the image width.")

    # Extract the left part of the image (before x_val)
    left_part = image[:, :x_val]  # Shape: (H, x_val, C) or (H, x_val) for grayscale

    # Flip the left part horizontally
    left_part_flipped = np.flip(left_part, axis=1)  # Flip along columns

    # Determine how much space we have on the right side
    available_width = image.shape[1] - x_val

    # Clip the flipped part to fit within the image width
    left_part_flipped = left_part_flipped[:, :available_width]

    # Create a copy of the original image
    flipped_image = image.copy()

    # Place the flipped portion on the right side
    flipped_image[:, x_val:x_val + left_part_flipped.shape[1]] = left_part_flipped

    return flipped_image

def get_bounding_box(mask):
    """Finds the bounding box of the nonzero region in a binary mask."""
    if mask.dtype != np.uint8:
        mask = (mask > 0).astype(np.uint8) * 255  # Ensure binary format (0 or 255)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        raise ValueError("No contours found in the binary mask.")
    
    x, y, w, h = cv2.boundingRect(contours[0])
    return x, y, w, h  # Return x, y position and width, height

def compute_scaling_factor(mask1, mask2):
    """Computes the scaling factor to resize mask1 to the size of mask2 while preserving aspect ratio."""
    _, _, w1, h1 = get_bounding_box(mask1)
    _, _, w2, h2 = get_bounding_box(mask2)
    
    scale_w = w2 / w1
    scale_h = h2 / h1

    # Use the average scale to preserve aspect ratio
    return (scale_w + scale_h) / 2

def get_centroid(mask):
    """Finds the centroid (center of mass) of the binary mask."""
    moments = cv2.moments(mask)
    if moments["m00"] == 0:
        raise ValueError("Mask is empty, cannot find centroid.")
    
    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])
    
    return cx, cy

def translate_image(image, dx, dy, target_shape):
    """Translates an image by dx, dy pixels while keeping its position in the frame."""
    h_target, w_target = target_shape
    h, w = image.shape[:2]

    # Create a blank canvas of the target shape
    translated_img = np.zeros((h_target, w_target), dtype=image.dtype)

    # Compute valid region inside the target frame
    x_start = max(0, dx)
    y_start = max(0, dy)
    x_end = min(w_target, w + dx)
    y_end = min(h_target, h + dy)

    # Compute valid region inside the source image
    x_start_src = max(0, -dx)
    y_start_src = max(0, -dy)
    x_end_src = x_start_src + (x_end - x_start)
    y_end_src = y_start_src + (y_end - y_start)

    # Place translated mask inside the blank canvas
    translated_img[y_start:y_end, x_start:x_end] = image[y_start_src:y_end_src, x_start_src:x_end_src]

    return translated_img

def apply_transforms_to_stack(image_stack, scale_factor, dx, dy, target_shape):
    """Applies scaling, translation, and size adjustments to a stack of frames, preserving NaN values."""
    processed_stack = []

    for img in image_stack:
        # Create a mask of NaN locations (boolean mask instead of uint8)
        nan_mask = np.isnan(img)

        # Replace NaNs with 0 before resizing
        img_no_nan = np.nan_to_num(img, nan=0)

        # Resize both the image and the NaN mask
        resized_img = cv2.resize(img_no_nan, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
        resized_nan_mask = cv2.resize(nan_mask.astype(np.float32), None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_NEAREST)

        # Convert back to boolean for safety
        resized_nan_mask = resized_nan_mask > 0  # True where NaNs should be

        # Translate both the image and the NaN mask
        translated_img = translate_image(resized_img, dx, dy, target_shape).astype(np.float32)  # Keep float
        translated_nan_mask = translate_image(resized_nan_mask.astype(np.float32), dx, dy, target_shape)

        # Convert translated_nan_mask back to boolean (avoid overwriting valid pixels)
        translated_nan_mask = translated_nan_mask > 0

        # Restore NaNs only in correct locations
        translated_img[translated_nan_mask] = np.nan  

        processed_stack.append(translated_img)

    return np.array(processed_stack, dtype=np.float32)  # Ensure output remains float32


if __name__ == "__main__":
    # Define the parent directory containing multiple experiment folders
    parent_exp_dir = Path(r"d:\wfield\NicoleData\TKO\7186")  
    path_to_mats = Path(r"d:\wfield\NicoleData\TKO\20250326_and_27th_TKO7184_7186")  

    # Loop through each experiment folder and run the function
    for exp_folder in parent_exp_dir.glob("*/*"):
        if exp_folder.is_dir():  # Ensure it's a directory
            print(f"Processing experiment: {exp_folder.name}")
            result = match_stimMat_file(exp_folder, path_to_mats, copy_files=True)
            print(f"Matched file: {result}\n")
