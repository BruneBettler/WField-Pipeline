# Author: Brune Bettler
# Last Modified: 2025-06-05

from pipeline_utils import *

import h5py
import os
import tqdm
import glob
from pipeline_utils import normalize_arr
from skimage.transform import AffineTransform
import sys
import h5py
import numpy as np
from tqdm import tqdm
import cv2
from scipy.linalg import svd
from sklearn.preprocessing import normalize
from GUIs.mask_drawing_gui import * 
from datetime import timedelta
from data_extraction import *
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, sosfiltfilt
import datetime
from scipy import stats
from visualization_utils import load_dat_analog


'''
pipeline_processing.py contains all functions necessary to run a modular processing pipeline/workflow for a single or series of experiemnts. 
Details on these functions and their use can be found in the Understanding_Processing.ipynb notebook. 
'''

# NOTE: analysis functions should have the following as input when called: exp_path, hdf5_path, progressLog_path


def process_experiments(progressLog_path, path_to_all, folder_structure, hdf5_save, analysis_function, *analysis_args, **analysis_kwargs):
    '''
    Function guides and completes the processing and logging steps for a set of experiments or animals.
    Inputs:
        - progressLog_paths: path to JSON file progress log. can be empty or already containing entries. 
        - path_to_all: Path to the outermost parent directory containing experimental data.
        - folder_structure: List defining folder structure (e.g., ['main', 'experimental_groups', 'animals', 'experiments']).
        - hdf5_save: The folder level in `folder_structure` where HDF5 files should be created.
        - analysis_function: The function that processes each experiment (e.g., `Nicole_preprocess`).
        - *analysis_args: Extra positional arguments for `analysis_function`.
        - **analysis_kwargs: Extra keyword arguments for `analysis_function`.
    '''
    progress = load_progress(progressLog_path)
    hdf5_level_idx = folder_structure.index(hdf5_save) + 1  # Determine index of the HDF5 level

    # Get all folders at the HDF5 level
    hdf5_folders = glob.glob(os.path.join(path_to_all, *['*'] * (hdf5_level_idx)))

    for hdf5_folder in tqdm(hdf5_folders, desc=f"Processing {hdf5_save}"):
        curr_hdf5_path = get_hdf5(hdf5_folder, verbose=True)

        with h5py.File(curr_hdf5_path, "r+") as hdf5_file:  # Open HDF5 
        
            # Process experiments under this HDF5 level
            experiment_folders = [f for f in glob.glob(os.path.join(hdf5_folder, '*')) if os.path.isdir(f)]

            for exp in tqdm(experiment_folders, desc="Experiments", leave=False):
                rel_exp_path = os.path.relpath(exp, path_to_all)

                exp_progress = check_progress(progress, rel_exp_path) or {}

                # If all steps are completed, skip
                if exp_progress and exp_progress.get("EXPERIMENT_PROCESSING_STATUS") == "completed":
                    tqdm.write(f"Skipping {rel_exp_path} (Already completed)")
                    continue

                tqdm.write(f"Processing: {rel_exp_path}")

                try:
                    exp_str = os.path.basename(exp) # get the str experiment name from the folder

                    # Create a nested group in HDF5 to match the experiment structure
                    exp_group = hdf5_file.require_group(rel_exp_path.replace(os.path.sep, "_"))
                
                    # TODO: understand kargs and args here 
                    analysis_function(exp_path=exp, hdf5_group=exp_group, progressLog_path=progressLog_path)

                except Exception as e:
                    tqdm.write(f"Error in {rel_exp_path}: {e}")
                    raise e
                    #continue  # Skip to the next experiment instead of stopping


# PREPROCSSING ----------------------------------
# ----------------------- MOTION CORRECTION ----------------------------------
# Motion correction functions adapted from:
#  wfield - tools to analyse widefield data - motion correction 
# Copyright (C) 2020 Joao Couto - jpcouto@gmail.com
# Licensed under GNU General Public License v3.0
# https://www.gnu.org/licenses/gpl-3.0.en.html
# Modified by Brune Bettler, 2025 for analysis in the Trenholm Lab

def hdf5_motion_correct(input_hdf5_dataset, output_hdf5_dataset, nreference=60, chunksize=512, dark_frames_to_remove=0):
    '''
    Motion correction by translation for HDF5 datasets. Estimates x and y shifts using phase correlation.
    Inputs:
        input_hdf5_dataset (h5py.Dataset)   : HDF5 dataset containing raw frames (shape: NFRAMES, NCHANNELS, H, W)
        output_hdf5_str (str)               : name of HDF5 dataset for storing corrected frames (output dataset will have same shape as hdf5_dataset)
        nreference (int)                    : Number of frames to take as reference (default: 60)
        chunksize (int)                     : Size of chunks for processing (default: 512)
        dark_frames_to_remove (int)         : number of frames to remove from the end of all channels, default 0
        
    Returns:
        yshifts (ndarray)                    : Y shifts (NFRAMES, NCHANNELS)
        xshifts (ndarray)                    : X shifts (NFRAMES, NCHANNELS)
        rshifts (ndarray)                    : Rotation shifts (NFRAMES, NCHANNELS)
    '''
    mode = 'ecc'
    nframes, nchan, h, w = input_hdf5_dataset.shape
    nframes -= dark_frames_to_remove  
    chunks = chunk_indices(nframes, chunksize)  # Compute chunk indices

    yshifts = []
    xshifts = []
    rshifts = []

    # Compute reference frames using the middle nreference frames (each channel independently)
    midpoint = int(nframes / 2)
    ref_chunk = input_hdf5_dataset[midpoint - nreference // 2 : midpoint + nreference // 2]
    refs = np.mean(ref_chunk, axis=0).astype('float32')  # Mean reference per channel

    # Align reference frames
    _, refs = _register_multichannel_stack(ref_chunk, refs, mode=mode)
    refs = np.mean(refs, axis=0).astype('float32')  # Compute final reference

    # Process chunks efficiently
    for c in tqdm(chunks, desc='Motion correction'):
        start, end = c[0], c[-1]

        # Read chunk directly from HDF5 (no conversion to NumPy array)
        localchunk = input_hdf5_dataset[start:end]  # (chunksize, nchan, h, w)

        # Run motion correction on the chunk
        (xs, ys, rot), corrected = _register_multichannel_stack(localchunk, refs, mode=mode)

        # Write corrected frames back to HDF5
        output_hdf5_dataset[start:end] = corrected

        # Store shifts for later use
        yshifts.append(ys)
        xshifts.append(xs)
        rshifts.append(rot)

    return (np.vstack(yshifts), np.vstack(xshifts)), np.vstack(rshifts)

def _register_multichannel_stack(frames, templates, mode='ecc', niter=25, eps0=1e-3, warp_mode=cv2.MOTION_EUCLIDEAN):
    '''
    Function registers a multichannel image stack for motion correction.
    Supports HDF5 dataset for memory efficiency.
    
    Inputs:
        frames (h5py.Dataset) : Frame stack (NFRAMES, NCHANNELS, H, W)
        templates (ndarray)   : Templates for alignment (NCHANNELS, H, W)
        mode (str)            : Registration mode ('ecc' or '2d')
        niter (int)           : Number of iterations for ECC
        eps0 (float)          : Convergence criteria for ECC
        warp_mode (int)       : OpenCV warp mode
    
    Returns:
        (xs, ys, rot)         : Translation and rotation shifts
        stack                 : Corrected image stack (HDF5 or ndarray)
    '''
    nframes, nchannels, h, w = frames.shape
    
    if mode == 'ecc':
        hann = cv2.createHanningWindow((w, h), cv2.CV_32FC1)
        hann = (hann * 255).astype('uint8')
    
    ys = np.zeros((nframes, nchannels), dtype=np.float32)
    xs = np.zeros((nframes, nchannels), dtype=np.float32)
    rot = np.zeros((nframes, nchannels), dtype=np.float32)
    
    # If an HDF5 dataset is provided, use it; otherwise, create an in-memory array
    stack = np.zeros_like(frames, dtype=np.uint16)
    
    for ichan in range(nchannels):
        for i in range(nframes):
            chunk = frames[i, ichan]  # Access frame without loading everything into memory
            
            if mode == '2d':
                res = _registration_upsample(chunk, templates[ichan])
                ys[i, ichan] = res[0][1]
                xs[i, ichan] = res[0][0]
            
            elif mode == 'ecc':
                res = _registration_ecc(chunk, templates[ichan], hann=hann, niter=niter, eps0=eps0, warp_mode=warp_mode)
                xy, rots = _xy_rot_from_affine([res[0]])
                ys[i, ichan] = xy[0][1]
                xs[i, ichan] = xy[0][0]
                rot[i, ichan] = rots[0]
            
            # Store processed frame either in HDF5 or numpy array
            stack[i, ichan] = res[1]
    
    return (xs, ys, rot), stack

def _registration_ecc(frame,template,
                     niter = 25,
                     eps0 = 1e-2,
                     warp_mode = cv2.MOTION_EUCLIDEAN,
                     prepare = True,
                     gaussian_filter = 1,
                     hann = None,
                     **kwargs):
    h,w = template.shape
    if hann is None:
        hann = cv2.createHanningWindow((w,h),cv2.CV_32FC1)
        hann = (hann*255).astype('uint8')
    dst = frame.astype('float32')
    M = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                niter,  eps0)
    (res, M) = cv2.findTransformECC(template,dst, M, warp_mode, criteria, inputMask=hann, gaussFiltSize=gaussian_filter)
    dst = cv2.warpAffine(frame, M, (w,h), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    return M, np.clip(dst,0,(2**16-1)).astype('uint16')

def _registration_upsample(frame,template):
    h,w = frame.shape
    dst = frame.astype('float32')
    (xs, ys), sf = cv2.phaseCorrelate(template.astype('float32'),dst)    
    M = np.float32([[1,0,xs],[0,1,ys]])
    dst = cv2.warpAffine(dst,M,(w, h))
    return (xs,ys),(np.clip(dst,0,(2**16-1))).astype('uint16')

def _xy_rot_from_affine(affines):
    '''
    helper function to parse affine parameters from ECC
    '''
    xy = []
    rot = []
    for r in affines:
        M = np.vstack([r, np.array([0,0,1])])
        M = AffineTransform(M)
        xy.append(M.translation)
        rot.append(M.rotation)
    rot = np.rad2deg(np.array(rot))
    xy = np.array(xy)
    return xy,rot

# ----------------------- BRAIN CONTOUR MASKING ----------------------------------

def mask_stack(exp_path, input_dataset, output_dataset, mask_paths, chunksize=512):
    '''
    Function applies "brain_mask" found in exp_path to the input_dataset if it exists or prompts 
    the user to create a brain mask if not. Function stores the masked_dataset in the output_dataset. 
    '''
    # check if 'brain_mask.png' exists
    mask_npy_path = os.path.join(exp_path, "brain_mask.npy")
           
    # if not, ask run the GUI and loop until the files exist
    while not os.path.exists(mask_npy_path):   
        if input_dataset.shape[1] == 2:
            image1 = input_dataset[0,0,...] # blue_frame_sample 
            image2 = input_dataset[0,1,...] # violet_frame_sample   
        else:
            image1 = input_dataset[0,0,...] # frame sample 
            image2 = image1

        app = QApplication(sys.argv)
        window = MaskDrawingApp((normalize_arr(image1)*255).astype(np.uint8), (normalize_arr(image2)*255).astype(np.uint8), output_path=exp_path, template_mask_paths=mask_paths)
        window.show()
        app.exec_()
        
   # go through chunked dataset and apply mask 
    mask = np.load(mask_npy_path).astype(input_dataset.dtype)
    mask = normalize_arr(mask)[None, None, :,:]
    chunks = chunk_indices(input_dataset.shape[0], chunksize)  # Compute chunk indices

    for c in tqdm(chunks, desc='Masking frames'):
        start, end = c[0], c[-1]

        # Read chunk directly from HDF5
        input_chunk = input_dataset[start:end]  # (chunksize, nchan, h, w)

        # Mask chunk (all channels)
        output_dataset[start:end] = input_chunk * mask
    
    return 0


# ----------------------- HEMODYNAMIC CORRECTION ----------------------------------
def highpass_filter_optimized_2D(data, fs=20, cutoff=0.1, order=5):
    """Fast filtering for 2D data (Time x Pixels)."""    
    # Design filter as second-order sections
    sos = butter(order, cutoff / (0.5 * fs), btype='highpass', output='sos')
    
    # Filter along time axis (axis=0) - no reshaping needed!
    filtered = sosfiltfilt(sos, data, axis=0)
    
    return filtered

def highpass_filter(data, fs, cutoff=0.1, order=5):
    """
    Applies a Butterworth high-pass filter to time-series data.

    Parameters:
    - data (ndarray): Time x Pixels (2D) or Time x Height x Width (3D) array.
    - fs (float): Sampling frequency in Hz.
    - cutoff (float): Cutoff frequency in Hz (default is 0.1 Hz).
    - order (int): Filter order (default is 5).

    Returns:
    - filtered_data (ndarray): High-pass filtered data, same shape as input.
    """

    # Design filter
    nyq = 0.5 * fs  # Nyquist frequency
    norm_cutoff = cutoff / nyq
    b, a = butter(order, norm_cutoff, btype='highpass', analog=False)

    # Reshape if 3D image stack (T, H, W) → (T, Pixels)
    original_shape = data.shape
    if data.ndim == 3:
        T, H, W = data.shape
        data = data.reshape(T, -1)

    # Apply filter along time axis
    filtered = filtfilt(b, a, data, axis=0)

    # Restore original shape
    if len(original_shape) == 3:
        filtered = filtered.reshape(original_shape)

    return filtered
    

def hemodynamic_correction(input_dataset, output_dataset=None, binary_npy_mask=None, highpass=True, verbose=True):
    '''
    Function takes blue and violet frame data and performs hemodynamic correction on blue frames using regression per pixel 
    input_dataset: (frame_num, 2, W, H) with the blue frames in the first dimension
    '''
    if verbose: print(f"\nStarting hemodynamic correction: {datetime.datetime.now()}")
    # regression: 
    # neuronal_signal_pixel = corresponding_blue_pix - B * corresponding_violet_pix 
    n_frames, _, H, W = input_dataset.shape

    # variables to track cropping
    crop_bbox = None

    if binary_npy_mask is not None:
        # Find bounding box of mask to crop to minimal region
        mask_indices = np.where(binary_npy_mask == 1)
        if len(mask_indices[0]) > 0:  # Check if mask has any valid pixels
            min_row, max_row = np.min(mask_indices[0]), np.max(mask_indices[0])
            min_col, max_col = np.min(mask_indices[1]), np.max(mask_indices[1])
            crop_bbox = (min_row, max_row + 1, min_col, max_col + 1)  # +1 for inclusive slicing
            
            # Crop the mask
            cropped_mask = binary_npy_mask[min_row:max_row + 1, min_col:max_col + 1]
            cropped_H, cropped_W = cropped_mask.shape
            
            work_H, work_W = cropped_H, cropped_W
            mask_flat = cropped_mask.astype(bool).reshape(-1)
            if verbose:
                print(f"Cropping from {H}x{W} to {cropped_H}x{cropped_W} (reduction: {((H*W - cropped_H*cropped_W)/(H*W)*100):.1f}%)")
        else:
            if verbose: print("Warning: Mask contains no valid pixels, processing full frame")
            crop_bbox = None
            work_H, work_W = H, W
            mask_flat = binary_npy_mask.astype(bool).reshape(-1)
    else:
        mask_flat = None
        work_H, work_W = H, W
    
    # Flatten spatial dimensions: shape -> (n_frames, pixels)
    blue_flat = np.empty((n_frames, work_H * work_W), dtype=np.float32)
    violet_flat = np.empty((n_frames, work_H * work_W), dtype=np.float32)
    chunk_size = 200 

    for i in range(0, n_frames, chunk_size):
        end = min(i + chunk_size, n_frames)
        
        if crop_bbox is not None:
            # Crop the chunks
            min_row, max_row, min_col, max_col = crop_bbox
            blue_flat[i:end] = input_dataset[i:end, 0, min_row:max_row, min_col:max_col].reshape(end - i, -1).astype(np.float32)
            violet_flat[i:end] = input_dataset[i:end, 1, min_row:max_row, min_col:max_col].reshape(end - i, -1).astype(np.float32)
        else:
            # Use full frames
            blue_flat[i:end] = input_dataset[i:end, 0, ...].reshape(end - i, -1).astype(np.float32)
            violet_flat[i:end] = input_dataset[i:end, 1, ...].reshape(end - i, -1).astype(np.float32)

    if highpass:
        blue_flat = highpass_filter_optimized_2D(blue_flat, fs=20, cutoff=0.1, order=2)
        violet_flat = highpass_filter_optimized_2D(violet_flat, fs=20, cutoff=0.1, order=2)

    # Apply mask if provided
    if binary_npy_mask is not None:
        blue_flat[:, ~mask_flat] = np.nan
        violet_flat[:, ~mask_flat] = np.nan
  
    # Compute regression coefficients (beta) per pixel
    # alpha = cov(blue, violet) / var(violet)
    numerator = np.nansum(blue_flat * violet_flat, axis=0)
    numerator[numerator == 0] = np.nan
    denominator = np.nansum(violet_flat ** 2, axis=0) 
    denominator[denominator == 0] = np.nan
    beta = numerator / (denominator) # shape gives 1 beta value per pixel

    # Predicted hemodynamic component
    predicted = violet_flat * beta  

    # Subtract to get corrected signal
    corrected_flat = blue_flat - predicted  
    
    # Reshape to working image dimensions
    corrected_flat = corrected_flat.reshape(n_frames, work_H, work_W)
    
    del blue_flat, violet_flat, predicted
    if 'numerator' in locals():
        del numerator
    if 'denominator' in locals():
        del denominator
    if 'beta' in locals():
        del beta
    
    if output_dataset is None:
        output_dataset = np.zeros((n_frames, H, W), dtype=corrected_flat.dtype)

    # Repad to original size if cropping was used
    if crop_bbox is not None:
        # Place cropped data back into original position
        min_row, max_row, min_col, max_col = crop_bbox
        for i in tqdm(range(0, n_frames, 200), desc="Writing hemocorrected frames to hdf5"):
            output_dataset[i:i+200, min_row:max_row, min_col:max_col] = corrected_flat[i:i+200]
    else:
        for i in tqdm(range(0, n_frames, 200), desc="Writing hemocorrected frames to hdf5"):
            output_dataset[i:i+200] = corrected_flat[i:i+200]

    if verbose: 
        print(f"\nFinished hemodynamic correction: {datetime.datetime.now()}")
    
    return output_dataset

# POSTPROCSSING ----------------------------------
# ----------------------- STIMULUS ALIGNMENT ----------------------------------

# FRAME TIME ALIGNMENT + STIMULUS STIM TIME (corrected for delay between wf and stim for "OLD" and "NEW" data formats)
def get_stim_times(experiment_path, stim_type, hdf5_output_dataset=None):
    '''
    Function returns the start and stop times of the stimulus presentation. To be used with datetime frametimes
    Funtion is the same as alignment_1_1 in exploration.utils when DATA_FORMAT == "NEW" and alignment_1_2 when DATA_FORMAT == "OLD"
    '''
    # determine whether the experiment data_format == "NEW" (after feb 12th 2025) or "OLD"
    data_format = get_dataFormat_type(experiment_path)
    stim_pulseTimes_array = []

    if data_format == "NEW":
        # get stim_pulse_times 
        stim_pulse_times = get_stim_pulse_times(experiment_path, stim_type, data_format)
        
        if stim_type in ["LED", "AUDIO"]:
            stim_time = get_stim_duration(experiment_path, stim_type)
            # Add STIM_TIME seconds to every second entry starting from the first
            for i in range(0, len(stim_pulse_times), 2):  # Start at index 1, step by 2
                stim_pulseTimes_array.append([stim_pulse_times[i], (stim_pulse_times[i] + timedelta(seconds=stim_time))])           
        
        elif stim_type == "OLFAC":
            mix_dur, odor_dur = get_odor_MixandDur(experiment_path, data_format)
            for i in range(0, len(stim_pulse_times), 2):  # Start at index 1, step by 2
                odor_start = stim_pulse_times[i] + timedelta(seconds=mix_dur)
                stim_pulseTimes_array.append([odor_start, odor_start + timedelta(seconds=odor_dur)])

        # OPTIONAL BUT RECOMMENDED: correct for the delay between wf and stim computer:
        delay_seconds = get_alignment_delay(experiment_path, data_format, stim_type)
        for pulse in stim_pulseTimes_array:
            pulse[0] -= timedelta(seconds = delay_seconds) 
            pulse[1] -= timedelta(seconds = delay_seconds)
    
    else: # "OLD" / use alignment_1_2
        analog_data, analog_dict = load_dat_analog(get_exp_file_path(experiment_path, 'A', dig=True))

        analog_x_values = np.arange(len(analog_data[1])) / 1000 # get time in seconds 
        wf_exp_stim_pulses = normalize_arr(analog_data[4])

        analog_onset = mattime_to_hour(analog_dict['onset'])
        wf_x_times = [analog_onset + timedelta(seconds=s) for s in analog_x_values]

        # find the index of every first 1 occuring after a 0 
        corrected_wf_experimental_stim = np.zeros_like(wf_exp_stim_pulses)
        corrected_wf_experimental_stim[1:] = (wf_exp_stim_pulses[1:] >= 0.5) & (wf_exp_stim_pulses[:-1] <= 0.5)
        corrected_wf_experimental_stim[0] = wf_exp_stim_pulses[0]
        # for each start index (pusle), determine and add the end index
        indices = np.where(corrected_wf_experimental_stim == 1)[0]

        # extract all the start and stop locations and put into an array of tuples 
        stim_pulseTimes_array = []
        STIM_TIME = get_stim_duration(experiment_path, stim_type)
       
        for i in indices:
            stim_pulseTimes_array.append([wf_x_times[i], wf_x_times[i+int((1000 * STIM_TIME))]])
        
        delay_seconds = get_alignment_delay(experiment_path, data_format, stim_type)
        for pulse in stim_pulseTimes_array:
            pulse[0] += timedelta(seconds = delay_seconds) 
            pulse[1] += timedelta(seconds = delay_seconds)

    if hdf5_output_dataset != None:
        # HDF5 does not support datetime objects so convert to seconds
        stim_pulseTimes_array = [[p[0].timestamp(), p[1].timestamp()] for p in stim_pulseTimes_array]
        new_shape = (len(stim_pulseTimes_array), 2)  # (25,2)
        # Resize the dataset before writing
        hdf5_output_dataset.resize(new_shape)  
        # Assign new data
        hdf5_output_dataset[:] = np.array(stim_pulseTimes_array)
    # read back with this line "timestamps = [[datetime.datetime.fromtimestamp(t) for t in row] for row in hdf5_output_dataset[:]]"
    return np.array(stim_pulseTimes_array) # returned array is an array of datetime objects


# ----------------------- TRIAL AVERAGE STACK w/ NORMALIZATION ----------------------------------

def get_deltaF(input_dataset, baseline_avg, output_dataset=None, mask=None, verbose=True):
    """
    datasets should have shape (frameNum, H, W)
    output_dataset: hdf5 dataset to be used if wanting to store output in hdf5 file
    """

    # Compute Delta F/F 
    num_frames = input_dataset.shape[0]  # Get total number of frames

    if num_frames > 512 or output_dataset: # delta F is long and should be chunked
        for i in tqdm(range(0, num_frames, 512), desc="Saving deltaF/F frames", unit="chunk"):
            end_idx = min(i + 512, num_frames)  # Prevent going out of bounds

            # Load chunk from input dataset
            input_chunk = input_dataset[i:end_idx, :, :]

            # Compute the baseline correction
            if mask is not None:
                submask = np.where(mask == 0, baseline_avg, 0)
                divmask = np.where(mask == 0, baseline_avg, 1)
                output_chunk = (input_chunk - submask) / divmask
            else:
                output_chunk = (input_chunk - baseline_avg) / baseline_avg  

            # Write to the HDF5 dataset
            output_dataset[i:end_idx, :, :] = output_chunk
    
    else: # small dataset or trial 
        output_dataset = np.full((input_dataset.shape), np.nan)
        # Compute the baseline correction
        if mask is not None and not np.isnan(baseline_avg.flatten()).any(): # if baseline avg contains np.nan, this means the input does as well and we do not need to concern ourselves with special non-zero divison.
            submask = np.where(mask != 0, baseline_avg, 0)
            divmask = np.where(mask != 0, baseline_avg, 1)
            output_dataset = (input_dataset - submask) / divmask
        else:
            output_dataset = (input_dataset - baseline_avg) / baseline_avg


    return output_dataset


def get_zScore(input_dataset, baseline_stack, output_dataset=None, mask=None, verbose=True):
    """
    input trial_data should have shape (frameNum, H, W)
    output_dataset: hdf5 dataset to be used if wanting to store output in hdf5 file
    """
    if output_dataset is None:
        # allocate output array only once if not passed in
        output_dataset = np.empty_like(input_dataset, dtype=np.float32)

    mean_intensity = np.nanmean(baseline_stack, axis=0)
    std_intensity = np.nanstd(baseline_stack, axis=0, ddof=1) 

    # debugging:
    #print(f"Baseline mean range: {np.nanmin(mean_intensity)} to {np.nanmax(mean_intensity)}")
    #print(f"Baseline std range: {np.nanmin(std_intensity)} to {np.nanmax(std_intensity)}")
    #print(f"Pixels with std < 10: {np.sum(std_intensity < 10)}")

    # Prevent division by zero (safe for low-variance pixels)
    std_intensity[std_intensity == 0] = np.nan # TODO: consider std_intensity < 0.1 rather than = 0? 
    std_intensity[std_intensity < 1] = 1
    
    num_frames = input_dataset.shape[0]  # Get total number of frames
    if num_frames > 512:
        for i in tqdm(range(0, num_frames, 512), desc="Saving zScored frames", unit="chunk"):
            end_idx = min(i + 512, num_frames)  # Prevent going out of bounds

            # Load chunk from input dataset
            input_chunk = input_dataset[i:end_idx, :, :]

            z_chunk = (input_chunk - mean_intensity) / std_intensity  

            output_dataset[i:end_idx] = z_chunk
    else:
        output_dataset[:] = (input_dataset - mean_intensity) / std_intensity
    
    return output_dataset # TODO: consider clipping output between -10 and 10 ? 




def get_trialAvg_stack(frameTimeMat_path, frames_array, stim_times_array, pre_post_stim=(1, 2), normalization="zScore", pre_onset_baseline_range=(1.0,.5), mask=None, hdf5_output_dataset=None, png_save_path=None, verbose=True):
    '''
    Returns a stack of frames trial averaged for a single animal / experimental session
    frames_array = hdf5 dataset or array containing frames data over which to calculate the average (preprocessed or not)
    stimTimes_array = hdf5 dataset or array containing datetime info (output of the get_stim_times function above) 
    pre_post_stim = tuple of seconds representing the amount of seconds to consider prior to stim onset and after stim offset when making a trial average
    normalization = "deltaF" (default), "zScore", or None. 
    pre_onset_baseline_range = tuple of seconds used when normalizing with deltaF. first value defines the baseline start time prior to start of stim onset in seconds (positive value only)
    '''
    if verbose: print(f"\nStarting trial averaging: {datetime.datetime.now()}")

    # extract frameTimes .mat file data
    ftimes, _ = get_frameTime_data(frameTimeMat_path, returnType="datetime")
    blue_ftimes = ftimes[0]

    # if there are more frameTimes than frames, shorten the frameTime array from the end
    if len(blue_ftimes) > frames_array.shape[0]:
        blue_ftimes = blue_ftimes[:frames_array.shape[0]]
    
    if isinstance(stim_times_array, h5py.Dataset):
        # because the dateTime stim times cannot be stored as datetime objects in hdf5 we must convert them (check alignment function for details)
        stim_times_array = [[datetime.datetime.fromtimestamp(t) for t in row] for row in stim_times_array[:]]

    # capture the start and stop times of each trial based on the dateTime_stim_times and the pre_post_stim times
    # check to make sure the baseline (if desired) is within the pre-post stim time requested. if not, extend start stop time for baseline to be included and applied.
    if normalization is not None and (pre_onset_baseline_range[0] > pre_post_stim[0]):
        # extend the adjusted trial start to include the baseline
        adjusted_tuples = np.array([(start - timedelta(seconds=pre_onset_baseline_range[0]), end + timedelta(seconds=pre_post_stim[1])) for start, end in stim_times_array])
    else:
        # desired baseline is included in adjusted trial
        adjusted_tuples = np.array([(start - timedelta(seconds=pre_post_stim[0]), end + timedelta(seconds=pre_post_stim[1])) for start, end in stim_times_array])

    # Initialize a list to store extracted data for each interval
    trial_lengths = []
    frame_trials = []
    trial_times = []
    stim_onset_indices = []

    baseline_nums = []
    # Loop through each interval
    for i, (start, end) in enumerate(adjusted_tuples):
        # Find indices corresponding to the interval
        start_idx = np.searchsorted(blue_ftimes, start, side='left') # first ≥ stim_start
        end_idx = np.searchsorted(blue_ftimes, end, side='right') # first > stim_stop
        
        # Extract the data for the interval and append to the list
        time_data = blue_ftimes[start_idx:end_idx]
        trial_data = frames_array[start_idx:end_idx][:]

        stim_onset_indices.append(np.searchsorted(time_data, stim_times_array[i][0]))

        # account for normalization if desired
        if normalization is not None:    
            # determine the frames to be used for baseline
            baseline_start_idx = np.searchsorted(time_data, (stim_times_array[i][0] - timedelta(seconds=pre_onset_baseline_range[0])), side='left')
            baseline_stop_idx = np.searchsorted(time_data, (stim_times_array[i][0] - timedelta(seconds=pre_onset_baseline_range[1])), side='right')
            baseline_frames = trial_data[baseline_start_idx:baseline_stop_idx]
            baseline_nums.append(baseline_frames.shape[0])
    
            if normalization == 'deltaF':
                # take the average of these frames 
                # normal to have an error as with the np.nan mask, certain rows and columns are all np.nan
                baseline_avg = np.nanmean(baseline_frames, axis=0)  # Shape: (height, width)
                # run the deltaF function with the desired slice :) ? 
                normalized_trial = get_deltaF(trial_data, baseline_avg, mask=mask)
            elif normalization == 'zScore':
                normalized_trial = get_zScore(trial_data, baseline_frames, mask=mask)
            
            frame_trials.append(normalized_trial)

        else:
            frame_trials.append(trial_data)
        
        trial_times.append(time_data)
        trial_lengths.append(len(trial_data))
    
    # Convert to arrays early for consistent handling
    trial_lengths = np.array(trial_lengths)
    stim_onset_indices = np.array(stim_onset_indices)

    if verbose: 
        print(f"Frames per trial block:{trial_lengths}")
        if normalization is not None : print(f"Number of frames used for normalization baseline: {baseline_nums}")

    # find the max number of frames
    mode_frame_num = stats.mode(trial_lengths, keepdims=False).mode
    print(mode_frame_num)

    valid_mask = trial_lengths == mode_frame_num

    # check if all trial lengths are equal
    if not np.all(valid_mask):
        n_removed = np.sum(~valid_mask)
        if verbose: print(f"Removing {n_removed} trials with non-modal lengths")
    
        # remove trial from trial_lengths, trial_times, frame_trials, baseline_nums
        trial_lengths = trial_lengths[valid_mask]
        stim_times_array = np.array(stim_times_array)[valid_mask]
        adjusted_tuples = adjusted_tuples[valid_mask]
        stim_onset_indices = stim_onset_indices[valid_mask]

        frame_trials = [trial for i, trial in enumerate(frame_trials) if valid_mask[i]]
        trial_times = [times for i, times in enumerate(trial_times) if valid_mask[i]]

        trial_times = np.array(trial_times)
        frame_trials = np.array(frame_trials)
    else:
        trial_times = np.array(trial_times)
        stim_onset_indices = np.array(stim_onset_indices)
    
    print(stim_onset_indices)
    mean_trial_frames = np.nanmean(np.array(frame_trials), axis=0)

    centered_trial_times = np.empty(trial_times.shape)
    for trial_num, trial_data in enumerate(trial_times):
        start = trial_data[0]
        for i in range(len(trial_data)):
            d = trial_data[i] - start
            centered_trial_times[trial_num][i] = d.total_seconds()

    average_times = np.mean(centered_trial_times, axis=0)
    print(f"stim start index: {stats.mode(stim_onset_indices, keepdims=False)[0]} = {average_times[stats.mode(stim_onset_indices, keepdims=False)[0]]}")

    if hdf5_output_dataset != None:
        new_shape = mean_trial_frames.shape
        # Resize the dataset before writing
        hdf5_output_dataset.resize(new_shape)  
        # Assign new data
        hdf5_output_dataset[:] = mean_trial_frames
    
    trial_time = (adjusted_tuples[0][1] - adjusted_tuples[0][0]).total_seconds()
    if verbose: print(f"Trial duration: theory {trial_time}s, practice {average_times[-1]}")
    trial_xTime = average_times # x-axis time in seconds

    if png_save_path:
        # organize the frames on a single image
        cols = 10  
        rows = (mode_frame_num + cols - 1) // cols 

        fig, axs = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))

        axs = axs.flatten()  # Flatten for easy indexing
        minVal = np.nanmin(mean_trial_frames)
        maxVal = np.nanmax(mean_trial_frames)
        for i in range(mode_frame_num):
            axs[i].imshow(mean_trial_frames[i], cmap='hot', vmin=-5, vmax=maxVal) 
            axs[i].set_title(str(round(trial_xTime[i], 3)))
            axs[i].axis('off')  # Remove axis ticks

        # Hide any unused subplots
        for i in range(mode_frame_num, len(axs)):
            fig.delaxes(axs[i])
        
        plt.tight_layout()
        plt.savefig(os.path.join(png_save_path, 'mean_trial_frames.png'), bbox_inches='tight')
        plt.close(fig)
        
        if verbose: print(f"Png showing trial average frames saved: {os.path.join(png_save_path, 'mean_trial_frames.png')}")
        
    if verbose: print(f"\nFinished trial averaging: {datetime.datetime.now()}")

    return mean_trial_frames, trial_xTime




if __name__ == "__main__":
    from matplotlib.animation import FuncAnimation
    from IPython.display import HTML

    EXP_GROUP = 'WT'
    ANIMAL_NUM = '7202'
    FW = 1
    pre_post_stim = (1,2)
    pre_onset_baseline_range=(1,.5)

    # no need to edit further down 
    EXPERIMENT_PATH = rf"D:\wfield\NicoleData\{EXP_GROUP}\{ANIMAL_NUM}\LED_530_R_F0.5_ND{FW-1}_FW{FW}"
    FRAMETIME_MAT_PATH = get_exp_file_path(EXPERIMENT_PATH, 'T', dig=True)
    HDF5_EXPERIMENT_PATH = get_hdf5(EXPERIMENT_PATH[:-23], verbose=True) # path to hdf5 file containing the desired preprocessed dataset
    HDF5_DATASET_PATH =  f"{EXP_GROUP}_{ANIMAL_NUM}_LED_530_R_F0.5_ND{FW-1}_FW{FW}/hemo_corrected" # path within the hdf5 file that leads to the desired dataset 

    MASK = np.load(rf"D:\wfield\NicoleData\{EXP_GROUP}\{ANIMAL_NUM}\LED_530_R_F0.5_ND{FW-1}_FW{FW}\brain_mask.npy")

    stim_times = get_stim_times(EXPERIMENT_PATH, "LED") # returns as array of datetime objects
    png_save_path = r"C:\Users\bbettl\PycharmProjects\wfield_pipeline\PIPELINE\OUTPUT_MASKS_TEMP"

    #dataset = None
    with h5py.File(HDF5_EXPERIMENT_PATH, 'r') as f:
        dataset = f[HDF5_DATASET_PATH]

        #frames_mean, single_trial_time, offset = get_trialAvg_stack(FRAMETIME_MAT_PATH, dataset, stim_times_arr, pre_post_stim=pre_post_stim, normalization="deltaF", mask=MASK)
        frames_mean, single_trial_time = get_trialAvg_stack(FRAMETIME_MAT_PATH, dataset, stim_times, pre_post_stim=pre_post_stim, pre_onset_baseline_range=pre_onset_baseline_range, png_save_path=png_save_path, normalization="zScore", mask=MASK)

        ROI_mask = np.load(r"C:\Users\bbettl\PycharmProjects\wfield_pipeline\PIPELINE\OUTPUT_MASKS_TEMP\brain_mask.npy").astype(np.float32)
        ROI_mask[ROI_mask == 0] = np.nan
        ROI_mask[ROI_mask >= 1] = 1
        
        frame_ROI_means = np.nanmean(frames_mean[:] * ROI_mask, axis=(1,2))

        plt.plot(single_trial_time, frame_ROI_means, linewidth=2)
        plt.xlabel("Trial Time")
        plt.ylabel("Frame ROI mean")
        plt.axvspan(pre_post_stim[0],pre_post_stim[0]+0.5, color='red', alpha=0.3, label="Stim LED on")      
        plt.scatter(single_trial_time, frame_ROI_means, s=3)  
        plt.title(f"{EXP_GROUP} {ANIMAL_NUM} FW{FW}\npre_post_stim={pre_post_stim} with mask, zScore {pre_onset_baseline_range}")
        plt.show()