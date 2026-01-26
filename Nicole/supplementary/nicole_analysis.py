# Author: Brune Bettler
# Last Modified: 2025-06-05

# import modules 
# widefield modules
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from pipeline_utils import * 
from pipeline_processing import * 
from GUIs.mask_drawing_gui import *
from PIL import Image

# general modules 
contour_mask_template_path = r"D:\wfield\NicoleData\default_brain_mask_down.png"
segmented_mask_template_path = r"D:\wfield\NicoleData\default_segmentation_down.png" 

mask_paths = [contour_mask_template_path, segmented_mask_template_path]

def Nicole_preprocess(exp_path, hdf5_group, progressLog_path, clean_when_done=True, verbose=True):
    '''
    Function runs the specific preprocessing steps on a single experiment for Nicole's data

    input:
        - exp_path: path to experiment containing all necessary files (.dat, .mat etc...)
        - hdf5_group: HDF5 group where processed data should be stored.
    '''
    progress = load_progress(progressLog_path)
    
    steps = [
        "raw_frames",
        "motion_corrected",
        "hemo_corrected",
        "normalized_trial_average"
    ]

    exp_rel_path = os.path.relpath(exp_path, os.path.dirname(progressLog_path))  # Relative path for logging
    
    # Load progress for this experiment
    exp_progress = check_progress(progress, exp_rel_path) or {}

    REMOVE_DARK_FRAMES = True # if True, dark frames wont be considered during preprocessing (motion_corrected, masked, hemo_corrected, deltaF, trial_averaged)
   
    try:
        # Step 1: Retreive or Store raw frames
        if exp_progress.get("raw_frames") != "completed":
            update_progress(progress, exp_rel_path, "raw_frames", "in progress", progressLog_path, saveProgress=True)

            dat_filepath = get_exp_file_path(exp_path, 'F', dig=True)
            compress = False
            rawFrame_dataset_str = datFrames_to_hdf5(dat_filepath, hdf5_group, '', compress, verbose)

            update_progress(progress, exp_rel_path, "raw_frames", "completed", progressLog_path, saveProgress=True)
    
        
        # Step 2: Motion Correction
        if exp_progress.get("motion_corrected") != "completed":
            update_progress(progress, exp_rel_path, "motion_corrected", "in progress", progressLog_path, saveProgress=True)

            if 'motion_corrected' in hdf5_group: del hdf5_group['motion_corrected']

            if REMOVE_DARK_FRAMES:
                # determine how many dark frames there are in the experiment and store this value for subsequent processing
                blue_dark_frame_num, violet_dark_frame_num = get_darkFrame_num(hdf5_group['raw_frames'])
                num_to_remove = max(blue_dark_frame_num, violet_dark_frame_num)
        
            
            output_dataset_shape = hdf5_group["raw_frames"].shape
            if REMOVE_DARK_FRAMES:
                output_dataset_shape = (output_dataset_shape[0] - num_to_remove, *output_dataset_shape[1:])
    
            motion_corrected_dataset = hdf5_group.create_dataset('motion_corrected', shape=output_dataset_shape, dtype=hdf5_group["raw_frames"].dtype) 
            
            hdf5_motion_correct(hdf5_group["raw_frames"], motion_corrected_dataset, nreference=60, chunksize=512, dark_frames_to_remove=num_to_remove)
            hdf5_group.file.flush()

            update_progress(progress, exp_rel_path, "motion_corrected", "completed", progressLog_path, saveProgress=True)

        
        # Step 4: Hemodynamic Correction
        if exp_progress.get("hemo_corrected") != "completed":
            update_progress(progress, exp_rel_path, "hemo_corrected", "in progress", progressLog_path, saveProgress=True)

            # get npy mask 
            mask =  np.load(os.path.join(exp_path, "brain_mask.npy"))

            if 'hemo_corrected' in hdf5_group: del hdf5_group['hemo_corrected']
            new_shape = (hdf5_group['motion_corrected'].shape[0], *hdf5_group['motion_corrected'].shape[2:])
            hemoCorrected_dataset = hdf5_group.create_dataset('hemo_corrected', shape=new_shape, dtype='float64', chunks=(200, *hdf5_group['motion_corrected'].shape[2:]), compression=None)
            
            hemodynamic_correction(hdf5_group['motion_corrected'], hemoCorrected_dataset, mask, highpass=True)
            
            hdf5_group.file.flush()

            update_progress(progress, exp_rel_path, "hemo_corrected", "completed", progressLog_path, saveProgress=True)

        # Step 5: Compute the trial average with DeltaF/F normalization per trial 
        if exp_progress.get("normalized_trial_average") != "completed":
            update_progress(progress, exp_rel_path, "normalized_trial_average", "in progress", progressLog_path, saveProgress=True)

            # get the stim times datetime 
            stim_times_arr = get_stim_times(exp_path, "LED")

            # create dataset in which frames are divided into trials 
            if 'normalized_trial_average' in hdf5_group: del hdf5_group['normalized_trial_average']
            trial_avg_frames_dataset = hdf5_group.create_dataset("normalized_trial_average", shape=(25, 640, 540), maxshape=(None, 640, 540), dtype='float32')  
            
            if 'trial_avg_xTimes' in hdf5_group: del hdf5_group['trial_avg_xTimes']
            
            trial_avg_xTimes_dataset = hdf5_group.create_dataset("trial_avg_xTimes", shape=(45,), maxshape=(None, ), dtype='float64')

            frameTime_mat_path = get_exp_file_path(exp_path, 'T', dig=True)

            mean_trial_frames, trial_xTime = get_trialAvg_stack(frameTime_mat_path, hdf5_group['hemo_corrected'], stim_times_arr, pre_post_stim=(1, 2), normalization="zScore", pre_onset_baseline_range=(1.0,.5), mask=None, hdf5_output_dataset=hdf5_group['normalized_trial_average'], png_save_path=exp_path, verbose=True)
            
            new_shape_frames = mean_trial_frames.shape
            new_shape_xTimes = trial_xTime.shape
            # Resize the dataset before writing
            trial_avg_frames_dataset.resize(new_shape_frames)  
            trial_avg_xTimes_dataset.resize(new_shape_xTimes)
            # Assign new data
            trial_avg_frames_dataset[:] = mean_trial_frames
            trial_avg_xTimes_dataset[:] = trial_xTime

            hdf5_group.file.flush()
            update_progress(progress, exp_rel_path, "normalized_trial_average", "completed", progressLog_path, saveProgress=True)

        '''# Step 5: Compute deltaF/F
        if exp_progress.get("deltaF") != "completed":
            update_progress(progress, exp_rel_path, "deltaF", "in progress", progressLog_path, saveProgress=True)

            if 'deltaF' in hdf5_group: del hdf5_group['deltaF']
            deltaF_dataset = hdf5_group.create_dataset('deltaF', shape=hdf5_group['hemo_corrected'].shape, dtype='float64')
            brain_mask = normalize_arr(np.load(os.path.join(exp_path, "brain_mask.npy")))
            get_hdf5_deltaF(hdf5_group['hemo_corrected'], deltaF_dataset, mask=brain_mask)
            hdf5_group.file.flush()
            update_progress(progress, exp_rel_path, "deltaF", "completed", progressLog_path, saveProgress=True)

        # Step 6: Create and store trial averaged frames
        if exp_progress.get("trial_averaged") != "completed":
            update_progress(progress, exp_rel_path, "trial_averaged", "in progress", progressLog_path, saveProgress=True)

            # get the stim times datetime 
            stim_times_arr = get_stim_times(exp_path, "LED")

            # create dataset in which frames are divided into trials 
            if 'trial_avg_frames' in hdf5_group: del hdf5_group['trial_avg_frames']
            trial_avg_frames_dataset = hdf5_group.create_dataset("trial_avg_frames", shape=(25, 640, 540), maxshape=(None, 640, 540), dtype='float32')  
            if 'trial_avg_xTimes' in hdf5_group: del hdf5_group['trial_avg_xTimes']
            trial_avg_xTimes_dataset = hdf5_group.create_dataset("trial_avg_xTimes", shape=(45,), maxshape=(None, ), dtype='float64')

            frameTime_mat_path = get_exp_file_path(exp_path, 'T', dig=True)

            # get the aligned stims 
            trial_avg_frames_arr, trial_xTimes_arr = get_trialAvg_stack(frameTime_mat_path, hdf5_group['deltaF'], stim_times_arr, pre_post_stim=(1, 2), png_save_path=exp_path)
            
            new_shape_frames = trial_avg_frames_arr.shape
            new_shape_xTimes = trial_xTimes_arr.shape
            # Resize the dataset before writing
            trial_avg_frames_dataset.resize(new_shape_frames)  
            trial_avg_xTimes_dataset.resize(new_shape_xTimes)
            # Assign new data
            trial_avg_frames_dataset[:] = trial_avg_frames_arr
            trial_avg_xTimes_dataset[:] = trial_xTimes_arr

            hdf5_group.file.flush()
            update_progress(progress, exp_rel_path, "trial_averaged", "completed", progressLog_path, saveProgress=True)'''

        # Add attributes (metadata)
        hdf5_group.attrs["processing_pipeline_run"] = "Nicole_preprocess"
        hdf5_group.attrs["processing_steps_run"] = steps
        hdf5_group.attrs["last_modified"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # to save space now that we have everything, keep only the following datasets in the hdf5 file
        if clean_when_done:
            to_keep = ['motion_corrected', 'hemo_corrected', 'normalized_trial_average', 'trial_avg_xTimes'] # ["motion_corrected", "deltaF", "trial_averaged"]
            for step in steps:
                if step not in to_keep and step in hdf5_group: 
                    del hdf5_group[step]
            
            hdf5_group.file.flush()
        # Final update: mark the experiment as completed
        update_progress(progress, exp_rel_path, "EXPERIMENT_PROCESSING_STATUS", "completed", progressLog_path, saveProgress=True)

    except Exception as e:
        update_progress(progress, exp_rel_path, "error", str(e), progressLog_path, saveProgress=True)
        raise e

'''
The result of the pipeline is one hdf5 file per animal containing the 6 experimental preprocessed data as well as alignment
'''

def Nicole_postprocess(path_to_all, folder_structure, hdf5_save):
    '''
    goes through each animal type and aligns the masks and creates a trial average
    '''   
    hdf5_level_idx = folder_structure.index(hdf5_save) + 1  # Determine index of the HDF5 level

    # Get all folders at the HDF5 level
    hdf5_folders = glob.glob(os.path.join(path_to_all, *['*'] * (hdf5_level_idx)))

    # prep arrays in which to store the per light-type and per animal type folder 
    data_dict = {'GNAT': [], 'TKO':[], 'WT': []}
    for hdf5_folder in hdf5_folders:
        # now we have access to each mouse folder (where a single HDF5 file per mouse is stored)
        # check which experiment it's coming from
        path = Path(hdf5_folder)
        parts = path.parts # ('D:\\', 'wfield', 'NicoleData', 'WT', '7202')
        nicole_idx = parts.index('NicoleData')
        experimental_group = parts[nicole_idx + 1] # you'll use this string to add data to the right dictionary entry

        curr_hdf5_path = get_hdf5(hdf5_folder, verbose=True)

        with h5py.File(curr_hdf5_path, "r+") as hdf5_file:  # Open HDF5 
            None

if __name__ == "__main__":
    # TODO: make sure to modify the mask template paths at the top of this file! 
    path_to_all = r"D:\wfield\NicoleData"
    progressLog_path = get_progressLog(path_to_all, verbose=True)
    folder_structure = ['experimental_groups', 'animals', 'experiments'] # if only running a single experiment only have: folder_structure = ["experiments"]
    hdf5_save = 'animals'
    
    process_experiments(progressLog_path, path_to_all, folder_structure, hdf5_save, Nicole_preprocess)
    #Nicole_postprocess(path_to_all, folder_structure, hdf5_save)

    
    