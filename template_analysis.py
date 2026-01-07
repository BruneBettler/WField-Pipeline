
# MODIFY THE CODE BELOW UNTIL YOU REACH THE NEXT ALL CAPS LINE
STEPS = [
        "raw_frames",
        "masked",
        "hemo_corrected",
        "deltaF",
        "trial_averaged"
        ]


REMOVE_DARK_FRAMES = True               # if True, dark frames wont be consider during preprocessing (motion_corrected, masked, hemo_corrected, deltaF, trial_averaged)

CLEAN_HDF5_WHEN_DONE = False
STEPS_TO_KEEP_POST_CLEAN = []           # leave empty if no CLEAN_HDF5_WHEN_DONE == FALSE

CONTOUR_MASK_TEMPLATE_PATH = None       # ADD HERE
SEGMENTED_MASK_TEMPLATE_PATH = None     # ADD HERE 

STIM_TYPE = "LED"                       # one of "LED", "OLFAC", "AUDIO"

SECONDS_BEFORE_STIM = 1                 # for trial averages 
SECONDS_AFTER_STIM = 2

# YOU'RE GOOD! NO NEED TO MODIFY BELOW THIS LINE -----------------------------------------------------------------

# import modules 
from pipeline_utils import * 
from processing import * 
from trace_brain import *
from PIL import Image


mask_paths = [CONTOUR_MASK_TEMPLATE_PATH, SEGMENTED_MASK_TEMPLATE_PATH]

def Template_analysis(exp_path, hdf5_group, progressLog_path, verbose=True):
    '''
    Function runs the specific preprocessing steps on a single experiment for Nicole's data

    input:
        - exp_path: path to experiment containing all necessary files (.dat, .mat etc...)
        - hdf5_group: HDF5 group where processed data should be stored.
    '''
    progress = load_progress(progressLog_path)

    exp_rel_path = os.path.relpath(exp_path, os.path.dirname(progressLog_path))  # Relative path for logging
    
    # Load progress for this experiment
    exp_progress = check_progress(progress, exp_rel_path) or {}
    
    blue_dark_frame_num, violet_dark_frame_num = 0, 0
    num_to_remove = 0

    try:
        for curr_i, step in enumerate(STEPS):
            if step == "raw_frames":
                # Store raw frames
                if exp_progress.get("raw_frames") != "completed":
                    update_progress(progress, exp_rel_path, "raw_frames", "in progress", progressLog_path, saveProgress=True)

                    dat_filepath = get_exp_file_path(exp_path, 'F', dig=True)
                    compress = False
                    datFrames_to_hdf5(dat_filepath, hdf5_group, '', compress, verbose)

                    update_progress(progress, exp_rel_path, "raw_frames", "completed", progressLog_path, saveProgress=True)
                if REMOVE_DARK_FRAMES:
                    # determine how many dark frames there are in the experiment and store this value for subsequent processing
                    blue_dark_frame_num, violet_dark_frame_num = get_darkFrame_num(hdf5_group['raw_frames'])
                    num_to_remove = max(blue_dark_frame_num, violet_dark_frame_num)
 
            elif step == "motion_corrected":
                # Motion Correction
                if exp_progress.get("motion_corrected") != "completed":
                    update_progress(progress, exp_rel_path, "motion_corrected", "in progress", progressLog_path, saveProgress=True)

                    if 'motion_corrected' in hdf5_group: del hdf5_group['motion_corrected']
                    
                    output_dataset_shape = hdf5_group["raw_frames"].shape
                    if REMOVE_DARK_FRAMES:
                        output_dataset_shape = (output_dataset_shape[0] - num_to_remove, *output_dataset_shape[1:])
            
                    motion_corrected_dataset = hdf5_group.create_dataset('motion_corrected', shape=output_dataset_shape, dtype=hdf5_group["raw_frames"].dtype) 
                    
                    hdf5_motion_correct(hdf5_group["raw_frames"], motion_corrected_dataset, nreference=60, chunksize=512, dark_frames_to_remove=num_to_remove)
                    hdf5_group.file.flush()

                    update_progress(progress, exp_rel_path, "motion_corrected", "completed", progressLog_path, saveProgress=True)

            elif step == "masked":
                # Apply Masks: take the prev step in steps to be the dataset to mask. 
                if exp_progress.get("masked") != "completed":
                    update_progress(progress, exp_rel_path, "masked", "in progress", progressLog_path, saveProgress=True)
                    if 'masked' in hdf5_group: del hdf5_group['masked']
                    masked_dataset = hdf5_group.create_dataset('masked', shape=hdf5_group[STEPS[curr_i-1]].shape, dtype=hdf5_group[STEPS[curr_i-1]].dtype)
                    mask_stack(exp_path, hdf5_group[STEPS[curr_i-1]], masked_dataset, mask_paths, chunksize=512)
                    hdf5_group.file.flush()
                    update_progress(progress, exp_rel_path, "masked", "completed", progressLog_path, saveProgress=True)

            elif step == "hemo_corrected":
                # Denoise & Compress (SVD) + Hemodynamic Correction: take the prev step in steps to be the dataset to hemoCorrect.
                if exp_progress.get("hemo_corrected") != "completed":
                    update_progress(progress, exp_rel_path, "hemo_corrected", "in progress", progressLog_path, saveProgress=True)

                    mean_frames = np.mean(hdf5_group[STEPS[curr_i-1]][:40], axis=0)
                    brain_mask = normalize_arr(np.load(os.path.join(exp_path, "brain_mask.npy")))
                    U, SVT = hdf5_approximate_svd(hdf5_group[STEPS[curr_i-1]], mean_frames, mask=brain_mask) 

                    if 'hemo_corrected' in hdf5_group: del hdf5_group['hemo_corrected']
                    new_shape = (hdf5_group[STEPS[curr_i-1]].shape[0], *hdf5_group[STEPS[curr_i-1]].shape[2:])
                    hemoCorrected_dataset = hdf5_group.create_dataset('hemo_corrected', shape=new_shape, dtype='float64')
                    hemoCorrect_stack(hemoCorrected_dataset, U, SVT[:,0::2], SVT[:,1::2])
                    hdf5_group.file.flush()
                    update_progress(progress, exp_rel_path, "hemo_corrected", "completed", progressLog_path, saveProgress=True)

            elif step == "deltaF":
                # Compute deltaF/F: take the prev step in steps to be the dataset to compute deltaF/F over.
                if exp_progress.get("deltaF") != "completed":
                    update_progress(progress, exp_rel_path, "deltaF", "in progress", progressLog_path, saveProgress=True)

                    if 'deltaF' in hdf5_group: del hdf5_group['deltaF']
                    deltaF_dataset = hdf5_group.create_dataset('deltaF', shape=hdf5_group[STEPS[curr_i-1]].shape, dtype='float64')
                    brain_mask = normalize_arr(np.load(os.path.join(exp_path, "brain_mask.npy")))
                    get_hdf5_deltaF(hdf5_group[STEPS[curr_i-1]], deltaF_dataset, mask=brain_mask)
                    hdf5_group.file.flush()
                    update_progress(progress, exp_rel_path, "deltaF", "completed", progressLog_path, saveProgress=True)

            elif step == "trial_averaged":
                # Create and store trial averaged frames: take the prev step in steps to be the dataset to compute trial averages over.
                if exp_progress.get("trial_averaged") != "completed":
                    update_progress(progress, exp_rel_path, "trial_averaged", "in progress", progressLog_path, saveProgress=True)

                    # get the stim times datetime 
                    stim_times_arr = get_stim_times(exp_path, STIM_TYPE)

                    # create dataset in which frames are divided into trials 
                    if 'trial_avg_frames' in hdf5_group: del hdf5_group['trial_avg_frames']
                    trial_avg_frames_dataset = hdf5_group.create_dataset("trial_avg_frames", shape=(25, 640, 540), maxshape=(None, 640, 540), dtype='float32')  
                    if 'trial_avg_xTimes' in hdf5_group: del hdf5_group['trial_avg_xTimes']
                    trial_avg_xTimes_dataset = hdf5_group.create_dataset("trial_avg_xTimes", shape=(45,), maxshape=(None, ), dtype='float64')

                    frameTime_mat_path = get_exp_file_path(exp_path, 'T', dig=True)

                    # get the aligned stims 
                    trial_avg_frames_arr, trial_xTimes_arr = get_trialAvg_stack(frameTime_mat_path, hdf5_group[STEPS[curr_i-1]], stim_times_arr, pre_post_stim=(SECONDS_BEFORE_STIM, SECONDS_AFTER_STIM), png_save_path=exp_path)
                    
                    new_shape_frames = trial_avg_frames_arr.shape
                    new_shape_xTimes = trial_xTimes_arr.shape
                    # Resize the dataset before writing
                    trial_avg_frames_dataset.resize(new_shape_frames)  
                    trial_avg_xTimes_dataset.resize(new_shape_xTimes)
                    # Assign new data
                    trial_avg_frames_dataset[:] = trial_avg_frames_arr
                    trial_avg_xTimes_dataset[:] = trial_xTimes_arr

                    hdf5_group.file.flush()
                    update_progress(progress, exp_rel_path, "trial_averaged", "completed", progressLog_path, saveProgress=True)

        # Add attributes (metadata)
        hdf5_group.attrs["processing_pipeline_run"] = "Nicole_preprocess"
        hdf5_group.attrs["processing_steps_run"] = STEPS
        hdf5_group.attrs["last_modified"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # to save space now that we have everything, keep only the following datasets in the hdf5 file
        if CLEAN_HDF5_WHEN_DONE:
            for step in STEPS:
                if step not in STEPS_TO_KEEP_POST_CLEAN and step in hdf5_group: 
                    del hdf5_group[step]
                    hdf5_group.file.flush()
        
        # Final update: mark the experiment as completed
        update_progress(progress, exp_rel_path, "EXPERIMENT_PROCSSING_STATUS", "completed", progressLog_path, saveProgress=True)

    except Exception as e:
        update_progress(progress, exp_rel_path, "error", str(e), progressLog_path, saveProgress=True)
        raise e
    

if __name__ == "__main__":
    # TODO: make sure to modify ALL global variables at the top of this file! 

    path_to_all = r"D:\wfield\SAMPLE EXPERIMENT"
    progressLog_path = get_progressLog(path_to_all, verbose=True)
    folder_structure = ['experimental_groups', 'LED_brightness', 'animal_experiments'] # if only running a single experiment only have: folder_structure = ["experiments"]
    hdf5_save = 'LED_brightness'
    
    process_experiments(progressLog_path, path_to_all, folder_structure, hdf5_save, Template_analysis)
    