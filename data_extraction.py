# Author: Brune Bettler
# Last Modified: 2025-03-17

from pipeline_utils import get_exp_file_path, normalize_arr, mattime_to_hour
import scipy
import pickle
import numpy as np
import datetime
import re

def get_dataFormat_type(experiment_path):
    '''
    Function returns the string "NEW" if the experiment was made on or after February 12th 2025 and "OLD" otherwise 
    '''
    # check the date in the .mat file name 
    mat_file_path = get_exp_file_path(experiment_path, "M", dig=True)
    # Extract the first 8 digits of the 15-digit number and convert to datetime
    file_date = datetime.datetime.strptime(re.search(r'(\d{8})\d{7}', mat_file_path).group(1), '%Y%m%d')

    # Compare with February 12, 2025
    comparison_date = datetime.datetime(2025, 2, 12)
    data_format = None
    if file_date < comparison_date:
        data_format = "OLD"
    else:
        data_format = "NEW"
    
    return data_format

def get_frameTime_data(frameTime_matFile_path, channel_num=2, returnType='original'):
    ''' File data in frameTimes.mat
    'removedFrames': single value
    'frameTimes':
    'preStim':
    'postStim':
    'imgSize': 1,4 array

    channel = 2 if blue and violet frames were recorded 
    type = 'datetime' for datetime objects, 'original' for the original matlab format type 
    '''
    fTime_data = scipy.io.loadmat(frameTime_matFile_path)
    
    raw_frameTimes = fTime_data['frameTimes']
    removed_frames = fTime_data['removedFrames'][0][0]
    pre_post_stim = (fTime_data['preStim'][0][0], fTime_data['postStim'][0][0])
    stim_imgSize = fTime_data['imgSize'][0]

    if returnType == 'datetime':
        # convert matlab time into datetime time 
        raw_frameTimes = np.array([mattime_to_hour(frame_time[0]) for frame_time in raw_frameTimes]) 
    elif returnType == 'original':
        raw_frameTimes = np.squeeze(raw_frameTimes, -1) # shape (2, frameNum, 1) --> (2, frameNum)


    ftimes = []
    if channel_num == 2: # we have access to blue and violet frame times 
        ftimes.append(raw_frameTimes[::2]) # blue frame occurs first
        ftimes.append(raw_frameTimes[1::2])
        # ftimes: [blue_fTimes, violet_fTimes]
        min_frames = min(len(ftimes[0]), len(ftimes[1]))
        ftimes[0] = ftimes[0][:min_frames]
        ftimes[1] = ftimes[1][:min_frames] # shape (2, frameNum)
    else:
        ftimes = raw_frameTimes

    return np.array(ftimes), {'removed_frames': removed_frames, 'pre_post_stim': pre_post_stim, 'stim_img_size': stim_imgSize}

def find_rising_edges(arr):
    """
    Convert an array into a binary array where values above 0.5 become 1 and values below 0.5 become 0.
    Then, return the indices where the signal transitions from 0 to 1.

    Parameters:
        arr (numpy array or list): Input array of numerical values.

    Returns:
        numpy array: Indices where the binary signal transitions from 0 to 1.
    """
    # Convert to numpy array if it's not already
    arr = np.array(arr)
    
    # Step 1: Convert to binary (threshold at 0.5)
    binary_arr = (arr > 0.5).astype(int)
    
    # Step 2: Find rising edge indices (where it transitions from 0 to 1)
    rising_edges = np.where((binary_arr[:-1] == 0) & (binary_arr[1:] == 1))[0] + 1

    return rising_edges

def get_alignment_delay(exp_path, data_format, stim_type, analog_onset=None):
    '''
    Returns the delay in seconds between the stim and wf computer as determined by
    stim pulses from the stim computer and the analog computer copy on the stim .mat file 
    '''
    # get path to stim computer .mat file
    matlab_data_path = get_exp_file_path(exp_path, 'M', dig=True)

    # get stim_configs from the matlab file
    stim_data = scipy.io.loadmat(matlab_data_path, simplify_cells=True)
    stim_sync = stim_data['sync']
    if not analog_onset: 
        if stim_type == "LED":
            rising_stim_pulse = find_rising_edges(normalize_arr(stim_sync[:,2]))

        elif stim_type == "OLFAC":
            rising_stim_pulse = find_rising_edges(normalize_arr(stim_sync[:,6]))
        
        rising_wf_pulse = find_rising_edges(normalize_arr(stim_sync[:,1]))
        stim_x_values = np.arange(stim_sync.shape[0]) / 10000 # TODO make this variable based on the experiment just in case
        rising_lags = np.abs(stim_x_values[rising_stim_pulse] - stim_x_values[rising_wf_pulse])
        return np.mean(rising_lags)
    else:
        stim_experiment_start_timestamp = pickle.loads(stim_data['experiment_start_timestamp'].tobytes())
        stim_onset = datetime.datetime.fromtimestamp(stim_experiment_start_timestamp)
        computer_time_diff = (analog_onset - stim_onset).total_seconds()

        return computer_time_diff    

def get_odor_MixandDur(exp_path, data_format):
    # get path to stim computer .mat file
    matlab_data_path = get_exp_file_path(exp_path, 'M', dig=False)

    # get stim_configs from the matlab file
    if data_format == "OLD":
        stim_data = scipy.io.loadmat(matlab_data_path, squeeze_me=True)
        stim_configs = pickle.loads(stim_data['configs'].tobytes())
    elif data_format == "NEW":
        stim_data = scipy.io.loadmat(matlab_data_path, simplify_cells=True)
        stim_configs = stim_data['configs'] 
                
    return (stim_configs['experiment_config']['MIX_DURATION'], stim_configs['experiment_config']['ODOR_DURATION'])
    
def get_stim_duration(exp_path, stim_type):
    data_format = get_dataFormat_type(exp_path)
    # get path to stim computer .mat file
    matlab_data_path = get_exp_file_path(exp_path, 'M', dig=True)

    # get stim_configs from the matlab file
    stim_data = scipy.io.loadmat(matlab_data_path, simplify_cells=True)
    stim_configs = None
    if data_format == "NEW":
        stim_configs = stim_data['configs'] 
    elif data_format == "OLD":
        stim_configs = pickle.loads(stim_data['configs'].tobytes())
                
    if stim_type == 'LED':
        return stim_configs['experiment_config']['FLASH_TIME']
    elif stim_type == "OLFAC":
        return stim_configs['experiment_config']['ODOR_DURATION']
    elif stim_type == "AUDIO":
        return stim_configs['experiment_config']['TONE_TIME']
    
    else: return None

def get_stim_pulse_times(exp_path, stim_type, data_format):
    '''
    stim_type = one of "LED", "OLFAC", "AUDIO"
    data_format = "NEW" if made after feb 14th 2025, "OLD" otherwise
    '''
    # get path to stim computer .mat file
    matlab_data_path = get_exp_file_path(exp_path, 'M', dig=True)

    if matlab_data_path != None:
        stim_data = scipy.io.loadmat(matlab_data_path, simplify_cells=True)
        if data_format == "OLD":
            stim_stim_pulse_info = pickle.loads(stim_data['stimulus_frame_info'].tobytes())
            stim_pulse_times = np.array([datetime.datetime.fromtimestamp(frame_info['time']) for frame_info in stim_stim_pulse_info])
            return stim_pulse_times
        elif data_format == "NEW": 
            stim_stim_pulse_info = stim_data['stimulus_frame_info']
            if stim_type in ["LED", "AUDIO"]:
                stim_pulse_times = np.array([datetime.datetime.fromtimestamp(float(frame_info['time'])) for frame_info in stim_stim_pulse_info])
            elif stim_type == "OLFAC":
                stim_pulse_times = []
                stim_blockStart_times = []
                # there are three array entries per "stim"
                # the first has the start time and block start "index"
                # the second has the odorname but we do not need to store this here as it's already in the recording.stim_configs['experiment_config']['DELIVERED_ODORS'] array
                # the third has the stop time (the block end has the same value as the block start)
                for i, _ in enumerate(stim_stim_pulse_info[0::3]):
                    j = i * 3
                    stim_pulse_times.append(datetime.datetime.fromtimestamp(float((stim_stim_pulse_info[j])['time']))) # start_time
                    stim_blockStart_times.append((stim_stim_pulse_info[j])['block_start']) # block_start 
                    stim_pulse_times.append(datetime.datetime.fromtimestamp(float((stim_stim_pulse_info[j+2])['time']))) # stop_time
                stim_pulse_times = np.array(stim_pulse_times)
            tstim = stim_data['tstim'] 
            return stim_pulse_times
    
    else:
        return None