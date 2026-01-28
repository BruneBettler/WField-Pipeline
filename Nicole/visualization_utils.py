import os
import numpy as np
from struct import unpack
import numpy as np
import datetime
from datetime import timedelta
from pipeline_utils import *
from data_extraction import *

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
#from pipeline_processing import *

def animate_image_stack(image_stack, interval_ms=50, cmap='gray',title=None):
    num_frames = image_stack.shape[0]

    fig, ax = plt.subplots()

    if title:
        plt.title(title)

    im = ax.imshow(image_stack[0], cmap=cmap, animated=True)

    ax.set_axis_off()

    text = ax.text(0.02, 0.95, '', transform=ax.transAxes,
                   color='white', fontsize=12, ha='left', va='top',
                   bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))

    def update(frame):
        im.set_array(image_stack[frame])
        text.set_text(f'Frame {frame + 1}/{num_frames}')
        return [im, text]

    ani = FuncAnimation(
        fig, update, frames=num_frames,
        interval=interval_ms, blit=True)

    plt.close(fig)  # prevent duplicate static plot output
    return HTML(ani.to_jshtml())


def save_video_of_stack(image_stack, output_path, fps=20, cmap='viridis', title=None):
    """
    Saves a 3D image stack (T, H, W) as a video file (MP4 or GIF).
    Uses 'ffmpeg' if available, otherwise falls back to 'pillow' (GIF).
    """
    num_frames = image_stack.shape[0]
    
    # Pre-calculate vmin/vmax from a subset for consistent contrast
    subset = image_stack[::max(1, num_frames // 20)]
    vmin = np.nanpercentile(subset, 1)
    vmax = np.nanpercentile(subset, 99)

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(image_stack[0], cmap=cmap, vmin=vmin, vmax=vmax)
    ax.axis('off')
    
    title_text = ax.set_title(f"{title} Frame 0" if title else "Frame 0")

    def animate(i):
        im.set_data(image_stack[i])
        if title:
            title_text.set_text(f"{title} Frame {i}")
        else:
            title_text.set_text(f"Frame {i}")
        return [im, title_text]

    ani = FuncAnimation(fig, animate, frames=num_frames, interval=1000/fps, blit=True)
    
    try:
        if output_path.endswith('.mp4'):
            ani.save(output_path, writer='ffmpeg', fps=fps)
        elif output_path.endswith('.gif'):
            ani.save(output_path, writer='pillow', fps=fps)
        else:
            # Default to mp4 if extension not clear, or append it
            output_path += ".mp4"
            ani.save(output_path, writer='ffmpeg', fps=fps)
        print(f"Saved video to {output_path}")
    except Exception as e:
        print(f"Primary save failed: {e}")
        # Fallback to GIF if MP4 failed (often due to missing ffmpeg)
        fallback_path = os.path.splitext(output_path)[0] + ".gif"
        print(f"Attempting fallback to GIF: {fallback_path}")
        try:
            ani.save(fallback_path, writer='pillow', fps=fps)
            print(f"Saved GIF to {fallback_path}")
        except Exception as e2:
            print(f"Failed to save fallback GIF: {e2}")

    plt.close(fig)


def plot_global_trace(trace_data, output_path, title="Global Average Trace", ylabel="dF/F"):
    """
    Plots a 1D trace and saves it to a file.
    """
    plt.figure(figsize=(12, 4))
    plt.plot(trace_data)
    plt.title(title)
    plt.xlabel("Frame")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved trace plot to {output_path}")
    plt.close()


# function below is from the churchland wfield code # TODO: find a way to better cite this?
def load_dat_frames(filename, nframes=None, offset=0, shape=None, dtype='uint16'):
    '''
    Loads image frames from a binary file.

    Inputs:
        filename (str)       : fileformat convention, file ends in _NCHANNELS_H_W_DTYPE.dat
        nframes (int)        : number of frames to read (default is None: the entire file)
        offset (int)         : offset frame number (default 0)
        shape (list|tuple)   : dimensions (NCHANNELS, HEIGHT, WIDTH) default is None
        dtype (str)          : datatype (default uint16)
    Returns:
        An array with size (NFRAMES,NCHANNELS, HEIGHT, WIDTH).

    Example:
        dat = load_dat(filename)
    '''
    if not os.path.isfile(filename):
        raise OSError('File {0} not found.'.format(filename))
    if shape is None or dtype is None:  # try to get it from the filename
        dtype, shape, _ = _parse_binary_fname(filename, shape=shape, dtype=dtype)
    if type(dtype) is str:
        dt = np.dtype(dtype)
    else:
        dt = dtype

    if nframes is None:
        # Get the number of samples from the file size
        nframes = int(os.path.getsize(filename) / (np.prod(shape) * dt.itemsize))
    framesize = int(np.prod(shape))

    offset = int(offset)
    with open(filename, 'rb') as fd:
        fd.seek(offset * framesize * int(dt.itemsize))
        buf = np.fromfile(fd, dtype=dt, count=framesize * nframes)
    buf = buf.reshape((-1, *shape), order='C')

    return buf

def _parse_binary_fname(fname, lastidx=None, dtype='uint16', shape=None, sep='_'):
    '''
    Gets the data type and the shape from the filename
    This is a helper function to use in load_dat.

    out = _parse_binary_fname(fname)

    With out default to:
        out = dict(dtype=dtype, shape = shape, fnum = None)
    '''
    fn = os.path.splitext(os.path.basename(fname))[0]
    fnsplit = fn.split(sep)
    fnum = None
    if lastidx is None:
        # find the datatype first (that is the first dtype string from last)
        lastidx = -1
        idx = np.where([not f.isnumeric() for f in fnsplit])[0]
        for i in idx[::-1]:
            try:
                dtype = np.dtype(fnsplit[i])
                lastidx = i
            except TypeError:
                pass
    if dtype is None:
        dtype = np.dtype(fnsplit[lastidx])
    # further split in those before and after lastidx
    before = [f for f in fnsplit[:lastidx] if f.isdigit()]
    after = [f for f in fnsplit[lastidx:] if f.isdigit()]
    if shape is None:
        # then the shape are the last 3
        shape = [int(t) for t in before[-3:]]
    if len(after) > 0:
        fnum = [int(t) for t in after]
    return dtype, shape, fnum

def load_dat_analog(file_path):
    """Convert a .dat file to a numpy ndarray\n
        First we read the file header.
        The first data [double] is representing the amount of data in the header
        The second double is the time of acquisition onset on first run
        The Third double is the number of recorded analog channels + timestamps
        The Fourth or last double is the number of values to read (set to inf since absolute recording duration is unknown at this point)

        After the Data is written as uint16"""
    with open(file_path,'rb') as fd:
        tstamp = unpack("d", fd.read(8))[0]
        onset = unpack("d", fd.read(8))[0]
        nchannels = int(unpack("<d", fd.read(8))[0])
        nsamples = unpack("<d", fd.read(8))[0]
        dat = np.fromfile(fd,dtype='uint16')
        dat = dat.reshape((-1,nchannels)).T
    
    return dat,dict(baseline=tstamp,
                    onset=onset,
                    nchannels=nchannels,
                    nsamples=nsamples)


## --------------------- ALIGNMENT FUNCTIONS -------------------------------

# 1.1: FRAME TIME ALIGNMENT + STIMULUS STIM TIME (corrected for delay between wf and stim)
def alignment_1_1(experiment_path, stim_type):
    '''
    Function is the same as the one currently used in the processing pipeline
    Function returns the start and stop times of the stimulus presentation. To be used with datetime frametimes
    Function is the same as get_stim_times from processing.py
    '''
    # determine whether the experiment data_format == "NEW" (after feb 12th 2025) or "OLD"
    data_format = get_dataFormat_type(experiment_path)

    # get stim_pulse_times 
    stim_pulse_times = get_stim_pulse_times(experiment_path, stim_type, data_format)
    
    stim_pulseTimes_array = []

    if stim_type in ["LED", "AUDIO"]:
        stim_time = get_stim_duration(experiment_path, stim_type)
        # Add STIM_TIME seconds to every second entry starting from the first
        for i in range(1, len(stim_pulse_times), 2):  # Start at index 1, step by 2
            stim_pulseTimes_array.append([stim_pulse_times[i-1], (stim_pulse_times[i-1] + timedelta(seconds=stim_time))])           
    
    elif stim_type == "OLFAC":
        mix_dur, odor_dur = get_odor_MixandDur(experiment_path, data_format)
        for i in range(1, len(stim_pulse_times), 2):  # Start at index 1, step by 2
            odor_start = stim_pulse_times[i-1] + timedelta(seconds=mix_dur)
            stim_pulseTimes_array.append([odor_start, odor_start + timedelta(seconds=odor_dur)])

    # OPTIONAL BUT RECOMMENDED: correct for the delay between wf and stim computer:
    if data_format == "OLD":
        """_, analog_dict = load_dat_analog(get_exp_file_path(experiment_path, 'A'))
        analog_onset = mattime_to_hour(analog_dict['onset'])
        delay_seconds = get_alignment_delay(experiment_path, data_format, stim_type, analog_onset)
        for pulse in stim_pulseTimes_array:
            pulse[0] += timedelta(seconds = delay_seconds) 
            pulse[1] += timedelta(seconds = delay_seconds)
        delay_seconds_n = get_alignment_delay(experiment_path, data_format, stim_type)
        for pulse in stim_pulseTimes_array:
            pulse[0] += timedelta(seconds = delay_seconds_n) 
            pulse[1] += timedelta(seconds = delay_seconds_n)"""
        pass
    else:
        delay_seconds = get_alignment_delay(experiment_path, data_format, stim_type)
        for pulse in stim_pulseTimes_array:
            pulse[0] -= timedelta(seconds = delay_seconds) 
            pulse[1] -= timedelta(seconds = delay_seconds)

    return np.array(stim_pulseTimes_array) # returned array is an array of datetime objects


# 1.2: FRAME TIME ALIGNMENT + WF_EXP_START_TIME
def alignment_1_2(experiment_path, stim_type):
    DATA_FORMAT = get_dataFormat_type(experiment_path)

    analog_data, analog_dict = load_dat_analog(get_exp_file_path(experiment_path, 'A'))

    analog_x_values = np.arange(len(analog_data[1])) / 1000 # get time in seconds 
    wf_exp_stim_pulses = normalize_arr(analog_data[4])

    analog_x_stepSize = analog_x_values[1]

    analog_onset = mattime_to_hour(analog_dict['onset'])
    wf_x_times = [analog_onset + timedelta(seconds=s) for s in analog_x_values]

    # now that we have the time of the wfield computer, let's get the start and stop times of the stimulus pulses from the analog computer
    # if we're working with the "LED" stimulus, we need to first correct the length of the 
    # start by turning the array into a binary signal 

    # find the index of every first 1 occuring after a 0 
    corrected_wf_experimental_stim = np.zeros_like(wf_exp_stim_pulses)
    corrected_wf_experimental_stim[1:] = (wf_exp_stim_pulses[1:] >= 0.5) & (wf_exp_stim_pulses[:-1] <= 0.5)
    corrected_wf_experimental_stim[0] = wf_exp_stim_pulses[0]
    # for each start index (pusle), determine and add the end index
    indices = np.where(corrected_wf_experimental_stim == 1)[0]

    # extract all the start and stop locations and put into an array of tuples 
    stim_time_tuples = []
    stim_data = scipy.io.loadmat(get_exp_file_path(experiment_path, 'M'), simplify_cells=True)
    stim_configs = None
    STIM_TIME = get_stim_duration(experiment_path, stim_type)
    if DATA_FORMAT == "OLD":
        stim_configs = pickle.loads(stim_data['configs'].tobytes())
    elif DATA_FORMAT == "NEW":
        stim_configs = stim_data['configs']

    if stim_type in ["LED", "AUDIO"]:
        for i in indices:
                stim_time_tuples.append([wf_x_times[i], wf_x_times[i+int((1000 * STIM_TIME))]])
    
    elif stim_type == "OLFAC":
        for i in indices:
                odor_start_i = i + int(stim_configs['experiment_config']['MIX_DURATION'] / analog_x_stepSize)
                odor_stop_i = odor_start_i + int(stim_configs['experiment_config']['ODOR_DURATION'] / analog_x_stepSize)
                stim_time_tuples.append([wf_x_times[odor_start_i], wf_x_times[odor_stop_i]])

    if DATA_FORMAT == "OLD":
        delay_seconds = get_alignment_delay(experiment_path, DATA_FORMAT, stim_type)
        for pulse in stim_time_tuples:
            pulse[0] += timedelta(seconds = delay_seconds) 
            pulse[1] += timedelta(seconds = delay_seconds)
    return np.array(stim_time_tuples)


# TODO: - consider using tstim here ?
# 2.1: FRAME TIME ALIGNMENT + WF INDICES (end-aligned / pushed back)
def alignment_2_1(experiment_path, stim_type, frame_num, camera_Hz):
    STIM_TIME = get_stim_duration(experiment_path, stim_type)
    analog_data, _ = load_dat_analog(get_exp_file_path(experiment_path, 'A'))
    
    analog_x_values = np.arange(len(analog_data[1])) / 1000 # get time in seconds 
    wf_exp_stim_pulses = normalize_arr(analog_data[4])
    frame_x_signal = np.arange(0, frame_num) / (camera_Hz / 2) # since 40 Hz is the camera but each channel has frames captured at 20Hz 

    analog_x_stepSize = analog_x_values[1]

    # shift the analog_x_values based on the last frame (first determine which frame is last)
    diff_blue_aligned = (analog_x_values[-1] - frame_x_signal[-1])
    blue_aligned_analog_x = analog_x_values - diff_blue_aligned
    #violet_aligned_analog_x = analog_x_values - diff_blue_aligned + 1/40

    shifted_wf_x_values = blue_aligned_analog_x

    # now that we have the time of the wfield computer, let's get the start and stop times of the stimulus pulses from the analog computer
    # if we're working with the "LED" stimulus, we need to first correct the length of the 
    # start by turning the array into a binary signal 
    
    # find the index of every first 1 occuring after a 0 
    corrected_wf_experimental_stim = np.zeros_like(wf_exp_stim_pulses)
    corrected_wf_experimental_stim[1:] = (wf_exp_stim_pulses[1:] >= 0.5) & (wf_exp_stim_pulses[:-1] <= 0.5)
    corrected_wf_experimental_stim[0] = wf_exp_stim_pulses[0]
    # for each start index (pusle), determine and add the end index
    indices = np.where(corrected_wf_experimental_stim == 1)[0]

    
    # extract all the start and stop locations and put into an array of tuples 
    stim_time_tuples = []
    if stim_type in ["LED", "AUDIO"]:
        for i in indices:
            stim_time_tuples.append((shifted_wf_x_values[i], shifted_wf_x_values[i+int((1000 * STIM_TIME))]))
    
    elif stim_type == "OLFAC":
        stim_data = scipy.io.loadmat(get_exp_file_path(experiment_path, 'M'), simplify_cells=True)
        stim_configs = stim_data['configs']
        for i in indices:
                odor_start_i = i + int(stim_configs['experiment_config']['MIX_DURATION'] / analog_x_stepSize)
                odor_stop_i = odor_start_i + int(stim_configs['experiment_config']['ODOR_DURATION'] / analog_x_stepSize)
                stim_time_tuples.append((shifted_wf_x_values[odor_start_i], shifted_wf_x_values[odor_stop_i]))
    
    return np.array(stim_time_tuples)
    
# 2.2: FRAME TIME ALIGNMENT + STIM INDICES (stop to blue[-1]-aligned / pushed back)
def alignment_2_2(experiment_path, stim_type, frame_num, camera_Hz, LED_channel=1):
    STIM_TIME = get_stim_duration(experiment_path, stim_type)
    # _LED_channel variable is for the stim_type == "LED" in which there are two stimuli
    stim_data = scipy.io.loadmat(get_exp_file_path(experiment_path, 'M'), simplify_cells=True)
    stim_sync = stim_data['sync']

    stim_x_values = np.arange(len(stim_sync[:,3])) / 10000

    if stim_type == "LED" and LED_channel == 2: 
        # DFAULT: channel 2 is the stimulus presentation with the correct duration
        stim_pulses = normalize_arr(stim_sync[:,2])
        stim_pulses = np.where(np.array(stim_pulses) > 0.5, 1, 0) # convert the array into a binary array for readability
    if (stim_type == "LED" and LED_channel == 1) or stim_type == "AUDIO":
        # channel 1 has the stimulus presentation onset (wrong falling edge but similar shape to the analog file)
        stim_uncorrected_rising = normalize_arr(stim_sync[:,1]) 
        stim_uncorrected_rising = np.where(np.array(stim_uncorrected_rising) > 0.5, 1, 0) # convert the array into a binary array for readability
        # correct the rising phase
        stim_pulses = np.zeros_like(stim_uncorrected_rising)
        stim_pulses[1:] = (stim_uncorrected_rising[1:] == 1) & (stim_uncorrected_rising[:-1] == 0)
        stim_pulses[0] = stim_uncorrected_rising[0]
        indices = np.where(stim_pulses == 1)[0]
        for i in indices:    # since our sampling rate is 10000 Hz, 0.5 seconds would be 5000 indices away. 
            stim_on = np.ones(int(10000 * STIM_TIME))
            stim_pulses[i:i+int((10000 * STIM_TIME))] = stim_on
            
    if stim_type == 'OLFAC':
        stim_data = scipy.io.loadmat(get_exp_file_path(experiment_path, 'M'), simplify_cells=True)
        stim_configs = stim_data['configs']
        # channel 1 contains the stimulus presentation with cleaner duration 
        # channel 2 contains noise
        # channel 5 contains the photodiode signal (?)
        # channel 6 contains the clean photodiode response (?)
        stim_pulses = normalize_arr(stim_sync[:,1])
        stim_pulses = np.where(np.array(stim_pulses) > 0.5, 1, 0) # convert the array into a binary array for readability
        # clean these so they are at the right time (we only want to include the odor delivery)
        get_indices = np.zeros_like(stim_pulses)
        get_indices[1:] = (stim_pulses[1:] == 1) & (stim_pulses[:-1] == 0)
        get_indices[0] = stim_pulses[0]
        indices = np.where(get_indices == 1)[0]
        for i in indices:    # since our sampling rate is 10000 Hz, 1 second would be 10000 indices away. 
                stim_on = np.ones(int(10000 * stim_configs['experiment_config']['ODOR_DURATION'])) # time the odor is being delivered 
                delivery_start_i = i + int((10000 * stim_configs['experiment_config']['MIX_DURATION']))
                stim_pulses[delivery_start_i:delivery_start_i+len(stim_on)] = stim_on
    
    stim_stop_c4_idx = np.where(normalize_arr(stim_sync[:,4]) > 0.5)[0][0] 
    stim_exp_stop_i_time = stim_x_values[stim_stop_c4_idx]

    frame_x_signal = np.arange(0, frame_num) / (camera_Hz / 2) # since 40 Hz is the camera but each channel has frames captured at 20Hz 

    frames = load_dat_frames(get_exp_file_path(experiment_path, 'F'))
    num_dark_blue, num_dark_violet = get_darkFrame_num(frames)

    last_nonDark_blue = frame_x_signal[-(num_dark_blue + 1)]
    last_nonDark_violet = frame_x_signal[-(num_dark_violet + 1)] + 1/40

    frameTimes, _ = get_frameTime_data(get_exp_file_path(experiment_path, 'T'), channel_num=2, returnType='datetime')
    blue_fTimes = frameTimes[0]
    violet_fTimes = frameTimes[1] 
    CONSISTENT_FAME_NUM = (frame_num) == (blue_fTimes.shape[0] + violet_fTimes.shape[0])
    if CONSISTENT_FAME_NUM: # the last non-dark frame is violet 
        diff = stim_exp_stop_i_time - last_nonDark_violet
    else: # the last non-dark frame is blue
        diff = stim_exp_stop_i_time - last_nonDark_blue

    aligned_stim_time = stim_x_values - diff

    # now that we have the time of the stim computer, let's get the start and stop times of the stimulus pulses from the stim computer
    # extract all the start and stop locations and put into an array of tuples
    rising_edges = np.where((stim_pulses[:-1] == 0) & (stim_pulses[1:] == 1))[0] + 1
    falling_edges = np.where((stim_pulses[:-1] == 1) & (stim_pulses[1:] == 0))[0] + 1

    # Pair them into tuples
    stim_time_tuples = []
    for i in range(len(rising_edges)):
        stim_time_tuples.append((aligned_stim_time[rising_edges[i]], aligned_stim_time[falling_edges[i]]))    
    
    return np.array(stim_time_tuples)

