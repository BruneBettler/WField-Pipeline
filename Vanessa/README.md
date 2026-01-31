# Vanessa's Widefield Alignment Pipeline

This folder contains the complete pipeline for aligning widefield calcium imaging data with behavioral recordings (Eye Camera) and stimulus triggers (HDF5).

## Pipeline Steps

### 1. Preprocessing (`run_pipeline.py`)
This script uses the `Nicole` library to perform:
- Motion correction on raw frames
- Hemodynamic correction (optional)
- $\Delta F/F$ calculation
- **Output**: `preprocessed_data/*.h5py` files containing the processed frames.

### 2. Temporal Alignment (`alignment_pipeline.py`)
Maps the disparate timelines of the experiment into a single global timeline (10kHz).
- **Inputs**: 
  - `Analog_*.dat` (local triggers)
  - `*.hdf5` (master clock)
  - `frameTimes_*.mat` (frame timestamps)
- **Outputs**: 
  - `alignment/aligned_full_matrix.npy`: A consolidated matrix of all signals.
  - `alignment/channel_names.txt`: Mapping of matrix columns.
  - Verification plots in `alignment/`.

### 3. Video Verification (`generate_alignment_video.py`)
Generates a side-by-side video of the Processed Widefield Frames and the Eye Camera Video to verify synchronization.
- **Features**:
  - Automatically detects eye camera triggers.
  - Corrects for startup offsets using backward alignment (Last Trigger ↔ Last Frame).
  - Handles multi-trial structure.
  - Visualizes frame-by-frame lock.

## Usage

**Run Preprocessing:**
```bash
python run_pipeline.py
```

**Run Alignment:**
```bash
# Uses default path in script or pass arguments
python alignment_pipeline.py --data_dir "D:/path/to/experiment"
```

**Generate Video:**
```bash
python generate_alignment_video.py --data_dir "D:/path/to/experiment"
```

## Authors
- Matthew Loukine (2026)
