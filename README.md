# Welcome! 

Possible preprocessing steps include the following:
- "raw_frames"
- "motion_corrected"
- "masked"
- "hemo_corrected"
- "deltaF"
- "zScored" (not yet completed but can be added!)
- "trial_averaged"
        

When running a pipeline, "raw_frames" must always be present in steps. Also make sure that the input to trial average is an array with 3 dimensions (nFrames, H, W). If you omit the hemo_correct function that naturally makes this conversion, make sure to reduce dimension on your end!  

## Project Structure

This repository is organized into two main folders:

*   **`Nicole/`**: Contains the core processing libraries, algorithms, and helper functions.
    *   `pipeline_processing.py`: Main logic for motion correction, hemodynamics, etc.
    *   `pipeline_utils.py`: Shared utilities (e.g., frame intensity checks).
    *   `visualization_utils.py`: Plotting and video generation tools.
*   **`Vanessa/`**: Contains the execution scripts and run configurations.
    *   `run_pipeline.py`: The entry point script to run the analysis.

## Getting Started

### 1. Set Up Virtual Environment (Optional but Recommended)

To keep things organized, create and activate a Python virtual environment:

**On Unix/MacOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows:**
```bash
python -m venv venv
.\venv\Scripts\activate
```

### 3. Install Required Packages

Use `pip` to install all the dependencies from `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 4. Run the Vanessa Pipeline

**Option A: GUI Launcher (Recommended)**
The easiest way to process one or multiple sessions is via the graphical interface.

```bash
cd Vanessa
python pipeline_launcher_gui.py
```
*   **Batch Processing**: Select multiple session folders to run in sequence.
*   **Mask Drawing**: Use the "Draw Mask" button to create masks for sessions before or during processing. Supports masking on Raw Data (.dat) if HDF5 files are not yet created.
*   **Resumable**: The pipeline automatically checks for existing output files and skips completed steps.

**Option B: Command Line**
To run the full workflow for a single session:

```bash
cd Vanessa
python process_and_align.py --data_dir "D:\Path\To\Experiment\Folder"
```

The pipeline performs the following steps:
1.  **Load Raw Frames**: Loads `.dat` files into an HDF5 file.
    *   *Automatic Frame Detection*: Checks Blue/Violet order via intensity.
2.  **Motion Correction**: Aligns frames to remove motion artifacts.
3.  **Masking**: Checks for `brain_mask.npy`. If missing, the GUI Launcher will prompt you to create one.
4.  **Hemodynamic Correction**: Removes hemodynamic artifacts using the Violet channel.
5.  **Delta F/F**: Calculates relative fluorescence change.
6.  **Alignment**: Synchronizes analog triggers (licks, wheel) and eye camera with WField frames.
7.  **Verification**: Generates a side-by-side video of behavior and brain activity.

### Troubleshooting Channel Order (Blue vs Violet)
The pipeline now uses an automatic intensity check (`check_blue_is_first_via_intensity`) to ensure Channel 0 is always Blue (Functional) and Channel 1 is always Violet (Ref/Hemo). If you suspect the channels are still swapped (e.g., if the DeltaF signal looks inverted), check the console output for "Detected Violet-First" messages or inspect the `Blue_vs_Violet_Trace.png` output.

### Additional Information
- Make sure your virtual environment is activated every time you want to run your project:
  ```bash
  source venv/bin/activate  # Unix/MacOS
  .\venv\Scripts\activate   # Windows
  ```
