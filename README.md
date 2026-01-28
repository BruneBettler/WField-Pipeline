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

To run the specific analysis workflow located in the `Vanessa` folder, navigate to that directory and execute the pipeline script:

```bash
cd Vanessa
python run_pipeline.py
```

This pipeline performs the following steps:
1.  **Load Raw Frames**: Loads `.dat` files into an HDF5 file.
    *   *Automatic Frame Detection*: The pipeline automatically detects if the Blue or Violet LED fired first based on image intensity and corrects the channel order.
2.  **Motion Correction**: Aligns frames to remove motion artifacts.
3.  **Masking**: Launches a GUI to define the brain mask (User interaction required).
4.  **Hemodynamic Correction**: Removes hemodynamic artifacts using the Violet channel.
5.  **Delta F/F**: Calculates the relative fluorescence change.
6.  **Visualization**: Generates global traces and a video preview.

### Troubleshooting Channel Order (Blue vs Violet)
The pipeline now uses an automatic intensity check (`check_blue_is_first_via_intensity`) to ensure Channel 0 is always Blue (Functional) and Channel 1 is always Violet (Ref/Hemo). If you suspect the channels are still swapped (e.g., if the DeltaF signal looks inverted), check the console output for "Detected Violet-First" messages or inspect the `Blue_vs_Violet_Trace.png` output.

### Additional Information
- Make sure your virtual environment is activated every time you want to run your project:
  ```bash
  source venv/bin/activate  # Unix/MacOS
  .\venv\Scripts\activate   # Windows
  ```
