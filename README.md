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

### 4. Run the Project

** Should the blue and violet Frames be off, modify lines 344, 345 of processing.py in the hemodynamic correction function or after the motion correction such that the indices of the blue (typically 0) and violet (typically 1) are switched. 

Now you're ready to run your Python scripts.

### Additional Information
- Make sure your virtual environment is activated every time you want to run your project:
  ```bash
  source venv/bin/activate  # Unix/MacOS
  .\venv\Scripts\activate   # Windows
  ```
