
"""
Master Pipeline Script
======================
Orchestrates the full processing workflow:
1. Preprocessing (Raw -> Motion Clean -> Hemo -> dF/F)
2. Alignment (Analog -> HDF5 Sync)
3. Verification (Side-by-Side Video)

Usage:
    python process_and_align.py --data_dir "D:\Vanessa_test_data\Tests_Jan23\23-Jan-2026_ledTTL_10random"
"""

import os
import sys
import argparse
import subprocess
import time

def run_step(script_name, args, description):
    print(f"\n{'='*60}")
    print(f"STARTING: {description}")
    print(f"Script: {script_name}")
    print(f"{'='*60}\n")
    
    script_path = os.path.join(os.path.dirname(__file__), script_name)
    
    # Use -u for unbuffered output so the launcher sees it immediately
    cmd = [sys.executable, '-u', script_path] + args
    
    try:
        subprocess.check_call(cmd)
        print(f"\n{'-'*60}")
        print(f"COMPLETED: {description}")
        print(f"{'-'*60}\n")
    except subprocess.CalledProcessError as e:
        print(f"\n!!!! FAILED: {description} (Exit Code: {e.returncode}) !!!!")
        sys.exit(e.returncode)

def main():
    parser = argparse.ArgumentParser(description="Full Processing & Alignment Pipeline")
    parser.add_argument('--data_dir', type=str, required=True, help="Path to raw data directory")
    parser.add_argument('--skip_preprocessing', action='store_true', help="Skip step 1 (Preprocessing)")
    parser.add_argument('--skip_alignment', action='store_true', help="Skip step 2 (Alignment)")
    parser.add_argument('--skip_video', action='store_true', help="Skip step 3 (Video Generation)")
    parser.add_argument('--trials_to_plot', type=str, default='all', help="Trials to plot in alignment step (e.g. 'all', '0,1,2')")
    parser.add_argument('--video_cmap', type=str, default='jet', help="Colormap for verification video (e.g. jet, viridis, plasma)")
    parser.add_argument('--max_frames', type=int, default=2000, help="Max frames for preview videos (default: 2000)")
    
    args = parser.parse_args()
    
    data_dir = args.data_dir
    if not os.path.exists(data_dir):
        print(f"Error: Directory not found: {data_dir}")
        sys.exit(1)

    print(f"Processing Data: {data_dir}")
    
    # --- Step 1: Preprocessing ---
    if not args.skip_preprocessing:
        # Args for preprocessing_pipeline.py
        # defined in: parser.add_argument('--data_dir'...)
        prep_args = ["--data_dir", data_dir, "--video_cmap", args.video_cmap, "--max_frames", str(args.max_frames)]
        run_step("preprocessing_pipeline.py", prep_args, "Preprocessing (Masking, Hemo, DeltaF)")
    
    # --- Step 2: Alignment ---
    if not args.skip_alignment:
        # Args for alignment_pipeline.py
        # Using default signal files (Analog_1.dat) so we just pass data_dir
        align_args = ["--data_dir", data_dir]
        if args.trials_to_plot:
             align_args.extend(["--trials_to_plot", args.trials_to_plot])
        
        run_step("alignment_pipeline.py", align_args, "Signal Alignment & Synchronization")
        
    # --- Step 3: Verification Video ---
    if not args.skip_video:
        vid_args = ["--data_dir", data_dir, "--cmap", args.video_cmap]
        run_step("generate_alignment_video.py", vid_args, "Generating Verification Video")

    print("\n\nAll pipeline steps completed successfully.")

if __name__ == "__main__":
    main()
