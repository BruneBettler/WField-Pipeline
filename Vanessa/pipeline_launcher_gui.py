import sys
import os
import glob
import subprocess
import numpy as np
import h5py
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QFileDialog, QListWidget, QListWidgetItem, 
                             QCheckBox, QProgressBar, QTextEdit, QMessageBox, QLineEdit)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QEventLoop

# ==========================================
# Setup Paths for Nicole
# ==========================================
def setup_paths():
    """Add Nicole directory to sys.path"""
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        current_dir = os.getcwd()

    workspace_nicole = r"c:\Users\mlouki1\Desktop\WField-Pipeline\Nicole"
    sibling_nicole = os.path.abspath(os.path.join(current_dir, '..', 'Nicole'))
    
    nicole_path = None
    if os.path.exists(workspace_nicole):
        nicole_path = workspace_nicole
    elif os.path.exists(sibling_nicole):
        nicole_path = sibling_nicole
    
    if nicole_path and os.path.exists(nicole_path):
        if nicole_path not in sys.path:
            sys.path.insert(0, nicole_path)
setup_paths()

try:
    from GUIs.registration_gui import RegistrationGUI
    from pipeline_utils import normalize_arr
except ImportError:
    print("Warning: Could not import Nicole modules. Mask drawing will be disabled.")
    RegistrationGUI = None

class PipelineWorker(QThread):
    progress_signal = pyqtSignal(int)  # 0-100 percentage
    log_signal = pyqtSignal(str)       # Log output
    finished_signal = pyqtSignal()     # All done
    error_signal = pyqtSignal(str)     # Error message
    request_mask_signal = pyqtSignal(str) # Request mask for session path

    def __init__(self, python_exe, script_path, session_paths, flags):
        super().__init__()
        self.python_exe = python_exe
        self.script_path = script_path
        self.session_paths = session_paths
        self.flags = flags
        self.is_running = True
        self.mask_response = None

    def run(self):
        total_sessions = len(self.session_paths)
        
        session_idx = 0
        while session_idx < total_sessions and self.is_running:
            session_path = self.session_paths[session_idx]
            
            self.log_signal.emit(f"\n{'='*40}\nProcessing {session_idx+1}/{total_sessions}: {os.path.basename(session_path)}\n{'='*40}\n")
            
            # Use -u for unbuffered output
            cmd = [self.python_exe, '-u', self.script_path, '--data_dir', session_path] + self.flags
            
            # Session base progress
            session_share = 100.0 / total_sessions
            current_progress_base = session_idx * session_share
            
            steps_completed = 0
            mask_missing_detected = False
            
            try:
                process = subprocess.Popen(
                    cmd, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.STDOUT, 
                    text=True, 
                    bufsize=1,
                    universal_newlines=True,
                    encoding='utf-8', 
                    errors='replace'
                )
                
                for line in process.stdout:
                    if not self.is_running:
                        process.terminate()
                        break
                    
                    line_clean = line.rstrip()
                    self.log_signal.emit(line_clean)
                    
                    # Detect Mask Error
                    if "MASK_MISSING" in line or "No mask found" in line:
                        mask_missing_detected = True

                    # Detect Step Changes for smoother progress
                    if "STARTING: Preprocessing" in line:
                        pass
                    elif "STARTING: Signal Alignment" in line:
                        steps_completed = 1
                    elif "STARTING: Generating Verification Video" in line:
                        steps_completed = 2
                    
                    current_pct = current_progress_base + (steps_completed / 3.0) * session_share
                    self.progress_signal.emit(int(current_pct))
                
                process.wait() # Ensure process is dead

                # Handle Mask Missing
                if mask_missing_detected and self.is_running:
                    self.log_signal.emit("\n!!! Mask Missing Detected. Requesting user input... !!!\n")
                    self.mask_response = None
                    self.request_mask_signal.emit(session_path)
                    
                    # Wait for response from GUI thread
                    while self.mask_response is None and self.is_running:
                         self.msleep(100)
                    
                    if self.mask_response == "RETRY":
                         self.log_signal.emit("\n... User created mask. Retrying session ...\n")
                         continue # Retry same index
                    else:
                         self.log_signal.emit("\n... Mask creation skipped. Moving to next session (if any) ...\n")
                         # Fall through to increment
                
                elif process.returncode != 0:
                    self.log_signal.emit(f"\n!!!! Error processing {session_path}. Exit code: {process.returncode} !!!!\n")
                else:
                    self.log_signal.emit(f"\nSuccessfully finished {session_path}\n")

            except Exception as e:
                self.error_signal.emit(str(e))
                self.log_signal.emit(f"Critical execution error: {str(e)}")
            
            # Increment to next session
            session_idx += 1
            self.progress_signal.emit(int(session_idx * session_share))
            
        self.finished_signal.emit()

    def stop(self):
        self.is_running = False

class PipelineLauncher(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("WField Pipeline Launcher")
        self.resize(1000, 750)
        
        # Determine paths
        self.python_exe = sys.executable
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.process_script = os.path.join(self.current_dir, "process_and_align.py")
        
        self.init_ui()
        
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # 1. Directory Selection
        dir_layout = QHBoxLayout()
        self.dir_label = QLabel("No directory selected")
        select_btn = QPushButton("Select Parent Directory")
        select_btn.clicked.connect(self.select_directory)
        dir_layout.addWidget(select_btn)
        dir_layout.addWidget(self.dir_label)
        dir_layout.addStretch()
        layout.addLayout(dir_layout)
        
        # 2. Session List
        layout.addWidget(QLabel("Select Sessions to Process:"))
        self.session_list = QListWidget()
        self.session_list.setSelectionMode(QListWidget.MultiSelection) # Though we use checkboxes
        layout.addWidget(self.session_list)
        
        # Select All/None buttons
        sel_layout = QHBoxLayout()
        self.btn_check_all = QPushButton("Select All")
        self.btn_check_all.clicked.connect(self.select_all)
        self.btn_check_none = QPushButton("Select None")
        self.btn_check_none.clicked.connect(self.select_none)
        self.btn_refresh = QPushButton("Refresh List")
        self.btn_refresh.clicked.connect(self.scan_directory)
        sel_layout.addWidget(self.btn_check_all)
        sel_layout.addWidget(self.btn_check_none)
        sel_layout.addWidget(self.btn_refresh)
        sel_layout.addStretch()
        layout.addLayout(sel_layout)
        
        # 3. Pipeline Options
        opt_layout = QHBoxLayout()
        self.cb_skip_pre = QCheckBox("Skip Preprocessing")
        self.cb_skip_align = QCheckBox("Skip Alignment")
        self.cb_skip_video = QCheckBox("Skip Video Generation")
        opt_layout.addWidget(QLabel("Options:"))
        opt_layout.addWidget(self.cb_skip_pre)
        opt_layout.addWidget(self.cb_skip_align)
        opt_layout.addWidget(self.cb_skip_video)
        opt_layout.addStretch()
        layout.addLayout(opt_layout)
        
        # 3b. Parameters
        param_layout = QHBoxLayout()
        param_layout.addWidget(QLabel("Trials to Plot (Alignment):"))
        self.txt_trials = QLineEdit("all")
        self.txt_trials.setPlaceholderText("e.g. 'all', '1,3,5', '0'")
        self.txt_trials.setToolTip("Comma-separated list of trial indices to plot, or 'all'")
        param_layout.addWidget(self.txt_trials)
        param_layout.addStretch()
        layout.addLayout(param_layout)
        
        # 4. Controls
        ctrl_layout = QHBoxLayout()
        
        # Mask Button
        self.btn_mask = QPushButton("Draw Mask (Selected)")
        self.btn_mask.setFixedHeight(40)
        self.btn_mask.clicked.connect(self.launch_mask_gui)
        ctrl_layout.addWidget(self.btn_mask)
        
        self.btn_run = QPushButton("RUN PIPELINE")
        self.btn_run.setFixedHeight(40)
        self.btn_run.setStyleSheet("font-weight: bold; background-color: #4CAF50; color: white;")
        self.btn_run.clicked.connect(self.start_processing)
        
        self.btn_stop = QPushButton("STOP")
        self.btn_stop.setFixedHeight(40)
        self.btn_stop.setStyleSheet("font-weight: bold; background-color: #f44336; color: white;")
        self.btn_stop.clicked.connect(self.stop_processing)
        self.btn_stop.setEnabled(False)
        
        ctrl_layout.addWidget(self.btn_run)
        ctrl_layout.addWidget(self.btn_stop)
        layout.addLayout(ctrl_layout)
        
        # 5. Progress
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)
        
        # 6. Log
        layout.addWidget(QLabel("Output Log:"))
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("font-family: Consolas; font-size: 10pt;")
        layout.addWidget(self.log_text)
        
        self.selected_dir = None
        self.worker = None

    def select_directory(self):
        d = QFileDialog.getExistingDirectory(self, "Select Parent Directory (containing Session folders)")
        if d:
            self.selected_dir = d
            self.dir_label.setText(d)
            self.scan_directory()
            
    def scan_directory(self):
        self.session_list.clear()
        if not self.selected_dir:
            return
            
        # List subdirectories
        try:
            items = os.listdir(self.selected_dir)
            potential_dirs = []
            for item in items:
                path = os.path.join(self.selected_dir, item)
                if os.path.isdir(path):
                    potential_dirs.append(path)
            
            # Smart filter: check for Frames_*.dat in immediate directory
            count_found = 0
            for d in potential_dirs:
                # Check if it looks like data
                has_dat = len(glob.glob(os.path.join(d, "Frames_*.dat"))) > 0
                display_text = os.path.basename(d)
                if has_dat:
                     display_text += " [DATA DETECTED]"
                
                item = QListWidgetItem(display_text)
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                if has_dat:
                     item.setCheckState(Qt.Checked)
                     count_found += 1
                else:
                     item.setCheckState(Qt.Unchecked)
                item.setData(Qt.UserRole, d)
                self.session_list.addItem(item)
                
            # Fallback: if no subdirs have data, check if the selected dir IS the data dir
            if count_found == 0:
                 if len(glob.glob(os.path.join(self.selected_dir, "Frames_*.dat"))) > 0:
                     self.session_list.clear()
                     item = QListWidgetItem(os.path.basename(self.selected_dir) + " [DATA DETECTED]")
                     item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                     item.setCheckState(Qt.Checked)
                     item.setData(Qt.UserRole, self.selected_dir)
                     self.session_list.addItem(item)
                     
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def select_all(self):
        for i in range(self.session_list.count()):
            self.session_list.item(i).setCheckState(Qt.Checked)
            
    def select_none(self):
        for i in range(self.session_list.count()):
            self.session_list.item(i).setCheckState(Qt.Unchecked)

    def launch_mask_gui(self, session_path=None):
        """Find selected session, load data (HDF5 or Raw), and open RegistrationGUI"""
        if RegistrationGUI is None:
            QMessageBox.critical(self, "Error", "RegistrationGUI could not be imported.")
            return False

        if not isinstance(session_path, str):
            selected_items = self.session_list.selectedItems()
            if not selected_items:
                QMessageBox.warning(self, "No Selection", "Please click/highlight a session to draw its mask.")
                return False
            session_path = selected_items[0].data(Qt.UserRole)
        
        preprocessed_dir = os.path.join(session_path, "preprocessed_data")
        
        # 1. Try to load from HDF5 if it exists
        blue_img, violet_img = None, None
        loaded_from_hdf5 = False
        
        h5_files = []
        if os.path.exists(preprocessed_dir):
            h5_files = glob.glob(os.path.join(preprocessed_dir, "*.h5*"))
        
        if h5_files:
            try:
                with h5py.File(h5_files[0], 'r') as f:
                    if 'led/motion_corrected' in f: dset = f['led/motion_corrected']
                    elif 'widefield/motion_corrected' in f: dset = f['widefield/motion_corrected']
                    elif 'led/raw_frames' in f: dset = f['led/raw_frames']
                    else: dset = None
                    
                    if dset is not None:
                        blue_img = (normalize_arr(dset[0,0,...]) * 255).astype('uint8')
                        violet_img = (normalize_arr(dset[0,1,...]) * 255).astype('uint8')
                        loaded_from_hdf5 = True
            except Exception as e:
                print(f"Warning: Failed to load existing HDF5: {e}")
        
        # 2. If no HDF5, try loading from RAW .dat files
        if not loaded_from_hdf5:
            dat_pattern = os.path.join(session_path, "Frames_*.dat")
            dat_files = sorted(glob.glob(dat_pattern))
            
            if not dat_files:
                QMessageBox.warning(self, "Missing Data", f"No .h5 (preprocessed) OR .dat (raw) files found in:\n{session_path}")
                return False
                
            try:
                # Parse metadata from first filename
                first_file = dat_files[0]
                dat_info = os.path.basename(first_file).split('_')
                try:
                    channels = int(dat_info[1])
                    H = int(dat_info[2])
                    W = int(dat_info[3])
                    dtype_str = dat_info[4]
                    dtype = np.dtype(dtype_str)
                except IndexError:
                    channels, H, W, dtype = 2, 640, 540, np.uint16
                
                # Load first few frames
                # Shape: [frames, channels, H, W]
                n_frames_to_load = 5
                raw_data = np.memmap(first_file, dtype=dtype, mode='r', shape=(n_frames_to_load, channels, H, W))
                
                # Assuming Blue is Ch0, Violet is Ch1 (standard)
                # If swapped, user will see swapped images, but mask is spatial so it's okay
                b_frame = raw_data[0, 0, :, :]
                v_frame = raw_data[0, 1, :, :]
                
                blue_img = (normalize_arr(b_frame) * 255).astype('uint8')
                violet_img = (normalize_arr(v_frame) * 255).astype('uint8')
                
                del raw_data
                
                # Ensure preprocessed_dir exists so we can save the mask there
                if not os.path.exists(preprocessed_dir):
                    os.makedirs(preprocessed_dir)
                    
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load Raw Data: {e}")
                return False

        # 3. Launch GUI
        original_cwd = os.getcwd()
        try:
            # Change to preprocessed_dir ensures mask is saved there by default
            if os.path.exists(preprocessed_dir):
                os.chdir(preprocessed_dir)
            else:
                os.chdir(session_path) # Fallback
                
            self.mask_window = RegistrationGUI(blue_img, violet_img)
            self.mask_window.show()
            return True
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open GUI: {e}")
            return False
        finally:
            os.chdir(original_cwd)

    def start_processing(self):
        # Gather selected paths
        selected_paths = []
        for i in range(self.session_list.count()):
            item = self.session_list.item(i)
            if item.checkState() == Qt.Checked:
                selected_paths.append(item.data(Qt.UserRole))
                
        if not selected_paths:
            QMessageBox.warning(self, "No Selection", "Please select at least one session folder.")
            return

        # Gather flags
        flags = []
        if self.cb_skip_pre.isChecked(): flags.append('--skip_preprocessing')
        if self.cb_skip_align.isChecked(): flags.append('--skip_alignment')
        if self.cb_skip_video.isChecked(): flags.append('--skip_video')
        
        # Add trials parameter
        t_txt = self.txt_trials.text().strip()
        if t_txt:
            flags.extend(['--trials_to_plot', t_txt])
        
        self.log_text.clear()
        self.progress_bar.setValue(0)
        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.session_list.setEnabled(False)
        self.cb_skip_pre.setEnabled(False)
        self.cb_skip_align.setEnabled(False)
        self.cb_skip_video.setEnabled(False)
        self.txt_trials.setEnabled(False)
        
        self.worker = PipelineWorker(self.python_exe, self.process_script, selected_paths, flags)
        self.worker.log_signal.connect(self.append_log)
        self.worker.progress_signal.connect(self.progress_bar.setValue)
        self.worker.finished_signal.connect(self.processing_finished)
        self.worker.error_signal.connect(self.processing_error)
        self.worker.request_mask_signal.connect(self.handle_mask_request)
        self.worker.start()

    def handle_mask_request(self, session_path):
        reply = QMessageBox.question(self, "Mask Missing", 
                                     f"No mask found for session:\n{os.path.basename(session_path)}\n\n"
                                     "Do you want to draw the mask now?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        
        if reply == QMessageBox.Yes:
            success = self.launch_mask_gui(session_path)
            if success and hasattr(self, 'mask_window'):
                # Block until window closes
                # We use polling loop here to ensure we don't return until truly closed
                # (Signals can sometimes be tricky with garbage collection)
                self.mask_window.setAttribute(Qt.WA_DeleteOnClose)
                while self.mask_window.isVisible():
                    QApplication.processEvents()
                    import time
                    time.sleep(0.05)
                
                self.worker.mask_response = "RETRY"
            else:
                self.worker.mask_response = "SKIP"
        else:
            self.worker.mask_response = "SKIP"

    def stop_processing(self):
        if self.worker:
            self.worker.stop()
            self.append_log("\n--- Stopping... ---\n")
            
    def append_log(self, text):
        self.log_text.append(text)
        # Auto scroll
        sb = self.log_text.verticalScrollBar()
        sb.setValue(sb.maximum())

    def processing_finished(self):
        self.append_log("\nAll tasks finished.")
        self.restore_ui()
        QMessageBox.information(self, "Done", "Processing pipeline completed.")

    def processing_error(self, err):
        self.append_log(f"Error: {err}")
        self.restore_ui()

    def restore_ui(self):
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.session_list.setEnabled(True)
        self.cb_skip_pre.setEnabled(True)
        self.cb_skip_align.setEnabled(True)
        self.cb_skip_video.setEnabled(True)
        self.txt_trials.setEnabled(True)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = PipelineLauncher()
    gui.show()
    sys.exit(app.exec_())
