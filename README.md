MEP Latency Detection Pipeline

This repository contains a Python script for the automatic detection of Motor-Evoked Potential (MEP) latencies from multichannel EMG data. The pipeline is designed to be run from the command line and can process entire folders of data in parallel.

The detection algorithm is based on a derivative-ratio method, incorporates artefact rejection gates, and can automatically distinguish between resting and active motor states based on the input filename.


Features

Automated Task Mode: The script can automatically apply different analysis rules for resting and active state data by searching for a token (e.g., "act") in the filename.

Artefact Rejection: Includes built-in gates to reject trials based on excessive pre-stimulus muscle activity and low peak-to-peak amplitude.

Parallel Processing: Can significantly speed up analysis by processing multiple files simultaneously across different CPU cores.

Configurable Parameters: Key detection parameters, such as the RMS multiplier for onset refinement, can be adjusted via command-line arguments.

Preprocessing: Includes optional 50 Hz mains noise filtering and signal smoothing.


Requirements and Installation

This script requires Python 3 and several common scientific computing libraries.

  pip install numpy pandas scipy joblib
  

Data Structure

For the script to work correctly, your data should be structured as follows:

EMG Data (--in-dir):

Place all your EMG data files in a single folder.

Files must be in the NumPy (.npy) format.

Each file should be a 3D array with the shape: (samples, frames, channels).

Channel File (--channels):

This is a single .npy file containing a list of channel names (strings). The order of names must match the order of channels in the third dimension of your EMG data arrays. 

This file should be saved in a separate folder to your EMG data.

File Naming for auto Task Mode:

If you use --task-mode auto, the script will look for a specific token in the filename to identify "active" recordings. The default token is act.

For example:

participant1_rest_map.npy → will be treated as rest.

participant2_active_map.npy → will be treated as active (if --active-token is "active").

S03_bicep_act.npy → will be treated as active (using the default token act).


Usage

The script is run from the command line. You must provide the input and output directories and the path to your channel file.

Basic Example Command

Here is a typical command to run the pipeline on the example data, using automatic task detection and 4 CPU cores:

python MEP_latency_derivative_ratio.py \
    --in-dir "data/control_data/lats" \
    --out-dir "data/output" \
    --channels "data/channels.npy" \
    --task-mode "auto" \
    --parallel 4

Command-Line Arguments

Argument	Description	Default
--in-dir	(Required) Path to the folder containing your .npy data files.	
--out-dir	(Required) Path to the folder where output .csv files will be saved.	
--channels	(Required) Path to the .npy file containing channel names.	
--task-mode	Sets the rule for pre-stimulus artefact rejection. <br>• rest: Stricter threshold, assumes low baseline EMG. <br>• active: More lenient threshold, assumes tonic muscle contraction. <br>• auto: Chooses the mode for each file based on the --active-token.	"rest"
--parallel	Number of CPU cores to use for processing files in parallel. 0 runs the script serially.	0
--rms-multiplier	A key parameter that refines the final latency candidate. It checks that the RMS amplitude in a window following the candidate onset is at least X times the baseline RMS. Lowering this value makes the check more lenient.	1.5
--active-token	The string to search for in filenames when --task-mode is set to auto.	"act"
--log	Sets the level of detail for console output (DEBUG, INFO, WARNING, ERROR).	"INFO"


The Algorithm at a Glance

The script processes each channel of each trial through the following steps:

Preprocessing: A 50 Hz notch filter is applied to remove mains noise, followed by a rolling average filter to smooth the signal.

Artefact Rejection: Trials are first screened for excessive baseline muscle activity. A second gate then removes trials where the peak-to-peak amplitude of the MEP is not sufficiently larger than the baseline amplitude.

Onset Candidacy: The algorithm calculates the signal's first derivative (change over time). It then slides through the MEP window, calculating the ratio of the mean derivative in the next 5 samples to the previous 5 samples. The point with the maximum ratio is identified as the primary candidate for the MEP onset.

Candidate Refinement: The primary candidate and nearby points are further validated. A candidate is confirmed as the final latency if, among other checks, the root-mean-square (RMS) amplitude of the signal in the window immediately following it is significantly greater than the baseline RMS.

Output: If a valid onset is found, its latency in milliseconds is recorded. If a trial is rejected or an onset cannot be reliably determined, a descriptive string (NaN, null_onset) is recorded instead.


Output Format
The script generates one .csv file for each .npy file found in the input directory.

Filename: The output filename will be the same as the input, but with a _latencies.csv suffix (e.g., participant1_rest_map_latencies.csv).

Structure: The CSV file is in a "wide" format.

Rows correspond to frames (trials), indexed starting from 1.

Columns correspond to the EMG channels.

Values: The cells contain the detected MEP latency in milliseconds, rounded to 3 decimal places. If no valid latency is found, the cell will contain a string descriptor.
