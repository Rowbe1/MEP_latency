# MEP Latency Detection Pipeline

This repository contains a Python implementation of a derivative-ratio method for automated detection of motor-evoked potential (MEP) onset latencies from multichannel EMG data.

The pipeline is designed for transcranial magnetic stimulation (TMS) studies involving repeated EMG epochs, including cortical mapping datasets with multiple muscles recorded simultaneously. It processes folders of NumPy `.npy` files and produces one wide-format latency `.csv` file per input file.

The script can be run from the command line, supports optional resampling, can process files in parallel, and includes console reporting of computational runtime.

---

## Features

- **Automated MEP onset latency detection**  
  Detects MEP onset latencies using a derivative-ratio method designed to identify the transition from baseline activity to the rising phase of the MEP.

- **Multichannel EMG support**  
  Processes data shaped as `samples × frames × channels`, allowing simultaneous analysis of multiple muscles.

- **Resting and active task modes**  
  Applies different pre-stimulus artefact rules for resting and active recordings. The script can also automatically infer task mode from the filename.

- **Artefact rejection gates**  
  Includes pre-stimulus RMS screening and an amplitude gate requiring the MEP peak-to-peak amplitude to exceed baseline activity.

- **Sampling-rate-aware parameters**  
  Key timing parameters are specified in milliseconds and converted to samples using the effective sampling rate.

- **Optional resampling**  
  Input data can be analysed at the native sampling rate or resampled to a target sampling rate before latency detection.

- **Parallel processing**  
  Multiple files can be processed simultaneously using `joblib`.

- **Runtime reporting**  
  For each file, the script reports total processing time, time per frame, and time per channel-frame/epoch.

- **Configurable command-line parameters**  
  Detection and preprocessing parameters can be adjusted from the command line for sensitivity analyses.

---

## Requirements

The script requires Python 3 and the following Python packages:

```bash
pip install numpy pandas scipy joblib
```

The script has been developed for offline analysis of epoched EMG data stored as NumPy arrays.

---

## Data structure

### EMG input files

Place the EMG files to be analysed in a single input folder.

Each EMG file must be a NumPy `.npy` file containing an array with shape:

```text
samples × frames × channels
```

For example, an 8-channel mapping file containing 80 frames sampled at 2 kHz for 0.75 seconds per frame would have shape:

```text
1500 × 80 × 8
```

If a file contains only a 2D array with shape:

```text
samples × frames
```

the script treats it as single-channel data.

### Channel file

The `--channels` argument should point to a `.npy` file containing a list of channel names.

For example:

```python
["FDI", "APB", "ADM", "EDC", "FDS", "TB", "BB", "AD"]
```

The order of channel names must match the channel order in the third dimension of the EMG arrays.

The channel file can be stored separately from the input EMG folder.

---

## File naming for automatic task mode

If `--task-mode auto` is used, the script checks each filename for a token indicating that the recording was performed during an active task.

The default active token is:

```text
act
```

Examples:

```text
participant1_rest_map.npy       -> analysed as rest
participant2_active_map.npy     -> analysed as active if --active-token active
S03_bicep_act.npy               -> analysed as active using the default token act
```

The active token is matched as a standalone text chunk, so filenames should be named clearly.

---

## Basic usage

Run the script from the command line by providing:

1. an input folder containing `.npy` EMG files,
2. an output folder for latency `.csv` files,
3. a channel-name `.npy` file.

Example:

```bash
python MEP_latency_derivative_ratio.py \
    --in-dir "data/control_data/lats" \
    --out-dir "data/output" \
    --channels "data/channels.npy" \
    --task-mode auto \
    --parallel 4
```

To run serially rather than in parallel:

```bash
python MEP_latency_derivative_ratio.py \
    --in-dir "data/control_data/lats" \
    --out-dir "data/output" \
    --channels "data/channels.npy" \
    --task-mode auto \
    --parallel 0
```

---

## Optional resampling

By default, the script analyses data at the native sampling rate specified by `--fs`.

For example, to analyse native 2 kHz data:

```bash
python MEP_latency_derivative_ratio.py \
    --in-dir "data/control_data/lats" \
    --out-dir "data/output" \
    --channels "data/channels.npy" \
    --fs 2000 \
    --task-mode auto
```

To resample 2 kHz data to 5 kHz before latency detection:

```bash
python MEP_latency_derivative_ratio.py \
    --in-dir "data/control_data/lats" \
    --out-dir "data/output" \
    --channels "data/channels.npy" \
    --fs 2000 \
    --resample-to-hz 5000 \
    --task-mode auto
```

Resampling is performed using `scipy.signal.resample_poly` along the sample axis only, preserving the frame and channel dimensions.

---

## Command-line arguments

| Argument | Description | Default |
|---|---|---|
| `--in-dir` | Required. Folder containing input `.npy` EMG files. | None |
| `--out-dir` | Required. Folder where output `.csv` files will be saved. | None |
| `--channels` | Required. Path to `.npy` file containing channel names. | None |
| `--fs` | Native input sampling rate in Hz before optional resampling. | `2000` |
| `--resample-to-hz` | Optional target sampling rate in Hz. If omitted, native sampling rate is used. | `None` |
| `--task-mode` | Task mode for pre-stimulus artefact rejection: `rest`, `active`, or `auto`. | `rest` |
| `--active-token` | Filename token used to identify active recordings when `--task-mode auto` is used. | `act` |
| `--parallel` | Number of parallel workers. Use `0` for serial processing. | `0` |
| `--log` | Console logging level: `DEBUG`, `INFO`, `WARNING`, or `ERROR`. | `INFO` |

---

## Key detection parameters

The following parameters can be adjusted from the command line for sensitivity analyses.

| Argument | Description | Default |
|---|---|---|
| `--ptp-factor` | MEP peak-to-peak amplitude must exceed this multiple of baseline peak-to-peak amplitude. | `1.1` |
| `--derivative-block-ms` | Window length, in milliseconds, used before and after each candidate point for derivative-ratio calculation. | `2.5` |
| `--derivative-ratio-thresh` | Candidate plateau threshold expressed as a fraction of the maximum derivative ratio. | `0.85` |
| `--search-back-factor` | Search-back limit expressed as a multiple of the peak-to-trough distance. | `1.75` |
| `--peak2trough-min-ms` | Minimum allowed peak-to-trough interval in milliseconds. | `5.0` |
| `--peak2trough-max-ms` | Maximum allowed peak-to-trough interval in milliseconds. | `7.5` |
| `--rms-multiplier` | Baseline RMS multiplier used during onset-candidate refinement. | `1.5` |
| `--smoothing` | Signal smoothing method: `rolling`, `gaussian`, or `none`. | `rolling` |
| `--rolling-smooth-ms` | Rolling smoothing window in milliseconds. | `2.5` |
| `--gaussian-sigma-ms` | Gaussian smoothing sigma in milliseconds. | `1.0` |
| `--refine-chunk-ms` | Initial refinement chunk length in milliseconds. | `2.0` |
| `--tpl-anchor-tol-ms` | Template-anchor tolerance in milliseconds. | `7.5` |

---

## Algorithm overview

For each input file, the script processes each channel and frame as follows.

### 1. Optional resampling

If `--resample-to-hz` is specified, the EMG block is resampled along the sample axis before filtering, template creation, gating, and latency detection.

### 2. Preprocessing

A 50 Hz notch filter is applied by default to reduce mains noise. The signal is then smoothed using either a rolling or Gaussian smoothing operation, depending on the selected configuration.

### 3. Template construction

For each channel, the script builds a normalised average MEP template from frames that pass the initial pre-stimulus and amplitude gates. This template is used to constrain the expected timing of the principal MEP deflection.

### 4. Artefact and amplitude gating

Frames are screened for excessive pre-stimulus activity. Frames are also rejected if the MEP-window peak-to-peak amplitude is not sufficiently greater than the baseline peak-to-peak amplitude.

Rejected frames are retained in the output table but assigned a missing or descriptive value rather than being removed.

### 5. Derivative-ratio scan

For each retained frame, the algorithm identifies the first major MEP deflection and scans backwards through the signal. At each candidate sample, it compares the mean absolute derivative after that point with the mean absolute derivative before that point.

The point with the largest derivative ratio is treated as the primary onset candidate.

### 6. Candidate refinement

The primary candidate and neighbouring samples are tested against additional criteria, including local slope consistency and post-candidate RMS amplitude relative to baseline RMS.

If a candidate passes these checks, the onset latency is recorded in milliseconds relative to the TMS pulse.

### 7. Output

If a valid onset is detected, the latency is saved in milliseconds. If the trial is rejected or no reliable onset is identified, a descriptive missing-value marker is saved instead.

---

## Output format

The script generates one `.csv` file per input `.npy` file.

Output filenames use the input filename stem with the suffix:

```text
_latencies.csv
```

For example:

```text
participant1_rest_map.npy
```

becomes:

```text
participant1_rest_map_latencies.csv
```

Each output file is a wide-format table:

- rows correspond to frames/trials,
- row indices start at 1,
- columns correspond to EMG channels,
- values are detected MEP onset latencies in milliseconds, rounded to 3 decimal places.

Example:

| frame | FDI | APB | ADM | EDC |
|---|---:|---:|---:|---:|
| 1 | 22.500 | 21.000 | NaN | 17.500 |
| 2 | 23.000 | null_onset | 21.500 | 18.000 |

---

## Runtime reporting

From v1.0.1, the script prints runtime information to the console for each processed file.

Example:

```text
Runtime: 2.134 s total | 26.68 ms/frame | 3.34 ms/channel-frame
```

For a typical 8-channel, 80-frame mapping file:

```text
80 frames × 8 channels = 640 channel-frames
```

The `ms/channel-frame` value therefore gives an approximate per-epoch processing time.

When processing multiple files, the script also prints an overall runtime summary.

---

## Parallel processing

The `--parallel` argument controls whether files are processed serially or in parallel.

Serial processing:

```bash
--parallel 0
```

Parallel processing with 4 workers:

```bash
--parallel 4
```

Parallel processing is most useful when analysing several files. For very small batches, serial processing may be easier for debugging and can avoid parallelisation overhead.

---

## Version notes

### v1.0.1

- Added console reporting of computational runtime per file.
- Runtime output includes total runtime, ms/frame, and ms/channel-frame.
- Added an overall runtime summary across processed files.
- Optimised the derivative-ratio detector by replacing per-frame pandas operations with NumPy-based operations.
- Preserved the existing command-line interface and output CSV structure.
- Output latencies are unchanged relative to v1.0.0 for validation checks performed on representative active and resting multichannel files.

### v1.0.0

- Initial public release of the derivative-ratio MEP onset latency detection pipeline.

---

## Citation

If you use this code, please cite the Zenodo record associated with the version used for your analysis.

For reproducible analyses, cite the specific version DOI rather than only the general repository link.

---

## Notes

This pipeline was developed for offline analysis of epoched EMG data from TMS studies. The script is intended to support scalable analysis of MEP onset latency in multichannel datasets, but users should inspect outputs and validate performance for their own recording setup, muscles, sampling rate, filtering, and participant population.
