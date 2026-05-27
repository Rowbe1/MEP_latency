#!/usr/bin/env python3
"""
advanced_latency_analysis_v11a_light_refactor_v5.py
----------------------------------------------------
Publication-ready MEP-latency pipeline with optional resampling and
sampling-rate-aware timing parameters.

Detect onsets in EMG epochs stored as NumPy *.npy files and save one wide CSV
per input file (frames × channels, latencies in ms).

Key flags
---------
    --fs 2000                         # native input sampling rate
    --resample-to-hz 5000             # optional target sampling rate
    --task-mode {rest,active,auto}    # prestim outlier rule
    --parallel 4                      # workers; 0 = serial

Sensitivity-analysis flags
--------------------------
    --ptp-factor 1.1
    --derivative-block-ms 2.5
    --search-back-factor 1.75
    --derivative-ratio-thresh 0.85
    --peak2trough-min-ms 5.0
    --peak2trough-max-ms 7.5

Important implementation detail
-------------------------------
Parameters that were previously hard-coded as sample counts are now stored in
milliseconds and converted to samples from cfg.fs. Because cfg.fs is replaced
with the effective sampling rate after optional resampling, these parameters
preserve their approximate duration at either native or resampled rates.
"""
from __future__ import annotations

import argparse
import logging
import math
import re
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from scipy.signal import filtfilt, iirnotch, resample_poly

###############################################################################
# 1 | config                                                                  #
###############################################################################

@dataclass(frozen=True)
class Cfg:
    """Configuration for the derivative-ratio latency detector.

    Notes
    -----
    ``fs`` should always represent the sampling rate currently used by the
    algorithm. At file-loading time this is the native sampling rate. If
    ``resample_to_hz`` is set, ``process_file()`` resamples the data and then
    replaces ``cfg.fs`` with the effective sampling rate before filtering,
    template construction, and latency detection.
    """

    # Native/effective sampling and event windows -----------------------------
    fs: int = 2000                  # Hz before resampling; becomes effective Hz afterwards
    stim_on: float = 0.25           # seconds from epoch start to TMS pulse
    mep_on: float = 0.005           # seconds relative to pulse
    mep_off: float = 0.045          # seconds relative to pulse
    prestim_start: float = -0.101   # seconds relative to pulse
    prestim_end: float = -0.001     # seconds relative to pulse
    resample_to_hz: int | None = None  # None = native fs; e.g. 5000 = resample to 5 kHz

    # Amplitude / derivative-ratio parameters ---------------------------------
    ptp_factor: float = 1.1
    derivative_block_ms: float = 2.5
    derivative_ratio_thresh: float = 0.85
    search_back_factor: float = 1.75
    latency_cap: float = 0.035      # seconds relative to pulse

    # Peak-to-trough constraints, now in time units ---------------------------
    peak2trough_min_ms: float = 5.0
    peak2trough_max_ms: float = 7.5

    # Filtering / smoothing ---------------------------------------------------
    mains_filter: bool = True
    notch_q: float = 30.0
    smoothing: str | None = "rolling"   # {'gaussian', 'rolling', None}
    rolling_smooth_ms: float = 2.5
    gaussian_sigma_ms: float = 1.0

    # Refinement / template-anchor parameters, now in time units --------------
    refine_chunk_ms: float = 2.0
    tpl_anchor_tol_ms: float = 7.5

    # Session/gating knobs ----------------------------------------------------
    rms_multiplier: float = 1.5       # window RMS > X × baseline RMS
    task_mode: str = "rest"           # {'rest', 'active', 'auto'}
    active_token: str = "act"         # token that marks active files

    # Cached indices ----------------------------------------------------------
    @property
    def idx(self) -> dict[str, int]:
        """Return key analysis-window indices for the current effective fs."""
        s2i = lambda t: int(round((t + self.stim_on) * self.fs))
        return {
            "stim": int(round(self.stim_on * self.fs)),
            "mep_on": s2i(self.mep_on),
            "mep_off": s2i(self.mep_off),
            "pre_start": s2i(self.prestim_start),
            "pre_end": s2i(self.prestim_end),
        }

    def ms_to_samples(self, ms: float, *, minimum: int = 1) -> int:
        """Convert milliseconds to samples using the current effective fs."""
        return max(minimum, int(math.floor((ms / 1000.0) * self.fs + 0.5)))

    @property
    def derivative_block(self) -> int:
        """Derivative-ratio block length in samples."""
        return self.ms_to_samples(self.derivative_block_ms)

    @property
    def peak2trough_min(self) -> int:
        """Minimum peak-to-trough interval in samples."""
        return self.ms_to_samples(self.peak2trough_min_ms)

    @property
    def peak2trough_max(self) -> int:
        """Maximum peak-to-trough interval in samples."""
        return max(self.peak2trough_min, self.ms_to_samples(self.peak2trough_max_ms))

    @property
    def rolling_smooth_n(self) -> int:
        """Rolling smoothing window in samples."""
        return self.ms_to_samples(self.rolling_smooth_ms)

    @property
    def gaussian_sigma_samples(self) -> float:
        """Gaussian smoothing sigma in samples."""
        return max(1.0, (self.gaussian_sigma_ms / 1000.0) * self.fs)

    @property
    def refine_chunk(self) -> int:
        """Initial refinement chunk length in samples."""
        return self.ms_to_samples(self.refine_chunk_ms)

    @property
    def tpl_anchor_tol(self) -> int:
        """Allowed template-anchor tolerance in samples."""
        return self.ms_to_samples(self.tpl_anchor_tol_ms)

    def derived_sample_summary(self) -> dict[str, int | float]:
        """Return derived sample-count parameters for logging/reproducibility."""
        return {
            "fs": self.fs,
            "derivative_block": self.derivative_block,
            "peak2trough_min": self.peak2trough_min,
            "peak2trough_max": self.peak2trough_max,
            "rolling_smooth_n": self.rolling_smooth_n,
            "gaussian_sigma_samples": round(self.gaussian_sigma_samples, 3),
            "refine_chunk": self.refine_chunk,
            "tpl_anchor_tol": self.tpl_anchor_tol,
        }

###############################################################################
# 2 | helpers                                                                 #
###############################################################################

def notch_50hz(x: np.ndarray, cfg: Cfg) -> np.ndarray:
    """Apply a 50 Hz notch filter along the sample axis if requested."""
    if not cfg.mains_filter:
        return x

    nyquist = cfg.fs / 2.0
    if 50.0 >= nyquist:
        raise ValueError(f"Cannot apply 50 Hz notch when Nyquist is only {nyquist:.1f} Hz")

    w0 = 50.0 / nyquist
    b, a = iirnotch(w0, cfg.notch_q)
    return filtfilt(b, a, x, axis=0)


def resample_emg_block(emg: np.ndarray, cfg: Cfg) -> tuple[np.ndarray, int]:
    """Optionally resample an epoched EMG block along the sample axis.

    Parameters
    ----------
    emg
        Epoched EMG block shaped ``samples × frames × channels``.
    cfg
        Configuration object. ``cfg.fs`` is interpreted as the native sampling
        rate before resampling, and ``cfg.resample_to_hz`` is the optional target.

    Returns
    -------
    tuple[np.ndarray, int]
        The possibly resampled EMG block and the effective sampling rate to use
        for all downstream indexing, filtering, and latency conversion.
    """
    target_fs = cfg.resample_to_hz
    original_fs = int(cfg.fs)

    if target_fs is None or int(target_fs) == original_fs:
        return emg, original_fs

    target_fs = int(target_fs)
    if target_fs <= 0:
        raise ValueError("resample_to_hz must be a positive integer or None")

    common = math.gcd(original_fs, target_fs)
    up = target_fs // common
    down = original_fs // common

    emg_rs = resample_poly(emg, up=up, down=down, axis=0)
    return emg_rs, target_fs


def smooth(sig: np.ndarray, cfg: Cfg) -> np.ndarray:
    """Smooth an EMG trace using windows defined in milliseconds.

    The actual sample width is derived from ``cfg.fs``, so smoothing is
    preserved in time units if the signal is resampled.
    """
    if cfg.smoothing == "gaussian":
        return gaussian_filter1d(sig, cfg.gaussian_sigma_samples, axis=0)

    if cfg.smoothing == "rolling":
        n = cfg.rolling_smooth_n
        return uniform_filter1d(uniform_filter1d(sig, n, axis=0), n, axis=0)

    return sig


def prestim_mask(baseline: np.ndarray, mode: str) -> np.ndarray:
    """Return a Boolean frame mask based on pre-stimulus RMS outliers.

    Parameters
    ----------
    baseline
        Array shaped ``samples_in_prestim × n_frames``.
    mode
        ``'rest'`` or ``'active'``. Active sessions use a looser z-score band.
    """
    if baseline.shape[0] == 0:
        return np.ones(baseline.shape[1], dtype=bool)

    rms = np.sqrt((baseline ** 2).mean(axis=0))
    mu, sd = rms.mean(), rms.std(ddof=0)
    if sd == 0 or not np.isfinite(sd):
        return np.ones_like(rms, dtype=bool)

    thr = 3.5 if mode == "active" else 2.0
    z = (rms - mu) / sd
    return np.abs(z) <= thr


def is_active_file(name: str, token: str) -> bool:
    """Return True if ``token`` appears as a standalone chunk in ``name``."""
    s = name.lower()
    pat = rf"(?<![a-z]){re.escape(token.lower())}(?![a-z])"
    return re.search(pat, s) is not None

###############################################################################
# 3 | template builder                                                        #
###############################################################################

def build_templates(emg: np.ndarray, chans: Sequence[str], cfg: Cfg):
    """Build one normalised average MEP template per channel."""
    idx, tpl = cfg.idx, {}
    for i, ch in enumerate(chans):
        sig = emg[:, :, i]
        base = sig[idx["pre_start"]:idx["pre_end"]]
        mep = sig[idx["mep_on"]:idx["mep_off"]]

        if base.shape[0] == 0 or mep.shape[0] == 0:
            tpl[ch] = None
            continue

        good = prestim_mask(base, cfg.task_mode)
        good &= np.ptp(mep, axis=0) > cfg.ptp_factor * np.ptp(base, axis=0)

        if not good.any():
            tpl[ch] = None
            continue

        waves = mep[:, good]
        waves = (waves - waves.mean(0)) / (waves.std(0) + 1e-9)
        tpl[ch] = waves.mean(1)

    return tpl

###############################################################################
# 4 | candidate refinement                                                    #
###############################################################################

def _refine(sig: np.ndarray, diff: np.ndarray, cand: int, start: int,
            mean_d: float, std_d: float, p2t: int, base_rms: float,
            cfg: Cfg) -> int | None:
    """Refine a candidate onset using local derivative/RMS criteria.

    This is a NumPy equivalent of the earlier pandas/DataFrame implementation.
    It keeps the same decision logic but avoids repeated DataFrame construction
    and ``.iloc`` slicing inside the per-frame loop.
    """
    chunk = cfg.refine_chunk
    win = p2t * 2
    span = max(1, p2t // 4)
    min_neg_samples = max(1, math.ceil(0.75 * chunk))
    mep_on = cfg.idx["mep_on"]

    def ok(j: int) -> bool:
        sig_w = sig[j:j + win]
        diff_w = diff[j:j + win]
        if sig_w.size == 0 or diff_w.size == 0:
            return False

        # Previous code used: d = mean_d - diff; cond = d < 0.
        # This is equivalent to diff > mean_d.
        cond = diff_w > mean_d
        rms = math.sqrt(float(np.mean(sig_w ** 2)))
        first_chunk = cond[:chunk]

        return (
            float(np.mean(cond)) > 0.5
            and int(np.sum(first_chunk)) >= min_neg_samples
            and (
                rms > cfg.rms_multiplier * base_rms
                or float(np.mean(diff_w)) > mean_d + 1.5 * std_d
            )
        )

    if ok(cand):
        return cand

    lower = max(cand - span, mep_on)
    for j in range(cand - 1, lower - 1, -1):
        if ok(j):
            return j

    for j in range(cand + 1, cand + span + 1):
        if j > start:
            break
        if ok(j):
            return j

    return None

###############################################################################
# 5 | per-channel pipeline                                                    #
###############################################################################

def _window_means_from_cumsum(values: np.ndarray, starts: np.ndarray,
                              width: int) -> np.ndarray:
    """Return fixed-width window means for many start indices.

    Parameters
    ----------
    values
        One-dimensional numeric array.
    starts
        Start indices for each window.
    width
        Number of samples in each window.

    Returns
    -------
    np.ndarray
        Mean value for ``values[start:start + width]`` for each start.

    Notes
    -----
    This replaces repeated small pandas/Python slices in the derivative-ratio
    scan. ``diff`` can contain a NaN at index 0 from ``np.diff(..., prepend)``;
    that value is converted to zero before cumulative sums. The derivative-ratio
    search windows are post-stimulus and should not include index 0 in normal
    use, so this conversion preserves practical behaviour while keeping the
    vectorised implementation robust.
    """
    clean = np.nan_to_num(values, nan=0.0)
    cs = np.concatenate(([0.0], np.cumsum(clean, dtype=float)))
    return (cs[starts + width] - cs[starts]) / width


def process_channel(sig: np.ndarray, tpl: np.ndarray | None, cfg: Cfg) -> np.ndarray:
    """Detect onset latencies for all frames in one channel.

    Optimisation notes
    ------------------
    The original implementation created a pandas ``DataFrame`` for every kept
    frame and then used repeated ``.iloc`` slices during the derivative-ratio
    search and refinement checks. This version keeps the same algorithmic steps
    but performs the inner-loop operations with NumPy arrays. It also smooths all
    kept frames for a channel in one vectorised call, rather than smoothing each
    frame separately.
    """
    idx, nF = cfg.idx, sig.shape[1]
    lat = np.full(nF, np.nan, dtype=object)

    if tpl is None:
        lat[:] = "NaN"
        return lat

    base = sig[idx["pre_start"]:idx["pre_end"]]
    mep = sig[idx["mep_on"]:idx["mep_off"]]

    if base.shape[0] == 0 or mep.shape[0] == 0:
        lat[:] = "NaN"
        return lat

    good_frames = prestim_mask(base, cfg.task_mode)
    lat[~good_frames] = "NaN"

    keep = good_frames & (np.ptp(mep, axis=0) > cfg.ptp_factor * np.ptp(base, axis=0))
    lat[~keep & good_frames] = "NaN"

    keep_idx = np.flatnonzero(keep)
    if keep_idx.size == 0:
        return lat

    tpl_anchor = min(np.argmax(tpl), np.argmin(tpl)) + idx["mep_on"]
    db = cfg.derivative_block
    eps = 1e-6

    # Smooth all kept frames at once. This is mathematically equivalent to
    # smoothing each frame independently because smoothing is along axis 0 only.
    sig_keep = sig[:, keep_idx]
    smoothed_keep = smooth(sig_keep, cfg)

    for local_col, f in enumerate(keep_idx):
        frame = sig_keep[:, local_col]
        smoothed = smoothed_keep[:, local_col]
        diff = np.abs(np.diff(smoothed, prepend=np.nan))

        pre_d = diff[idx["pre_start"]:idx["pre_end"]]
        mean_d = float(np.nanmean(pre_d))
        std_d = float(np.nanstd(pre_d))

        mwin = smoothed[idx["mep_on"]:idx["mep_off"]]
        if mwin.size == 0 or np.all(np.isnan(mwin)):
            lat[f] = "null_onset"
            continue

        peak = int(np.nanargmax(mwin) + idx["mep_on"])
        trough = int(np.nanargmin(mwin) + idx["mep_on"])
        p2t = int(np.clip(abs(peak - trough), cfg.peak2trough_min, cfg.peak2trough_max))
        start_idx = min(peak, trough)

        if not (tpl_anchor - cfg.tpl_anchor_tol <= start_idx <= tpl_anchor + cfg.tpl_anchor_tol):
            lat[f] = "null_onset"
            continue

        lower = max(start_idx - int(p2t * cfg.search_back_factor), idx["mep_on"])
        first_i = start_idx - db
        if first_i < lower:
            lat[f] = "null_onset"
            continue

        # Preserve the original scan order: start_idx - db, then step backward.
        i_vals = np.arange(first_i, lower - 1, -1, dtype=int)
        prev_starts = i_vals - db
        nxt_starts = i_vals

        # Defensive bounds check. In normal use the lower bound prevents this.
        valid = (prev_starts >= 0) & (nxt_starts + db <= diff.size)
        if not np.any(valid):
            lat[f] = "null_onset"
            continue

        i_vals = i_vals[valid]
        prev_starts = prev_starts[valid]
        nxt_starts = nxt_starts[valid]

        prev = _window_means_from_cumsum(diff, prev_starts, db)
        nxt = _window_means_from_cumsum(diff, nxt_starts, db)
        ratios = np.abs(nxt) / (np.abs(prev) + eps)

        if ratios.size == 0:
            lat[f] = "null_onset"
            continue

        max_pos = int(np.argmax(ratios))
        mx = int(i_vals[max_pos])
        mxr = float(ratios[max_pos])
        thr = cfg.derivative_ratio_thresh * mxr

        # Recreate the previous neighbour expansion around mx using a dictionary
        # for exact key lookup. This avoids subtle changes in candidate ordering.
        ratio_lookup = dict(zip(i_vals.tolist(), ratios.tolist()))
        cands = [mx]

        j = mx - 1
        while ratio_lookup.get(j, 0) >= thr:
            cands.append(j)
            j -= 1

        j = mx + 1
        while ratio_lookup.get(j, 0) >= thr:
            cands.append(j)
            j += 1

        base_rms = math.sqrt(float(np.mean(frame[idx["pre_start"]:idx["pre_end"]] ** 2)))
        refined = None

        for c in cands:
            if (c - idx["stim"]) / cfg.fs > cfg.latency_cap:
                continue
            refined = _refine(frame, diff, c, start_idx, mean_d, std_d, p2t, base_rms, cfg)
            if refined is not None:
                break

        lat[f] = (
            "null_onset" if refined is None
            else round((refined - idx["stim"]) / cfg.fs * 1000, 3)
        )

    return lat

###############################################################################
# 6 | file pipeline                                                           #
###############################################################################

def process_file(path: Path, out_dir: Path, chans: Sequence[str], cfg: Cfg,
                 log: logging.Logger):
    """Process one ``.npy`` EMG block, write a wide latency CSV, and report runtime.

    Runtime is measured per input file and includes loading, optional resampling,
    filtering, template construction, latency detection, DataFrame construction,
    and CSV export. The printed per-frame value treats each stimulation frame as
    one trial containing all recorded channels. The per-channel-frame value gives
    a rough per-epoch speed, where one epoch is one frame from one channel.
    """
    file_timer_start = time.perf_counter()
    log.info("Processing %s", path.name)

    emg = np.load(path)
    if emg.ndim == 2:
        emg = emg[:, :, None]

    emg, effective_fs = resample_emg_block(emg, cfg)
    if effective_fs != cfg.fs:
        log.info("↳ resampled %s from %d Hz to %d Hz", path.name, cfg.fs, effective_fs)

    if cfg.task_mode == "auto":
        file_mode = "active" if is_active_file(path.stem, cfg.active_token) else "rest"
    else:
        file_mode = cfg.task_mode

    eff_cfg = replace(
        cfg,
        fs=effective_fs,
        task_mode=file_mode,
        # Preserve the existing behaviour: stricter RMS gate at rest, looser during contraction.
        rms_multiplier=(2.0 if file_mode == "rest" else 1.5),
    )

    log.debug("Task mode for %s → %s", path.name, file_mode)
    log.info("↳ effective config: %s", eff_cfg.derived_sample_summary())

    emg = notch_50hz(emg, eff_cfg)
    tpls = build_templates(emg, chans, eff_cfg)

    out = {
        ch: process_channel(emg[:, :, i], tpls[ch], eff_cfg)
        for i, ch in enumerate(chans)
    }

    df = pd.DataFrame(out)
    df.index = np.arange(1, len(df) + 1)
    out_csv = out_dir / f"{path.stem}_latencies.csv"
    df.to_csv(out_csv, index_label="frame")
    log.info("↳ saved %s", out_csv.name)

    elapsed_s = time.perf_counter() - file_timer_start
    n_frames = int(df.shape[0])
    n_channels = int(df.shape[1])
    n_channel_frames = n_frames * n_channels

    sec_per_frame = elapsed_s / n_frames if n_frames else float("nan")
    sec_per_channel_frame = (
        elapsed_s / n_channel_frames if n_channel_frames else float("nan")
    )

    runtime_msg = (
        f"↳ runtime {path.name}: {elapsed_s:.3f} s total | "
        f"{sec_per_frame * 1000:.2f} ms/frame | "
        f"{sec_per_channel_frame * 1000:.2f} ms/channel-frame "
        f"({n_frames} frames × {n_channels} channels)"
    )
    log.info(runtime_msg)

    return {
        "file": path.name,
        "seconds": elapsed_s,
        "frames": n_frames,
        "channels": n_channels,
        "channel_frames": n_channel_frames,
        "seconds_per_frame": sec_per_frame,
        "seconds_per_channel_frame": sec_per_channel_frame,
    }

###############################################################################
# 7 | CLI                                                                     #
###############################################################################

def cli_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="MEP latency detector (wide CSV)")

    p.add_argument("--in-dir", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--channels", required=True, type=Path)

    p.add_argument("--fs", type=int, default=Cfg.fs,
                   help=f"Native input sampling rate in Hz before optional resampling (default {Cfg.fs}).")
    p.add_argument("--resample-to-hz", type=int, default=Cfg.resample_to_hz,
                   help="Optional target sampling rate in Hz before latency detection, e.g. 5000. "
                        "Omit to analyse at native --fs.")

    p.add_argument("--parallel", type=int, default=0)
    p.add_argument("--log", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    p.add_argument("--task-mode", default=Cfg.task_mode, choices=["rest", "active", "auto"],
                   help="Prestim outlier rule: 'rest', 'active', or 'auto' per file")
    p.add_argument("--active-token", default=Cfg.active_token,
                   help="Token that marks active files when --task-mode=auto")

    p.add_argument("--ptp-factor", type=float, default=Cfg.ptp_factor,
                   help="Amplitude gate: MEP ptp must exceed this × baseline ptp.")
    p.add_argument("--derivative-block-ms", type=float, default=Cfg.derivative_block_ms,
                   help="Derivative-ratio block length in milliseconds.")
    p.add_argument("--derivative-ratio-thresh", type=float, default=Cfg.derivative_ratio_thresh,
                   help="Candidate plateau threshold as a fraction of Rmax.")
    p.add_argument("--search-back-factor", type=float, default=Cfg.search_back_factor,
                   help="Search-back limit as a multiple of peak-to-trough distance.")
    p.add_argument("--peak2trough-min-ms", type=float, default=Cfg.peak2trough_min_ms,
                   help="Minimum peak-to-trough interval in milliseconds.")
    p.add_argument("--peak2trough-max-ms", type=float, default=Cfg.peak2trough_max_ms,
                   help="Maximum peak-to-trough interval in milliseconds.")

    p.add_argument("--smoothing", default=Cfg.smoothing, choices=["rolling", "gaussian", "none"],
                   help="Smoothing mode. Use 'none' to disable smoothing.")
    p.add_argument("--rolling-smooth-ms", type=float, default=Cfg.rolling_smooth_ms,
                   help="Rolling smoothing window in milliseconds.")
    p.add_argument("--gaussian-sigma-ms", type=float, default=Cfg.gaussian_sigma_ms,
                   help="Gaussian smoothing sigma in milliseconds.")
    p.add_argument("--refine-chunk-ms", type=float, default=Cfg.refine_chunk_ms,
                   help="Initial refinement chunk length in milliseconds.")
    p.add_argument("--tpl-anchor-tol-ms", type=float, default=Cfg.tpl_anchor_tol_ms,
                   help="Allowed template-anchor tolerance in milliseconds.")
    p.add_argument("--rms-multiplier", type=float, default=Cfg.rms_multiplier,
                   help="Base RMS multiplier. Existing rest/active replacement is preserved in process_file().")

    return p.parse_args()


def main(ns: argparse.Namespace | None = None):
    """Run batch latency detection."""
    ns = ns or cli_args()
    logging.basicConfig(level=getattr(logging, ns.log),
                        format="%(levelname)s:%(name)s:%(message)s")
    log = logging.getLogger("latency")

    smoothing = None if ns.smoothing == "none" else ns.smoothing

    cfg = Cfg(
        fs=ns.fs,
        resample_to_hz=ns.resample_to_hz,
        task_mode=ns.task_mode,
        active_token=ns.active_token,
        rms_multiplier=ns.rms_multiplier,
        ptp_factor=ns.ptp_factor,
        derivative_block_ms=ns.derivative_block_ms,
        derivative_ratio_thresh=ns.derivative_ratio_thresh,
        search_back_factor=ns.search_back_factor,
        peak2trough_min_ms=ns.peak2trough_min_ms,
        peak2trough_max_ms=ns.peak2trough_max_ms,
        smoothing=smoothing,
        rolling_smooth_ms=ns.rolling_smooth_ms,
        gaussian_sigma_ms=ns.gaussian_sigma_ms,
        refine_chunk_ms=ns.refine_chunk_ms,
        tpl_anchor_tol_ms=ns.tpl_anchor_tol_ms,
    )

    chans = np.load(ns.channels, allow_pickle=True).tolist()
    ns.out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(ns.in_dir.glob("*.npy"))
    if not files:
        log.error("No .npy found in %s", ns.in_dir)
        sys.exit(1)

    def _safe_process(f: Path):
        try:
            return process_file(f, ns.out_dir, chans, cfg, log)
        except Exception as e:
            log.error("FAILED %s: %s", f.name, e, exc_info=True)
            (ns.out_dir / f"{f.stem}_latencies.csv").write_text("frame\n")
            return None

    if ns.parallel:
        runtime_results = Parallel(n_jobs=ns.parallel)(
            delayed(_safe_process)(f) for f in files
        )
    else:
        runtime_results = []
        for f in files:
            runtime_results.append(_safe_process(f))

    runtime_results = [r for r in runtime_results if isinstance(r, dict)]
    if runtime_results:
        total_seconds = sum(r["seconds"] for r in runtime_results)
        total_frames = sum(r["frames"] for r in runtime_results)
        total_channel_frames = sum(r["channel_frames"] for r in runtime_results)

        log.info(
            "Runtime summary: %.3f s total | %.2f ms/frame | %.2f ms/channel-frame "
            "(%d files, %d frames, %d channel-frames)",
            total_seconds,
            (total_seconds / total_frames * 1000) if total_frames else float("nan"),
            (total_seconds / total_channel_frames * 1000) if total_channel_frames else float("nan"),
            len(runtime_results),
            total_frames,
            total_channel_frames,
        )

###############################################################################
# 8 | Spyder fallback                                                         #
###############################################################################

if __name__ == "__main__":
    if len(sys.argv) == 1:   # launched via F5 in Spyder
        sys.argv += [
            "--in-dir", r"C:\Users\rowbi\OneDrive - Imperial College London\BRC PhD Fellowship\Code\Neuromap\data\control_data\Practice",
            "--out-dir", r"C:\Users\rowbi\OneDrive - Imperial College London\BRC PhD Fellowship\Code\Neuromap\data\control_data\Practice",
            "--channels", "data/channels.npy",
            "--fs", "2000",                 # native input sampling rate
          # "--resample-to-hz", "5000",     # remove these two entries to analyse at native --fs
            "--parallel", "4",
            "--task-mode", "auto",
            "--active-token", "act",        # string to look for in filename denoting active trials
            "--rms-multiplier", "1.5",
        ]
    main()
