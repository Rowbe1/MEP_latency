#!/usr/bin/env python3
"""
latency_detect.py – publication-ready MEP-latency pipeline
----------------------------------------------------------
Detect onsets in EMG epochs stored as NumPy *.npy files and save *one wide CSV
per input* (frames × channels, latencies in **ms**).  Key flags:

    --task-mode {rest,active,auto}      # prestim outlier rule
    --rms-multiplier 1.6           # refine-candidate gate (default 2.0)
    --parallel 4                   # workers (0 = serial)

Run inside Spyder with F5 – defaults are injected automatically.
"""
from __future__ import annotations
import argparse, logging, math, sys
from dataclasses import dataclass, replace
import re
from pathlib import Path
from typing import Sequence

import numpy as np, pandas as pd
from scipy.signal import iirnotch, filtfilt
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from joblib import Parallel, delayed

###############################################################################
# 1 | config                                                                  #
###############################################################################

@dataclass(frozen=True)
class Cfg:
    fs: int = 2000                # Hz
    stim_on: float = 0.25         # s
    mep_on: float = 0.005
    mep_off: float = 0.045
    prestim_start: float = -0.101
    prestim_end: float = -0.001

    ptp_factor: float = 1.1
    derivative_block: int = 5
    derivative_ratio_thresh: float = 0.85
    latency_cap: float = 0.035
    peak2trough_min: int = 10     # samples
    peak2trough_max: int = 15
    mains_filter: bool = True
    notch_q: float = 30.0
    smoothing: str | None = "rolling"   # {'gaussian','rolling',None}

    # new knobs --------------------------------------------------------------
    rms_multiplier: float = 1.5         # window RMS > X × baseline RMS
    task_mode: str = "rest"             # {'rest','active','auto'}
    active_token: str = "act"           # token that marks active files
    
    # cached indices ---------------------------------------------------------
    @property
    def idx(self):
        s2i = lambda t: int(round((t + self.stim_on) * self.fs))
        return {
            "stim": int(round(self.stim_on * self.fs)),
            "mep_on": s2i(self.mep_on),
            "mep_off": s2i(self.mep_off),
            "pre_start": s2i(self.prestim_start),
            "pre_end": s2i(self.prestim_end),
        }

###############################################################################
# 2 | helpers                                                                 #
###############################################################################

def notch_50hz(x: np.ndarray, cfg: Cfg) -> np.ndarray:
    if not cfg.mains_filter: return x
    w0 = 50.0 / (cfg.fs / 2); b, a = iirnotch(w0, cfg.notch_q)
    return filtfilt(b, a, x, axis=0)

def smooth(sig: np.ndarray, cfg: Cfg) -> np.ndarray:
    if cfg.smoothing == "gaussian":
        return gaussian_filter1d(sig, 2.0, axis=0)
    if cfg.smoothing == "rolling":
        return uniform_filter1d(uniform_filter1d(sig, 5, axis=0), 5, axis=0)
    return sig

def prestim_mask(baseline: np.ndarray, mode: str) -> np.ndarray:
    """True = keep frame. Use per-frame RMS in the pre-stim window.

    baseline: array shaped (samples_in_prestim, n_frames)
    mode    : 'rest' or 'active' (affects the z-score threshold only)
    """
    # Per-frame RMS across the pre-stimulus window
    rms = np.sqrt((baseline ** 2).mean(axis=0))
    mu, sd = rms.mean(), rms.std(ddof=0)
    if sd == 0 or not np.isfinite(sd):     # degenerate baseline
        return np.ones_like(rms, dtype=bool)

    # Two-sided z-score filter around the session mean RMS
    thr = 3.5 if mode == "active" else 2.0
    z = (rms - mu) / sd
    good = np.abs(z) <= thr                 # keep frames within the band
    return good

def is_active_file(name: str, token: str) -> bool:
    """
    True if `token` appears as a standalone chunk in `name` (case-insensitive),
    e.g. '_act', '-act', 'act_'. Avoids matching 'artifact'.
    """
    s = name.lower()
    pat = rf"(?<![a-z]){re.escape(token.lower())}(?![a-z])"
    return re.search(pat, s) is not None

###############################################################################
# 3 | template builder                                                        #
###############################################################################

def build_templates(emg: np.ndarray, chans: Sequence[str], cfg: Cfg):
    idx, tpl = cfg.idx, {}
    for i, ch in enumerate(chans):
        sig = emg[:, :, i]
        base = sig[idx["pre_start"]:idx["pre_end"]]
        good = prestim_mask(base, cfg.task_mode)
        mep = sig[idx["mep_on"]:idx["mep_off"]]
        good &= np.ptp(mep,  axis=0) > cfg.ptp_factor * np.ptp(base, axis=0)
        if not good.any(): tpl[ch] = None; continue
        waves = mep[:, good]
        waves = (waves - waves.mean(0)) / (waves.std(0) + 1e-9)
        tpl[ch] = waves.mean(1)
    return tpl

###############################################################################
# 4 | candidate refinement                                                    #
###############################################################################

def _refine(df: pd.DataFrame, cand: int, start: int, mean_d: float,
            std_d: float, p2t: int, base_rms: float, cfg: Cfg) -> int | None:
    chunk, win, span = 4, p2t * 2, p2t // 4

    def ok(j: int) -> bool:
        w = df.iloc[j:j+win]
        if w.empty: return False
        cond = w["d"] < 0
        rms = math.sqrt((w["sig"]**2).mean())
        return (cond.mean() > .5 and cond.iloc[:chunk].sum() >= 3 and
                (rms > cfg.rms_multiplier * base_rms or
                 w["diff"].mean() > mean_d + 1.5*std_d))

    if ok(cand): return cand
    lower = max(cand-span, cfg.idx["mep_on"])
    for j in range(cand-1, lower-1, -1):
        if ok(j): return j
    for j in range(cand+1, cand+span+1):
        if j > start: break
        if ok(j): return j
    return None

###############################################################################
# 5 | per-channel pipeline                                                   #
###############################################################################

def process_channel(sig: np.ndarray, tpl: np.ndarray | None, cfg: Cfg) -> np.ndarray:
    idx, nF = cfg.idx, sig.shape[1]
    lat = np.full(nF, np.nan, dtype=object)
    if tpl is None: lat[:] = "NaN"; return lat

    base = sig[idx["pre_start"]:idx["pre_end"]]
    good_frames = prestim_mask(base, cfg.task_mode)
    lat[~good_frames] = "NaN"

    mep = sig[idx["mep_on"]:idx["mep_off"]]
    keep = (good_frames &
            (np.ptp(mep,  axis=0) > cfg.ptp_factor * np.ptp(base, axis=0)))
    lat[~keep & good_frames] = "NaN"

    tpl_anchor = min(np.argmax(tpl), np.argmin(tpl)) + idx["mep_on"]

    for f in np.flatnonzero(keep):
        frame = sig[:, f]
        df = pd.DataFrame({"sig": frame})
        df["smoothed"] = smooth(frame, cfg)
        diff = np.abs(np.diff(df["smoothed"], prepend=np.nan))
        df["diff"] = diff
        pre_d = diff[idx["pre_start"]:idx["pre_end"]]
        mean_d, std_d = pre_d.mean(), pre_d.std()
        df["d"] = mean_d - df["diff"]

        mwin = df.iloc[idx["mep_on"]:idx["mep_off"]]
        peak, trough = mwin["smoothed"].idxmax(), mwin["smoothed"].idxmin()
        p2t = np.clip(abs(peak-trough), cfg.peak2trough_min, cfg.peak2trough_max)
        start_idx = min(peak, trough)

        if not (tpl_anchor-15 <= start_idx <= tpl_anchor+15):
            lat[f] = "null_onset"; continue

        ratios, eps = {}, 1e-6
        lower = max(start_idx - int(p2t*1.75), idx["mep_on"])
        i = start_idx - cfg.derivative_block
        while i >= lower:
            prev = df["diff"].iloc[i-cfg.derivative_block:i].mean()
            nxt  = df["diff"].iloc[i:i+cfg.derivative_block].mean()
            ratios[i] = abs(nxt)/(abs(prev)+eps)
            i -= 1
        if not ratios: lat[f] = "null_onset"; continue

        mx, mxr = max(ratios.items(), key=lambda kv: kv[1])
        thr = cfg.derivative_ratio_thresh * mxr
        cands = [mx]
        j = mx-1
        while ratios.get(j,0)>=thr: cands.append(j); j-=1
        j = mx+1
        while ratios.get(j,0)>=thr: cands.append(j); j+=1

        base_rms = math.sqrt((frame[idx["pre_start"]:idx["pre_end"]]**2).mean())
        refined = None
        for c in cands:
            if (c-idx["stim"])/cfg.fs > cfg.latency_cap: continue
            refined = _refine(df, c, start_idx, mean_d, std_d, p2t, base_rms, cfg)
            if refined is not None: break

        lat[f] = ("null_onset" if refined is None
                  else round((refined-idx["stim"])/cfg.fs*1000, 3))
    return lat

###############################################################################
# 6 | file pipeline                                                           #
###############################################################################

def process_file(path: Path, out_dir: Path, chans: Sequence[str], cfg: Cfg,
                 log: logging.Logger):
    log.info("Processing %s", path.name)
    # Decide task mode per file if requested
    if cfg.task_mode == "auto":
        file_mode = "active" if is_active_file(path.stem, cfg.active_token) else "rest"
    else:
        file_mode = cfg.task_mode
    eff_cfg = replace(
    cfg,
    task_mode=file_mode,
    # Window-RMS gate: stricter at rest, looser during contraction
    rms_multiplier=(2.0 if file_mode == "rest" else 1.5),
    )
    log.debug("Task mode for %s → %s", path.name, file_mode)
    emg = np.load(path)
    emg = notch_50hz(emg, eff_cfg)
    tpls = build_templates(emg, chans, eff_cfg)
    out = {ch: process_channel(emg[:,:,i], tpls[ch], eff_cfg)
           for i, ch in enumerate(chans)}
    df = pd.DataFrame(out)
    # Create and assign a new 1-based index
    df.index = np.arange(1, len(df) + 1) 
    # Save to CSV with the new index
    df.to_csv(out_dir / f"{path.stem}_latencies.csv", index_label="frame")
    log.info("↳ saved %s", f"{path.stem}_latencies.csv")

###############################################################################
# 7 | CLI                                                                     #
###############################################################################

def cli_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MEP latency detector (wide CSV)")
    p.add_argument("--in-dir", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--channels", required=True, type=Path)
    p.add_argument("--parallel", type=int, default=0)
    p.add_argument("--log", default="INFO",
                   choices=["DEBUG","INFO","WARNING","ERROR"])
    p.add_argument("--rms-multiplier", type=float, default=1.5)
    p.add_argument("--task-mode", default="rest", choices=["rest","active","auto"],
                   help="Prestim outlier rule: 'rest', 'active', or 'auto' per file")
    p.add_argument("--active-token", default="act",
                   help="Token that marks active files when --task-mode=auto (default 'act')")
    return p.parse_args()

def main(ns: argparse.Namespace|None=None):
    ns = ns or cli_args()
    logging.basicConfig(level=getattr(logging, ns.log),
                        format="%(levelname)s:%(name)s:%(message)s")
    log = logging.getLogger("latency")
    cfg = Cfg(rms_multiplier=ns.rms_multiplier,
              task_mode=ns.task_mode,
              active_token=ns.active_token)
    chans = np.load(ns.channels, allow_pickle=True).tolist()
    ns.out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(ns.in_dir.glob("*.npy")) or log.error("No .npy found") or sys.exit(1)
    work = delayed(process_file)
    if ns.parallel:
        Parallel(ns.parallel)(work(f, ns.out_dir, chans, cfg, log) for f in files)
    else:
        for f in files: process_file(f, ns.out_dir, chans, cfg, log)

###############################################################################
# 8 | Spyder fallback                                                         #
###############################################################################

if __name__ == "__main__":
    if len(sys.argv) == 1:   # launched via F5 in Spyder
        sys.argv += [
            "--in-dir", "data/control_data/lats",
            "--out-dir", "data/control_data/lats_output_temp",
            "--channels", "data/channels.npy",
            "--parallel", "4",
            "--task-mode", "auto",
            "--active-token", "act", #string to look for in filename denoting active trials
            "--rms-multiplier", "1.5",
        ]
    main()
