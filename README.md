# AV-Sync Metric Calibration Toolkit

Measurement scripts, reference-set manifest, and all measured data for the paper
*"Measuring the Ruler: Calibrating Audio-Visual Synchronization Metrics for
Long-Form Generated Video"* (ICASSP 2027 submission).

Audio-visual synchronization metrics (Synchformer-based offset estimators,
event-agreement scores) are routinely applied outside their validated regime.
This toolkit measures, on real synchronized footage with exactly known injected
offsets, each metric's noise floor, sensitivity slope, saturation range, and
container-path artifacts — and packages the five-component calibration protocol
from the paper so any lab can re-derive every number.

## Contents

```
scripts/
  desync_core.py            # Synchformer wrapper: sliding-window DeSync curves
  calibrate_desync.py       # single-video injected-offset calibration
  calibrate_desync_batch.py # batch calibration on generated distributions
  xcorr_desync.py           # cross-correlation auxiliary measurement
  segment_sensitivity.py    # window-length sensitivity
  av_align_curve.py         # AV-Align (original one-to-one matcher) + peak cache
  avalign_offset_calib.py   # AV-Align injected-offset sweep (peak-timestamp shift)
  avalign_crosspair.py      # cross-pairing control (mismatched audio x video)
  avalign_aac_path.py       # AAC container-path artifact measurement
  nscaling_avalign.py       # n-scaling / CI-width analysis
  build_refset.py           # reference-set reconstruction from manifest
  make_table2.py            # Table 2 with moving-block bootstrap (canonical
                            # aggregation; per-video JSON `resolvable` fields
                            # are legacy i.i.d. — do not cite them directly)
data/
  *.json                    # every measured number behind every figure/table
  peaks_cache/              # cached audio/visual peak timestamps (14 refs)
  generated_transfer_sets/  # injected-offset gain on 7 generated sets (Sec. 5;
                            # index.json summarises floor/slope/CI per set)
refset/
  (built locally)           # videos are NOT redistributed; build_refset.py
                            # reconstructs from data/refset_meta.json
                            # (source URLs, timestamps, licenses, attribution)
```

## Environment

Python ≥ 3.10, PyTorch ≥ 2.1, plus: `soundfile librosa opencv-python scipy`.
Synchformer weights auto-download on first run
(`synchformer_state_dict.pth`, MMAudio release). AV-Align scoring uses the
original TempoTokens implementation (`av_align.py`, vendored path in
`av_align_curve.py` — adjust `TT_DIR`).

## Reproducing the paper

| Paper item | Command |
|---|---|
| DeSync floor / slope / saturation (Fig. 1, Table 1) | `calibrate_desync.py --video refNN.mp4` per ref, aggregate |
| Sub-grid slope, out-of-range foldback | same, offsets list incl. ±0.02–±2.5 |
| AAC container path (+64 ms, onset collapse) | `avalign_aac_path.py` |
| AV-Align flat response + chance level (Fig. 2a) | `avalign_offset_calib.py` |
| Cross-pairing control (§3.3) | `avalign_crosspair.py` |
| n-scaling / CI contraction (Fig. 2b) | `nscaling_avalign.py` |
| Table 2 (block-bootstrap) | `make_table2.py` |
| Generated-distribution transfer (§5, `data/generated_transfer_sets/`) | `calibrate_desync_batch.py` per set, offsets ±0.2/±0.5 s |

Every JSON in `data/` is the exact file the paper's figures were rendered from.

## Reference set

14 real, verifiably synchronized clips across five audio classes. We publish
the manifest (`data/refset_meta.json`: source URL, time span, license,
attribution) rather than the media. `build_refset.py --manifest
data/refset_meta.json` downloads and cuts the exact clips.

## License

Code: MIT (see `LICENSE`). Measured data (`data/*.json`): CC BY 4.0.
Reference videos remain under their original licenses (see manifest).
