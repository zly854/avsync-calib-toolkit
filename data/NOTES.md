# Data index

Every file here is the exact measurement the paper's figures and tables were rendered from.

- `avalign_offset_calib_v2.json`: AV-Align under peak-timestamp shifts, 15 offsets, two clip lengths (Fig. 4a).
- `null_model_mc.json`: exact Monte Carlo chance model (one-to-one matcher, B = 200): pooled chance 0.211,
  r = 0.935 against redrawn peaks, kappa = -0.03 +/- 0.02, 14/14 videos at or below their own chance level (Table 1).
- `avalign_crosspair.json`: cross-pairing control, 182 mismatched audio x video pairs (Table 1).
- `avalign_aac_path.json`: AAC container path, 0.188 -> 0.105; onset collapse on ref05/06/12/13 (Fig. 3, Table 1).
- `peaks_cache/ref*_peaks.json` (14): optical-flow motion peaks and audio onset peaks; the basis of every offline re-computation.
- `ref*_desync_calib.json` (14): expectation and argmax readouts per window under 7 injected offsets.
- `ref*_desync_calib_ext.json` (14): extended sweep, 17 offsets including sub-grid and beyond-range (Fig. 2, Table 4).
- `calib_realref_aggregate.json`: pooled floor / slope / saturation over the reference set (Table 2).
- `clicktrain_xcorr.json`: click-train cross-correlation lag through the container and lossless paths (Fig. 3a).
- `ref05_envelope_case.json`: onset-envelope traces behind Fig. 3b.
- `nscaling_realref.json`: per-window SD of AV-Align vs clip length, 0.072 / 0.056 / 0.043 / 0.034 at 4 / 8 / 15 / 30 s (Fig. 4b).
- `generated_transfer_sets/`: injected-offset gain on generated sets A-G plus the collapsed set H (Table 3); `index.json`
  summarises floor, slope and CI per set.
- `refset_meta.json`: reference-set manifest (source URL, time span, license, attribution); `scripts/build_refset.py` rebuilds the clips.
- `analyze_avalign_calib.py`: helper that aggregates the AV-Align calibration JSONs above.

Matcher note: all IoU re-computations must use the one-to-one matcher of the original AV-Align implementation;
the many-to-one copy shipped with a downstream benchmark reads about 50% higher on identical peak sets (Sec. 3.3).
