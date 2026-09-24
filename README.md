# AV-Sync Metric Calibration Toolkit

Scripts and measured data for **Measuring the Ruler: Calibrating Audio-Visual
Synchronization Metrics for Long-Form Generated Video** (ICASSP 2027 submission).

The reference set contains 14 screened, naturally synchronized, one-minute
Wikimedia Commons clips: four speech, four music, four impact, two ambient.
Injected offsets are known exactly; residual source synchronization errors were
not independently measured. Reported floors are specific to this reference set,
checkpoint, readout, and processing path. The measured +64 ms AAC artifact is
not a universal property of all AAC decoders.

## Reproduce from released measurements (no GPU or media downloads)

Run from the repository root, with Python 3.10 or newer:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-offline.txt
python -m unittest discover -s tests -v
python scripts/make_table2.py
python scripts/avalign_crosspair.py --refdir data --out output/crosspair.json
python scripts/avalign_offset_calib.py --refdir data --cached-only --out output/offsets.json
python scripts/render_paper_figures.py --out output/figures
```

Expected block-bootstrap counts (advanced / delayed audio), at absolute offsets
0.05, 0.10, 0.20, 0.50, 1.00, 1.50, 2.50, 3.00 seconds:

```
advanced: [3, 3, 6, 9, 12, 12, 8, 5]
delayed:  [5, 5, 5, 7, 12, 12, 6, 3]
```

Cross-pairing gives matched mean 0.1902 (14 clips), mismatched mean 0.1907
(182 ordered pairs), random-peak mean 0.2114 (200 draws/clip, seed 0), and static
control 0.0000. Pairwise tests in the historical output are descriptive: pairs
reuse clips and are not independent experimental units.

`make_table2.py` is the historical filename; it now reproduces **Table 3** in the
revised manuscript. It uses circular moving-block bootstrap (block length 8,
2,000 draws, seed 0). Stored per-video `resolvable` fields and `--iid` are legacy
iid-window results and must not substitute for the block-bootstrap counts.

| Revised paper item | Released data / reproduction |
|---|---|
| Fig. 1, calibration checks | `render_paper_figures.py` schematic |
| Fig. 2, extraction artifacts | `data/clicktrain_xcorr.json`, `data/ref05_envelope_case.json` |
| Fig. 3, offset readout | `data/ref*_desync_calib_ext.json` |
| Fig. 4, event agreement and duration | `data/avalign_offset_calib_v2.json`, `data/null_model_mc.json`, `data/nscaling_realref.json` |
| Table 1, event controls | cached offset and cross-pairing commands above; AAC/generated controls in `data/` |
| Table 2, calibration summary | measurements summarized across the listed controls |
| Table 3, separability | `make_table2.py` |
| Table 4, generated transfer | `data/generated_transfer_sets/index.json`, per-set JSONs |

## Reconstruct reference media

Requires `curl` and `ffmpeg` on PATH. Media are not redistributed. The manifest
records source URLs, titles, authors, licenses and extraction spans. Its source
licenses still apply; inspect attribution before using or redistributing media.

```sh
python scripts/build_refset.py --manifest data/refset_meta.json --out refset --dry-run
python scripts/build_refset.py --manifest data/refset_meta.json --out refset
```

This uses the fixed manifest, not live search results. It builds silent 25 fps
H.264 video and a separate 16 kHz mono PCM WAV. Start/duration arguments retain
the original two-decimal ffmpeg rounding. Both outputs must exist before a clip
is skipped; the complete manifest is preserved on reruns. Upstream files may
change or disappear. Historical source hashes and original dependency versions
were not recorded, so a byte-identical historical rebuild cannot be guaranteed.
The released measurement caches remain the exact inputs used for the paper.

## Fresh model inference and peak extraction

Fresh DeSync inference requires a compatible CUDA/Linux environment and the
[official Synchformer environment](https://github.com/v-iashin/Synchformer#environment-preparation).
The checkpoint is **24-01-04T16-39-21**, not the MMAudio feature-extractor weights.
There is no automatic model download in this toolkit.

```sh
mkdir -p third_party
git clone https://github.com/v-iashin/Synchformer.git third_party/Synchformer
git -C third_party/Synchformer checkout b66668a1521d7567cc760e5544b2b5b53179b687
```

Install that checkout's `conda_env.yml` and activate `synchformer`. Download the
[config](https://a3s.fi/swift/v1/AUTH_a235c0f452d648828f745589cde1219a/sync/sync_models/24-01-04T16-39-21/cfg-24-01-04T16-39-21.yaml)
and [checkpoint](https://a3s.fi/swift/v1/AUTH_a235c0f452d648828f745589cde1219a/sync/sync_models/24-01-04T16-39-21/24-01-04T16-39-21.pt)
into `third_party/Synchformer/logs/sync_models/24-01-04T16-39-21/`, preserving
filenames. `SYNCHFORMER_DIR` can override the checkout location. Use only trusted
checkpoint files, as the upstream checkpoint is loaded as a PyTorch object.

```sh
python scripts/calibrate_desync.py --video refset/ref00.mp4 --audio refset/ref00.wav --device cuda:0 --offsets -3 -2.5 -1.5 -1 -.5 -.2 -.1 -.05 0 .05 .1 .2 .5 1 1.5 2.5 3 --out output/ref00_desync_calib_ext.json
python scripts/calibrate_desync_batch.py --dir /path/to/generated_mp4_and_wav --device cuda:0
```

Repeat the first command for ref00 through ref13, then point `make_table2.py
--data-dir output` at those files. Positive injected offsets advance audio;
negative offsets delay it. A global slope fitted across the full extended sweep
includes foldback and is not the in-range sensitivity slope.

For new AV-Align peaks, install `opencv-python librosa soundfile` and clone the
[official TempoTokens implementation](https://github.com/guyyariv/TempoTokens):

```sh
git clone https://github.com/guyyariv/TempoTokens.git third_party/TempoTokens
git -C third_party/TempoTokens checkout 41e024c91266ffc32d7fca05209c5d992c1b95d9
python scripts/avalign_offset_calib.py --refdir refset --out output/new_offsets.json --workers 4
python scripts/avalign_aac_path.py --refdir refset --out output/aac_path.json
python scripts/nscaling_avalign.py --videos 'refset/ref*.mp4' --out output/nscaling.json
```

Cached controls use a local implementation of TempoTokens' ordered one-to-one
matcher with strict one-frame bounds. Input order is preserved, including for
random audio peaks. Optical flow and visual-peak extraction use upstream code.
The pinned revisions above were inspected for this release; the historical
experiment did not record upstream commit hashes. Fresh GPU inference and full
media extraction were not rerun in the submission packaging environment.

## Data and limitations

`data/generated_transfer_sets/` provides measurements for A-G (24 outputs each)
and collapsed-content control H (8 outputs). These are variants of one system,
not an independent multi-model benchmark. Generated media and the generator's
training/sampling pipeline are not included. The release reproduces cached
analyses; it does not claim to regenerate all source material from scratch.

## License and acknowledgment

Code: MIT, see [LICENSE](LICENSE). Released measurement JSONs: CC BY 4.0, see
[data/LICENSE.md](data/LICENSE.md). Source media and external dependencies retain
their own licenses. Source author/license metadata does not transfer ownership.

Supported by the Shenzhen Science and Technology Project under Grant
KJZD20240903103210014. The authors declare no conflicts of interest.
