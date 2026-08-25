"""AV-Align(t)：音频能量峰 vs 视频运动峰的 IoU 随时间曲线。

基于 TempoTokens `av_align.py` 的定义（光流峰 vs librosa onset，±1/fps 容差），
改成分箱的时间曲线，并附带 n-scaling 稳定性检验。

两条约束（Phase 0 标定结论）：
  1. 音频从无损 wav 读（AAC 的 +64ms priming 与 ±1/fps=62.5ms 容差同量级）。
  2. 该指标在短片段上极不稳定（4s 上非单调、方差极大，见 EXPERIMENT_LOG 关键结论 5）；
     `--check-scaling` 会给出各分箱长度下的自助法置信带宽度，用于确认所选 bin 够稳。

用法：
  python eval/drift/av_align_curve.py --video x.mp4 [--audio x.wav] --bin 2.0 --out x.json
"""
import argparse, json, sys
from pathlib import Path

import numpy as np

TT_DIR = Path(__file__).resolve().parents[1] / "third_party" / "TempoTokens"
sys.path.insert(0, str(TT_DIR))

import cv2                                                        # noqa: E402
import librosa                                                    # noqa: E402
import soundfile as sf                                            # noqa: E402
from av_align import compute_of, find_local_max_indexes, calc_intersection_over_union  # noqa: E402


def read_frames(video):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f"打不开视频 {video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return frames, fps


def audio_peaks(wav, sr):
    env = librosa.onset.onset_strength(y=wav, sr=sr)
    fr = librosa.onset.onset_detect(onset_envelope=env, sr=sr)
    return librosa.frames_to_time(fr, sr=sr)


def video_peaks(frames, fps):
    flow = [compute_of(frames[0], frames[1])] + \
           [compute_of(frames[i - 1], frames[i]) for i in range(1, len(frames))]
    return find_local_max_indexes(flow, fps), flow


def iou_in_window(apk, vpk, fps, t0, t1):
    a = [p for p in apk if t0 <= p < t1]
    v = [p for p in vpk if t0 <= p < t1]
    if len(a) + len(v) == 0:
        return float("nan"), 0, 0
    return calc_intersection_over_union(a, v, fps), len(a), len(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--audio", default=None)
    ap.add_argument("--bin", type=float, default=2.0, help="分箱长度（秒）")
    ap.add_argument("--out", default=None)
    ap.add_argument("--check-scaling", action="store_true",
                    help="额外报告 bin ∈ {2,4,8,15} 的自助法置信带宽度")
    args = ap.parse_args()

    video = str(Path(args.video).resolve())
    wav_path = str(Path(args.audio).resolve()) if args.audio else str(Path(video).with_suffix(".wav"))
    if not Path(wav_path).exists():
        raise FileNotFoundError(
            f"缺少无损音轨 {wav_path}。AV-Align 的匹配容差是 ±1/fps（16fps → ±62.5ms），"
            f"与 mp4 AAC 轨的 +64ms priming 同量级，必须从 wav 读。")

    frames, fps = read_frames(video)
    wav, sr = sf.read(wav_path)
    wav = wav if wav.ndim == 1 else wav.mean(1)
    dur = min(len(frames) / fps, len(wav) / sr)
    apk, (vpk, flow) = audio_peaks(wav.astype(np.float32), sr), video_peaks(frames, fps)
    print(f"[{Path(video).name}] dur={dur:.2f}s fps={fps:.2f}  音频峰={len(apk)} 视频峰={len(vpk)}")

    edges = np.arange(0.0, dur + 1e-9, args.bin)
    t, vals, na, nv = [], [], [], []
    for i in range(len(edges) - 1):
        v, a_n, v_n = iou_in_window(apk, vpk, fps, edges[i], edges[i + 1])
        t.append(float((edges[i] + edges[i + 1]) / 2)); vals.append(v)
        na.append(a_n); nv.append(v_n)

    arr = np.asarray(vals, dtype=float)
    ok = ~np.isnan(arr)
    n = int(ok.sum())
    k = max(1, int(round(0.2 * n)))
    valid = arr[ok]; tv = np.asarray(t)[ok]
    d_end = float(valid[-k:].mean() - valid[:k].mean()) if n >= 2 else float("nan")
    try:
        from scipy.stats import theilslopes
        slope = float(theilslopes(valid, tv)[0]) if n >= 3 else float("nan")
    except Exception:
        slope = float(np.polyfit(tv, valid, 1)[0]) if n >= 2 else float("nan")

    out = dict(video=video, audio=wav_path, fps=float(fps), dur=float(dur), bin=args.bin,
               n_audio_peaks=len(apk), n_video_peaks=len(vpk),
               t=t, av_align=vals, n_audio_in_bin=na, n_video_in_bin=nv,
               scalars=dict(mean=float(np.nanmean(arr)), D_end=d_end, slope=slope,
                            n_valid_bins=n))

    if args.check_scaling:
        rng = np.random.default_rng(0)
        scal = {}
        for b in [2.0, 4.0, 8.0, 15.0]:
            e2 = np.arange(0.0, dur + 1e-9, b)
            vv = [iou_in_window(apk, vpk, fps, e2[i], e2[i + 1])[0] for i in range(len(e2) - 1)]
            vv = np.asarray([x for x in vv if not np.isnan(x)])
            if len(vv) < 2:
                continue
            boot = [rng.choice(vv, len(vv)).mean() for _ in range(2000)]
            lo, hi = np.percentile(boot, [2.5, 97.5])
            scal[f"bin{b:g}"] = dict(n_bins=len(vv), mean=float(vv.mean()),
                                     ci_lo=float(lo), ci_hi=float(hi), ci_width=float(hi - lo))
            print(f"  bin={b:4.1f}s  n={len(vv):3d}  均值={vv.mean():.3f}  "
                  f"95%CI 宽度={hi-lo:.3f}")
        out["scaling_check"] = scal

    dst = args.out or str(Path(video).with_name(Path(video).stem + "_avalign.json"))
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  AV-Align 均值={np.nanmean(arr):.4f}  D_end={d_end:+.4f}  slope={slope:+.5f}")
    print(f"  -> {dst}")


if __name__ == "__main__":
    main()
