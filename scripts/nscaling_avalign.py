"""AV-Align n-scaling 检验（Paper B 核心证据；EXPERIMENT_LOG 关键结论 5 的 Phase 1.4 欠账）。

问题：AV-Align 在多短的片段上不可信？多长才稳？
方法：对每条长视频**全片提峰一次**（音频 onset + 光流局部极大），再按片段时长
d ∈ {4,8,15,30,60}s 以 stride=d/2 平铺子窗计算 IoU，报告逐时长的读数分布：
  - 窗间标准差 σ(d)（同一视频内子窗读数的离散度——指标自身噪声）
  - 每视频均值的自助法 95% CI 宽度（"报一个数"时的不确定度）
  - 每窗峰数分布（噪声的机理解释：分母小 → 方差大）

注意（写进 B 的方法节）：本模式（mode=window）把提峰固定在全片上，隔离了
"窗口统计量本身的 n-scaling"；独立短片提峰还会额外引入 librosa 分帧栅格与
光流首帧效应（关键结论 5 的非单调现象），该分量在 refset 上另测（mode=clip，
逐片段独立提峰，慢一个量级，仅在 RealRef 上跑）。

用法：
  python scripts/nscaling_avalign.py --videos "results/<generated_set>/sample_*.mp4" \
      --durations 4 8 15 30 60 --out results/<generated_set>/aggregate/nscaling.json
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from av_align_curve import read_frames, audio_peaks, video_peaks, iou_in_window  # noqa: E402
import soundfile as sf  # noqa: E402


def one_video(video, durations):
    video = Path(video)
    wav_path = video.with_suffix(".wav")
    if not wav_path.exists():
        raise FileNotFoundError(f"缺 {wav_path}（wav 测量通路强制，关键结论 1）")
    wav, sr = sf.read(str(wav_path), dtype="float64")
    if wav.ndim > 1:
        wav = wav.mean(1)
    frames, fps = read_frames(video)
    dur = min(len(frames) / fps, len(wav) / sr)
    apk = audio_peaks(wav.astype(np.float32), sr)
    vpk, _ = video_peaks(frames, fps)

    out = {}
    for d in durations:
        vals, na_l, nv_l = [], [], []
        stride = d / 2
        t0 = 0.0
        while t0 + d <= dur + 1e-6:
            iou, na, nv = iou_in_window(apk, vpk, fps, t0, t0 + d)
            if np.isfinite(iou):
                vals.append(iou); na_l.append(na); nv_l.append(nv)
            t0 += stride
        out[str(d)] = dict(vals=vals, n_audio=na_l, n_video=nv_l)
    return out, dur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True, help="glob 模式")
    ap.add_argument("--durations", type=float, nargs="+", default=[4, 8, 15, 30, 60])
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    files = sorted(glob.glob(args.videos))
    assert files, f"no match: {args.videos}"
    per_video = {}
    for f in files:
        try:
            res, dur = one_video(f, args.durations)
            per_video[Path(f).stem] = res
            print(f"[{Path(f).stem}] dur={dur:.1f}s  " + "  ".join(
                f"d={d:g}:n={len(res[str(d)]['vals'])}" for d in args.durations))
        except Exception as e:
            print(f"[{Path(f).stem}] FAIL {e}")

    rng = np.random.default_rng(0)
    summary = {}
    for d in args.durations:
        key = str(d)
        all_vals = np.concatenate([np.asarray(v[key]["vals"]) for v in per_video.values()
                                   if v[key]["vals"]]) if per_video else np.array([])
        vid_means = np.asarray([np.mean(v[key]["vals"]) for v in per_video.values()
                                if v[key]["vals"]])
        within_sd = np.asarray([np.std(v[key]["vals"], ddof=1)
                                for v in per_video.values() if len(v[key]["vals"]) > 1])
        # 自助法：整体均值的 95% CI 宽度（对"报一个数"的不确定度）
        if len(vid_means) > 1:
            idx = rng.integers(0, len(vid_means), size=(args.n_boot, len(vid_means)))
            boots = vid_means[idx].mean(1)
            ci_w = float(np.percentile(boots, 97.5) - np.percentile(boots, 2.5))
        else:
            ci_w = None
        na = np.concatenate([np.asarray(v[key]["n_audio"]) for v in per_video.values()
                             if v[key]["n_audio"]])
        nv = np.concatenate([np.asarray(v[key]["n_video"]) for v in per_video.values()
                             if v[key]["n_video"]])
        summary[key] = dict(
            n_windows=int(len(all_vals)),
            mean=float(all_vals.mean()) if len(all_vals) else None,
            within_video_sd=float(within_sd.mean()) if len(within_sd) else None,
            between_video_sd=float(vid_means.std(ddof=1)) if len(vid_means) > 1 else None,
            ci95_width_of_mean=ci_w,
            peaks_per_window=dict(audio=float(na.mean()), video=float(nv.mean())),
        )
        def _f(v, nd=3):
            return "—" if v is None else f"{v:.{nd}f}"
        print(f"d={d:g}s: mean={_f(summary[key]['mean'])}  窗内σ={_f(summary[key]['within_video_sd'])}  "
              f"CI95宽={_f(ci_w)}  峰/窗 a={na.mean():.1f} v={nv.mean():.1f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(dict(videos=list(per_video), durations=args.durations,
                       mode="window", per_video=per_video, summary=summary), f, indent=1)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
