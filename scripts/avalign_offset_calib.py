"""AV-Align 注入偏移灵敏度标定 v2（Paper B §3.1/§3.3/§4，RealRef 版）。

v1（波形平移注入）发现并实锤了一个注入方法学伪影：负偏移的头部补零制造
静音→内容边界，其谱通量尖峰成为 onset 包络最大值，librosa onset_detect 的
相对阈值随之抬升，全片真 onset 被压制（ref05 实测 292→3 峰，env_max 4.3→24.3）。
故 v2 改为**峰时间戳平移**：对 AV-Align 而言峰集合是充分统计量，
apk' = apk − delta 与音频整体平移严格等价，且无任何边界伪影。
所有偏移在共同支撑 [pad, dur−pad]（pad=max|delta|）上评估，密度可比。

v1 的波形平移结果保留为 --mode waveform（伪影演示用，勿作响应曲线）。

符号约定不变：delta>0 = 音频提前（事件时刻变早）。
峰提取（光流+onset）每条视频只算一次并缓存到 <refdir>/peaks_cache/，
之后任意偏移扫描为纯算术。

用法：
  python scripts/avalign_offset_calib.py \
    --refdir results/RealRef \
    --out results/RealRef/avalign_offset_calib_v2.json --workers 6
"""
import argparse, json, os, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from av_align_curve import read_frames, audio_peaks, video_peaks, iou_in_window  # noqa: E402

import cv2                                                        # noqa: E402
import soundfile as sf                                            # noqa: E402

OFFSETS = [-1.0, -0.5, -0.2, -0.128, -0.064, -0.04, -0.02,
           0.0, 0.02, 0.04, 0.064, 0.128, 0.2, 0.5, 1.0]


def shift_wav(wav, delta_sec, sr):
    n = int(round(abs(delta_sec) * sr))
    if n == 0:
        return wav.copy()
    if delta_sec > 0:
        return np.concatenate([wav[n:], np.zeros(n, dtype=wav.dtype)])
    return np.concatenate([np.zeros(n, dtype=wav.dtype), wav[:-n]])


def extract_peaks(task):
    """光流峰 + onset 峰，一次并缓存。"""
    video, wav_path, cache = task
    os.nice(10)
    cv2.setNumThreads(2)
    name = Path(video).stem
    cp = Path(cache) / f"{name}_peaks.json"
    if cp.exists():
        d = json.load(open(cp))
        print(f"[{name}] cache hit", flush=True)
        return name, d
    frames, fps = read_frames(video)
    wav, sr = sf.read(wav_path)
    wav = (wav if wav.ndim == 1 else wav.mean(1)).astype(np.float32)
    dur = min(len(frames) / fps, len(wav) / sr)
    vpk, _ = video_peaks(frames, fps)
    del frames
    apk = audio_peaks(wav, sr)
    d = dict(fps=float(fps), dur=float(dur),
             apk=[float(t) for t in apk], vpk=[float(t) for t in vpk])
    cp.parent.mkdir(parents=True, exist_ok=True)
    with open(cp, "w") as f:
        json.dump(d, f)
    print(f"[{name}] peaks: audio={len(apk)} video={len(vpk)} dur={dur:.1f}s", flush=True)
    return name, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refdir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--offsets", type=float, nargs="+", default=OFFSETS)
    ap.add_argument("--win", type=float, default=4.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    refdir = Path(args.refdir)
    videos = sorted(refdir.glob("ref*.mp4"))
    if args.limit:
        videos = videos[: args.limit]
    tasks = [(str(v), str(v.with_suffix(".wav")), str(refdir / "peaks_cache"))
             for v in videos]
    for _, w, _ in tasks:
        assert Path(w).exists(), f"缺 wav sidecar: {w}"
    print(f"{len(tasks)} 条 × {len(args.offsets)} 偏移（峰时间戳平移法），"
          f"win={args.win}s workers={args.workers}", flush=True)

    from multiprocessing import Pool
    with Pool(args.workers) as pool:
        peaks = dict(pool.map(extract_peaks, tasks))

    pad = max(abs(d) for d in args.offsets)
    per_video = {}
    for name, pk in peaks.items():
        fps, dur = pk["fps"], pk["dur"]
        apk0, vpk = np.asarray(pk["apk"]), np.asarray(pk["vpk"])
        lo, hi = pad, dur - pad
        edges = np.arange(lo, hi + 1e-9, args.win)
        res = {"fps": fps, "dur": dur, "support": [lo, hi],
               "n_vpk_support": int(((vpk >= lo) & (vpk < hi)).sum()), "offsets": {}}
        for d in args.offsets:
            apk = apk0 - d                       # delta>0 = 音频提前
            full, na, nv = iou_in_window(apk, vpk, fps, lo, hi)
            wins = [iou_in_window(apk, vpk, fps, edges[i], edges[i + 1])[0]
                    for i in range(len(edges) - 1)]
            res["offsets"][f"{d:+.3f}"] = dict(
                full_iou=full, n_apk=int(na),
                win_iou=[None if np.isnan(v) else float(v) for v in wins])
        per_video[name] = res

    summary = {}
    for k in per_video[next(iter(per_video))]["offsets"]:
        fulls = np.array([per_video[n]["offsets"][k]["full_iou"] for n in per_video])
        pool4 = np.array([v for n in per_video
                          for v in per_video[n]["offsets"][k]["win_iou"] if v is not None])
        summary[k] = dict(full_mean=float(np.nanmean(fulls)), full_std=float(np.nanstd(fulls)),
                          win_mean=float(pool4.mean()), win_std=float(pool4.std()),
                          n_win=int(len(pool4)))
    out = dict(method="peak_timestamp_shift", refdir=str(refdir), win=args.win,
               offsets=args.offsets, support_pad=pad, n_videos=len(per_video),
               per_video=per_video, summary=summary)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)

    print(f"\n{'偏移':>8} {'全片均值':>8} {'std':>6} | {'4s窗均值':>8} {'std':>6} {'n':>5}")
    for k in sorted(summary, key=float):
        s = summary[k]
        print(f"{k:>8} {s['full_mean']:>8.3f} {s['full_std']:>6.3f} | "
              f"{s['win_mean']:>8.3f} {s['win_std']:>6.3f} {s['n_win']:>5}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
