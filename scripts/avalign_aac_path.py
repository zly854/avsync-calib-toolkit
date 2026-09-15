"""AV-Align 的真实 AAC 容器通路直测（Paper B §3.1 终极数字）。

对每条 RealRef：wav → ffmpeg AAC(m4a) → 解码（-ignore_editlist 1，复现被审计
工具链"不认 priming 元数据"的读端行为）→ librosa onset → 与 peaks_cache 缓存的
光流峰算 IoU。与无损通路（v2 δ=0）在同一支撑上对照。
机制背景：priming 近静音头的谱通量尖峰成为 onset 包络 max，相对阈值全局抬升
（ref05 直测 292→3 onsets）。本脚本给出 n=14 的 IoU 母体数字。

用法：
  python scripts/avalign_aac_path.py \
    --refdir results/RealRef \
    --out results/RealRef/avalign_aac_path.json --support-pad 1.0
"""
import argparse, json, subprocess, sys, tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from av_align_curve import audio_peaks, iou_in_window  # noqa: E402
import soundfile as sf                                  # noqa: E402


def aac_noedit(wav_path, sr_out=16000):
    with tempfile.TemporaryDirectory() as td:
        m4a, dec = f"{td}/a.m4a", f"{td}/a.wav"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", wav_path,
                        "-c:a", "aac", "-b:a", "192k", m4a], check=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ignore_editlist", "1",
                        "-i", m4a, "-ar", str(sr_out), dec], check=True)
        y, sr = sf.read(dec)
    return (y if y.ndim == 1 else y.mean(1)).astype(np.float32), sr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refdir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--support-pad", type=float, default=1.0,
                    help="与 v2 相同的共同支撑边距")
    args = ap.parse_args()
    refdir = Path(args.refdir)
    per = {}
    print(f"{'ref':>6} {'onsets(无损)':>11} {'onsets(AAC)':>11} {'IoU无损':>8} {'IoUAAC':>8}")
    for cp in sorted((refdir / "peaks_cache").glob("ref*_peaks.json")):
        name = cp.stem.replace("_peaks", "")
        pk = json.load(open(cp))
        fps, dur = pk["fps"], pk["dur"]
        vpk = np.asarray(pk["vpk"]); apk0 = np.asarray(pk["apk"])
        lo, hi = args.support_pad, dur - args.support_pad
        y, sr = aac_noedit(str(refdir / f"{name}.wav"))
        apk_aac = audio_peaks(y, sr)
        iou0, na0, _ = iou_in_window(apk0, vpk, fps, lo, hi)
        ioua, naa, _ = iou_in_window(apk_aac, vpk, fps, lo, hi)
        per[name] = dict(iou_lossless=iou0, iou_aac_noedit=ioua,
                         n_apk_lossless=int(na0), n_apk_aac=int(naa),
                         support=[lo, hi])
        print(f"{name:>6} {na0:>11} {naa:>11} {iou0:>8.3f} {ioua:>8.3f}")
    l = np.array([v["iou_lossless"] for v in per.values()])
    a = np.array([v["iou_aac_noedit"] for v in per.values()])
    summ = dict(lossless_mean=float(l.mean()), lossless_std=float(l.std()),
                aac_mean=float(a.mean()), aac_std=float(a.std()),
                n=len(per),
                collapse_refs=[k for k, v in per.items()
                               if v["n_apk_aac"] < 0.2 * v["n_apk_lossless"]])
    with open(args.out, "w") as f:
        json.dump(dict(per_video=per, summary=summ), f, indent=1)
    print(f"\n无损 {summ['lossless_mean']:.3f}±{summ['lossless_std']:.3f}  vs  "
          f"AAC(no-editlist) {summ['aac_mean']:.3f}±{summ['aac_std']:.3f}")
    print(f"onset 坍塌条目(峰数<20%): {summ['collapse_refs']}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
