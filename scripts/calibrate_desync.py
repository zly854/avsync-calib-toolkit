"""DeSync 指标灵敏度标定。

在已知同步的真实视频上人为注入 ±0.2/0.5/1.0s 音轨平移，检查 DeSync 曲线能否读出。
这是审稿人问"你的指标可信吗"时的挡箭牌，也是解释生成视频 DeSync 曲线的噪声底。

产出三个数：
  - **噪声底**：0 偏移时 |DeSync| 的均值（读数不可能比它更准）
  - **灵敏度**：读出偏移 vs 注入偏移的回归斜率（理想 = 1.0）与符号约定
  - **可分辨阈**：多大的偏移才能与 0 偏移显著区分（bootstrap）

用法：python eval/drift/calibrate_desync.py --video ref60.mp4 --audio ref60.wav
"""
import argparse, json
from pathlib import Path

import numpy as np
import torch

from desync_core import build_model, load_streams, desync_curve, AFPS


def shift_wav(wav: torch.Tensor, delta_sec: float) -> torch.Tensor:
    """把音频相对视频平移 delta 秒。delta>0 = 丢掉开头 → 音频事件提前（音频领先）。"""
    n = int(round(abs(delta_sec) * AFPS))
    if n == 0:
        return wav.clone()
    if delta_sec > 0:
        return torch.cat([wav[n:], torch.zeros(n, dtype=wav.dtype)])
    return torch.cat([torch.zeros(n, dtype=wav.dtype), wav[:-n]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--audio", default=None)
    ap.add_argument("--offsets", type=float, nargs="+",
                    default=[-1.0, -0.5, -0.2, 0.0, 0.2, 0.5, 1.0])
    ap.add_argument("--stride", type=float, default=1.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    video = str(Path(args.video).resolve())
    audio = str(Path(args.audio).resolve()) if args.audio else None
    out = str(Path(args.out).resolve()) if args.out else None

    M = build_model(device=args.device)
    rgb, wav0, meta = load_streams(video, audio)
    print(f"标定源: {Path(video).name}  {min(len(rgb)/25, len(wav0)/AFPS):.1f}s\n")

    rows, per_offset = [], {}
    print(f"{'注入偏移':>10} {'读出均值':>10} {'读出中位':>10} {'|读数|均值':>11} {'std':>7} {'饱和':>6}")
    for d in args.offsets:
        c = desync_curve(M, rgb, shift_wav(wav0, d), meta, stride=args.stride, path=video)
        e = np.asarray(c["expect"])
        per_offset[f"{d:+.2f}"] = dict(curve=c,
                                       mean=float(e.mean()), median=float(np.median(e)),
                                       std=float(e.std()), mean_abs=float(np.abs(e).mean()),
                                       sat_frac=float(np.mean(c["saturated"])))
        rows.append((d, e))
        print(f"{d:>+10.2f} {e.mean():>+10.3f} {np.median(e):>+10.3f} "
              f"{np.abs(e).mean():>11.3f} {e.std():>7.3f} {np.mean(c['saturated'])*100:>5.0f}%")

    inj = np.array([r[0] for r in rows])
    read = np.array([r[1].mean() for r in rows])
    A = np.vstack([inj, np.ones_like(inj)]).T
    slope, icpt = np.linalg.lstsq(A, read, rcond=None)[0]
    r = float(np.corrcoef(inj, read)[0, 1])

    zero = np.asarray(per_offset["+0.00"]["curve"]["expect"])
    noise_floor = float(np.abs(zero).mean())

    # 可分辨阈：各偏移的窗级读数与 0 偏移读数做 bootstrap 均值差检验
    rng = np.random.default_rng(0)
    resolvable = {}
    for d, e in rows:
        if d == 0.0:
            continue
        diffs = [rng.choice(e, len(e)).mean() - rng.choice(zero, len(zero)).mean()
                 for _ in range(2000)]
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        resolvable[f"{d:+.2f}"] = dict(ci_lo=float(lo), ci_hi=float(hi),
                                       separable=bool(lo > 0 or hi < 0))

    print(f"\n灵敏度回归: 读出 = {slope:.3f} × 注入 {icpt:+.3f}   (理想斜率 1.0)   r = {r:.4f}")
    print(f"符号约定: 注入正偏移(音频提前) → 读数{'为正' if slope > 0 else '为负'}")
    print(f"噪声底 |DeSync| @0偏移: {noise_floor:.3f}s")
    print("可分辨性（与 0 偏移的 95% bootstrap CI 是否跨 0）:")
    for k, v in resolvable.items():
        print(f"  {k}s: CI=[{v['ci_lo']:+.3f}, {v['ci_hi']:+.3f}]  "
              f"{'可分辨' if v['separable'] else '不可分辨'}")

    res = dict(video=video, offsets=args.offsets, stride=args.stride,
               sensitivity=dict(slope=float(slope), intercept=float(icpt), pearson_r=r),
               noise_floor_abs=noise_floor, resolvable=resolvable,
               per_offset={k: {kk: vv for kk, vv in v.items() if kk != "curve"}
                           for k, v in per_offset.items()},
               curves={k: v["curve"] for k, v in per_offset.items()})
    dst = out or str(Path(video).with_name(Path(video).stem + "_desync_calib.json"))
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\n-> {dst}")


if __name__ == "__main__":
    main()
