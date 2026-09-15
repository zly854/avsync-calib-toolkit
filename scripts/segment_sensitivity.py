"""D0b 逐时段灵敏度标定。

**为什么必须做**：W2 反转判决的第二条证据是"曲线平不是指标失灵"，但它引用的是
`calibrate_desync.py` 的**全局**注入标定（在一条真实视频上，r=0.9989, slope=0.675）。
结论 11 已证明 DeSync 的读数水平被视频质量强烈混淆（音轨一动不动、只给视频加空间抖动，
读数就 0.250 → 0.516）。而生成视频的质量本身随时间退化 →
**灵敏度可能随 t 衰减**，那样"后段平"就是"指标在后段读不出偏移"，而非"真的不漂"。

方法：在**生成样本自己身上**逐时段就地注入已知偏移，测每一段的读出增益 gain(t)。

判据用**绝对增益水平**，不用末/首比值（8/19 ref60 控制暴露了比值规则的错误——见文件末注）：
  read = gain × true + floor，floor ≈ 0.20-0.25s（RealRef 实测噪声底）
  → 能露出噪声底的最小真实漂移 ≈ floor / gain。
  - gain ≥ 0.5 → 指标可用，曲线平坦可作"不漂"的证据；
  - gain < 0.5 → 指标在该内容上近乎失明，曲线平坦**不能**排除 floor/gain 量级以下的漂移。
  ⚠️ 不要做 obs/gain(t) 的逐点"校正"：观测被噪声底主导，除以小增益只会放大地板，
     不构成漂移证据（8/19 W2 复判已按此处理）。

实测（2026-08-19）：生成内容 gain(seg≥1) 中位 **0.176**；真实视频 ref60 **0.88**
→ **5.0× 增益差**，且生成内容上最小可分辨漂移 ≈ 1.2s。

注意：注入是**全局平移整条音轨**再逐段回归，不是分段平移——分段平移会在段边界
制造不连续，本身就会污染读数（结论 10 的教训）。全局平移下，每一段的真实偏移
都等于注入量，因此逐段回归的斜率就是该段的局部灵敏度。

用法：
  python scripts/segment_sensitivity.py --dir results/<generated_set> --stride 24 \
      --workdir tmp/segment_work --out results/<generated_set>/aggregate/segment_sensitivity.json
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import torch

from desync_core import build_model, load_streams, desync_curve, AFPS
from calibrate_desync import shift_wav


def per_segment_slope(curves, offsets, seg_len, dur):
    """curves[off] = desync_curve 结果。返回 (seg_centers, slopes, n_per_seg)。

    每个滑窗按**窗心**归段（desync_core 返回的 t 已是窗心）。
    """
    t = np.asarray(curves[offsets[0]]["t"])
    reads = {o: np.asarray(curves[o]["expect"]) for o in offsets}
    n_seg = max(1, int(np.ceil(dur / seg_len)))
    inj = np.asarray(offsets, dtype=float)
    A = np.vstack([inj, np.ones_like(inj)]).T

    centers, slopes, counts = [], [], []
    for s in range(n_seg):
        m = (t >= s * seg_len) & (t < (s + 1) * seg_len)
        if m.sum() < 2:
            continue
        y = np.array([reads[o][m].mean() for o in offsets])
        slope, _ = np.linalg.lstsq(A, y, rcond=None)[0]
        centers.append(float((s + 0.5) * seg_len))
        slopes.append(float(slope))
        counts.append(int(m.sum()))
    return centers, slopes, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=None, help="样本目录（取其中 sample_*.mp4 + 同名 .wav）")
    ap.add_argument("--videos", nargs="+", default=None, help="显式视频列表（覆盖 --dir）")
    ap.add_argument("--every", type=int, default=2,
                    help="从 --dir 里每 N 条取 1 条（默认 2 → 48 条取 24 条子集）")
    ap.add_argument("--offsets", type=float, nargs="+", default=[-0.5, 0.0, 0.5])
    ap.add_argument("--seg", type=float, default=10.0, help="分段长度（秒）")
    ap.add_argument("--stride", type=float, default=1.0, help="DeSync 滑窗步长")
    ap.add_argument("--workdir", required=True,
                    help="**必须独立**：desync 的 reencode 缓存按文件 stem 命名，跨 run 会撞（E1 踩过）")
    ap.add_argument("--start-index", type=int, default=0, help="多卡分片用")
    ap.add_argument("--end-index", type=int, default=10**9)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    if args.videos:
        vids = [str(Path(v).resolve()) for v in args.videos]
    else:
        vids = sorted(glob.glob(str(Path(args.dir).resolve() / "sample_*.mp4")))[:: args.every]
    vids = vids[args.start_index:args.end_index]
    if not vids:
        raise SystemExit("没有匹配到样本")

    offsets = [round(float(o), 4) for o in args.offsets]
    assert 0.0 in offsets and len(offsets) >= 3, "偏移集须含 0 且 ≥3 点才能回归"

    M = build_model(device=args.device)
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    per_video = {}
    for i, v in enumerate(vids):
        wav_path = Path(v).with_suffix(".wav")
        if not wav_path.exists():
            print(f"[skip] 无 wav sidecar: {v}")
            continue
        rgb, wav0, meta = load_streams(v, str(wav_path), workdir=args.workdir)
        dur = min(len(rgb) / 25.0, len(wav0) / AFPS)
        curves = {}
        for o in offsets:
            curves[o] = desync_curve(M, rgb, shift_wav(wav0, o), meta,
                                     stride=args.stride, path=v)
        centers, slopes, counts = per_segment_slope(curves, offsets, args.seg, dur)
        per_video[Path(v).stem] = dict(dur=dur, seg_centers=centers, slopes=slopes,
                                       n_windows=counts)
        print(f"[{i+1}/{len(vids)}] {Path(v).stem}  dur={dur:.1f}s  "
              f"slope(t)=" + " ".join(f"{s:.3f}" for s in slopes))

    # ---- 聚合 ----
    lens = {len(v["slopes"]) for v in per_video.values()}
    n_seg = min(lens) if lens else 0
    S = np.array([v["slopes"][:n_seg] for v in per_video.values()])  # [n_video, n_seg]
    centers = list(per_video.values())[0]["seg_centers"][:n_seg]

    rng = np.random.default_rng(0)
    boot = np.array([S[rng.integers(0, len(S), len(S))].mean(0) for _ in range(2000)])
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5], axis=0)
    mean_slope = S.mean(0)
    ratio = float(mean_slope[-1] / mean_slope[0]) if mean_slope[0] != 0 else float("nan")
    ratio_boot = boot[:, -1] / boot[:, 0]
    r_lo, r_hi = (float(x) for x in np.percentile(ratio_boot, [2.5, 97.5]))

    # 判据以**绝对增益水平**为准，不用末/首比值。
    # 理由（8/19 ref60 控制暴露）：真实视频 ref60 的 seg0 增益 -0.064（开场镜头无同步内容），
    # 末/首 = -10.8，按比值规则会把"增益最好的那条视频"判成衰减。真正决定指标能否读出
    # 漂移的是增益的绝对值：read = gain × true + floor，floor ≈ 0.2-0.25s（RealRef 实测），
    # 所以 gain=0.18 时，真实漂移要 > floor/gain ≈ 1.2s 才能露出噪声底。
    level = float(np.median(mean_slope[1:])) if len(mean_slope) > 1 else float(mean_slope[0])
    detectable = 0.22 / level if level > 0 else float("inf")   # 能露出噪声底的最小真实漂移
    verdict = (f"增益水平 {level:.3f}（seg≥1 中位）→ 可分辨的最小真实漂移 ≈ {detectable:.2f}s；"
               + ("指标可用，曲线平坦可作证据" if level >= 0.5 else
                  "⚠️ 指标在该内容上近乎失明，曲线平坦**不能**排除该量级以下的漂移"))

    print(f"\n{'段中心(s)':>10} {'slope':>8} {'95%CI':>20}")
    for j, c in enumerate(centers):
        print(f"{c:>10.1f} {mean_slope[j]:>8.3f}   [{ci_lo[j]:+6.3f},{ci_hi[j]:+6.3f}]")
    print(f"\nn_video={len(S)}  seg≥1 增益中位 = {level:.3f}  "
          f"（参照：真实视频 ref60 = 0.88）  末段/首段 = {ratio:.3f} [{r_lo:.3f},{r_hi:.3f}]")
    print(f"→ {verdict}")

    res = dict(videos=len(S), offsets=offsets, seg_len=args.seg, stride=args.stride,
               seg_centers=centers, mean_slope=mean_slope.tolist(),
               ci_lo=ci_lo.tolist(), ci_hi=ci_hi.tolist(),
               ratio_last_first=ratio, ratio_ci=[r_lo, r_hi],
               gain_level=level, min_detectable_drift_s=detectable,
               threshold_gain=0.5, verdict=verdict, per_video=per_video)
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2)
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
