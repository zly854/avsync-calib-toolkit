"""B.2b AV-Align 交叉配对控制（Paper B §3.3 "density not alignment" 收口）。

R2 残留质疑：AV-Align 读数不响应对齐，但现稿只有注入偏移的平坦响应作证。
本脚本用 peaks_cache 纯算术补上错配对照：
  matched   = (apk_i, vpk_i)  14 条对齐内容
  mismatch  = (apk_i, vpk_j), i≠j  14×13 有序对——音频与视频来自不同真实视频，
              任何"对齐"读数都是密度巧合
  chance    = 对每条 matched 视频，均匀随机重采样同数量音频峰 × R 次
  static    = vpk = ∅ 对照（无运动视频的读数由构造 = 0，密度决定论的退化锚点）

判据：错配 ≈ 对齐 ≈ chance → §3.3 主张完全闭合；
若 matched 显著高于 mismatch → 现稿收窄表述已兼容，补一句限定。

另报密度预测检验：独立泊松近似 E[IoU]（由 n_apk, n_vpk, dur, ±1/fps 容差算出）
对 196 个观测（14 matched + 182 mismatch）的 Pearson r——r 高 = 读数被密度
单独预测。

用法：
  python eval/drift/avalign_crosspair.py \
    --refdir results/RealRef \
    --out results/RealRef/avalign_crosspair.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from av_align_curve import iou_in_window  # noqa: E402


def analytic_iou(na, nv, dur, fps):
    """独立泊松近似的期望 IoU：匹配容差 ±1/fps。"""
    if na + nv == 0 or dur <= 0:
        return float("nan")
    w = 2.0 / fps
    p = 1.0 - np.exp(-nv * w / dur)
    m = na * p
    return m / (na + nv - m) if (na + nv - m) > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refdir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chance-draws", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cache = Path(args.refdir) / "peaks_cache"
    peaks = {p.stem.replace("_peaks", ""): json.load(open(p))
             for p in sorted(cache.glob("*_peaks.json"))}
    names = sorted(peaks)
    print(f"{len(names)} refs loaded from {cache}")
    rng = np.random.default_rng(args.seed)

    def score(apk, vpk, fps, t1):
        iou, na, nv = iou_in_window(np.asarray(apk), np.asarray(vpk), fps, 0.0, t1)
        return iou, na, nv

    matched, mism, chance, static = [], [], [], []
    pred_obs = []  # (analytic, observed) for matched+mismatch
    for i in names:
        pi = peaks[i]
        iou, na, nv = score(pi["apk"], pi["vpk"], pi["fps"], pi["dur"])
        matched.append(dict(name=i, iou=iou, na=na, nv=nv, dur=pi["dur"]))
        pred_obs.append((analytic_iou(na, nv, pi["dur"], pi["fps"]), iou))
        # chance：均匀随机 apk 同数量
        for _ in range(args.chance_draws):
            rapk = rng.uniform(0, pi["dur"], size=len(pi["apk"]))
            c, _, _ = score(rapk, pi["vpk"], pi["fps"], pi["dur"])
            chance.append(c)
        # static：vpk 置空
        s, _, _ = score(pi["apk"], [], pi["fps"], pi["dur"])
        static.append(0.0 if np.isnan(s) else s)
        for j in names:
            if j == i:
                continue
            pj = peaks[j]
            t1 = min(pi["dur"], pj["dur"])
            iou, na, nv = score(pi["apk"], pj["vpk"], pj["fps"], t1)
            mism.append(dict(a=i, v=j, iou=iou, na=na, nv=nv, dur=t1))
            pred_obs.append((analytic_iou(na, nv, t1, pj["fps"]), iou))

    m = np.array([x["iou"] for x in matched])
    x = np.array([x["iou"] for x in mism])
    ch = np.array(chance)
    from scipy import stats
    t, p_t = stats.ttest_ind(m, x, equal_var=False)
    u, p_u = stats.mannwhitneyu(m, x, alternative="two-sided")
    sp = np.sqrt(((len(m) - 1) * m.std(ddof=1) ** 2 + (len(x) - 1) * x.std(ddof=1) ** 2)
                 / (len(m) + len(x) - 2))
    d = (m.mean() - x.mean()) / sp if sp > 0 else float("nan")
    po = np.array([q for q in pred_obs if np.isfinite(q[0]) and np.isfinite(q[1])])
    r_pred = float(np.corrcoef(po[:, 0], po[:, 1])[0, 1])

    out = dict(
        n_matched=len(m), n_mismatch=len(x),
        n_chance=len(ch), chance_draws_per_video=args.chance_draws,
        matched=dict(mean=float(m.mean()), std=float(m.std(ddof=1)), values=m.tolist()),
        mismatch=dict(mean=float(x.mean()), std=float(x.std(ddof=1))),
        chance=dict(mean=float(ch.mean()), std=float(ch.std(ddof=1))),
        static=dict(mean=float(np.mean(static)), values=static),
        welch=dict(t=float(t), p=float(p_t)),
        mannwhitney=dict(u=float(u), p=float(p_u)),
        cohens_d=float(d),
        density_pred_r=r_pred,
        matched_rows=matched, mismatch_rows=mism,
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)

    print(f"\nmatched  : {m.mean():.4f} ± {m.std(ddof=1):.4f}  (n={len(m)})")
    print(f"mismatch : {x.mean():.4f} ± {x.std(ddof=1):.4f}  (n={len(x)})")
    print(f"chance   : {ch.mean():.4f} ± {ch.std(ddof=1):.4f}  (n={len(ch)})")
    print(f"static   : {np.mean(static):.4f}")
    print(f"Welch t={t:+.2f} p={p_t:.3f} | MWU p={p_u:.3f} | Cohen's d={d:+.2f}")
    print(f"密度预测 r={r_pred:.3f} (n={len(po)})")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
