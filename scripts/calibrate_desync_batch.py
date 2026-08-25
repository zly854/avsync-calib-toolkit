"""B.2a 生成视频注入偏移标定（Paper B §5 "迁移已验证" / 限界升级，R1 称 decisive）。

在生成分布（E2r S0 60s）上重复 RealRef 的注入偏移标定：wav 通路峰值平移
{±0.2, ±0.5, ±1.0}s → Synchformer 读出 → 逐视频 OLS 斜率。0 偏移基线复用
既有 sample_XXXX_desync.json（同一模型同一 stride 产出，勿重算）。

判据双赢：slope_gen ≈ slope_real(0.64) → §5 加
"迁移已验证"短段；显著折损 → 量化"生成内容上仪器增益折损"，限界声明
升级为测量结果。

⚠️ Table 2 口径纪律：本脚本落库 per-video 窗级 expect 列表，供 make_table2.py
的 moving-block bootstrap 聚合；任何 resolvable 判断勿用 iid 假设直算。

用法（服务器，每 GPU 一个进程分摊视频）：
  python eval/drift/calibrate_desync_batch.py \
    --dir results/E2r_s0_60s --limit 24 --shard 0 --nshard 2 \
    --device cuda:0 --workdir results/E2r_s0_60s/inject_calib/work0
聚合（CPU）：--aggregate-only
"""
import argparse
import json
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

OFFSETS = [-1.0, -0.5, -0.2, 0.2, 0.5, 1.0]   # 0.0 基线复用既有 desync json


def run_shard(args):
    import torch
    from desync_core import build_model, load_streams, desync_curve, AFPS

    def shift_wav(wav, delta_sec):
        n = int(round(abs(delta_sec) * AFPS))
        if n == 0:
            return wav.clone()
        if delta_sec > 0:
            return torch.cat([wav[n:], torch.zeros(n, dtype=wav.dtype)])
        return torch.cat([torch.zeros(n, dtype=wav.dtype), wav[:-n]])

    root = Path(args.dir)
    out_dir = root / "inject_calib"
    out_dir.mkdir(parents=True, exist_ok=True)
    videos = sorted(root.glob("sample_*.mp4"))[: args.limit][args.shard :: args.nshard]
    print(f"shard {args.shard}/{args.nshard}: {len(videos)} videos on {args.device}",
          flush=True)

    M = build_model(device=args.device)
    for v in videos:
        op = out_dir / f"{v.stem}_inject.json"
        if op.exists():
            print(f"[{v.stem}] exists, skip", flush=True)
            continue
        wav_path = v.with_suffix(".wav")
        assert wav_path.exists(), f"缺 wav sidecar: {wav_path}"
        rgb, wav0, meta = load_streams(str(v), str(wav_path), workdir=args.workdir)
        res = {}
        for d in args.offsets:
            c = desync_curve(M, rgb, shift_wav(wav0, d), meta,
                             stride=args.stride, path=str(v))
            e = np.asarray(c["expect"])
            res[f"{d:+.2f}"] = dict(
                expect=[float(x) for x in e],
                mean=float(e.mean()), median=float(np.median(e)),
                mean_abs=float(np.abs(e).mean()),
                sat_frac=float(np.mean(c["saturated"])))
            print(f"[{v.stem}] {d:+.2f}: mean={e.mean():+.3f} "
                  f"sat={np.mean(c['saturated']):.2f}", flush=True)
        with open(op, "w") as f:
            json.dump(dict(video=v.name, stride=args.stride, offsets=res), f, indent=1)
    print("shard done", flush=True)


def aggregate(args):
    root = Path(args.dir)
    out_dir = root / "inject_calib"
    rows = []
    for op in sorted(out_dir.glob("sample_*_inject.json")):
        d = json.load(open(op))
        stem = op.stem.replace("_inject", "")
        base = root / f"{stem}_desync.json"
        b = json.load(open(base))
        e0 = np.asarray(b["desync_expect"], float)
        xs, ys = [0.0], [float(e0.mean())]
        for k, v in d["offsets"].items():
            xs.append(float(k))
            ys.append(v["mean"])
        xs, ys = np.asarray(xs), np.asarray(ys)
        slope = float(np.polyfit(xs, ys, 1)[0])
        rows.append(dict(video=stem, slope=slope,
                         floor_abs0=float(np.abs(e0).mean()),
                         readout={f"{x:+.2f}": float(y) for x, y in zip(xs, ys)}))
    sl = np.array([r["slope"] for r in rows])
    rng = np.random.default_rng(0)
    boots = [np.mean(rng.choice(sl, size=len(sl))) for _ in range(10000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    fl = np.array([r["floor_abs0"] for r in rows])
    out = dict(n_videos=len(rows), stride_note="0 偏移来自既有 desync json（同模型同 stride）",
               slope_mean=float(sl.mean()), slope_std=float(sl.std(ddof=1)),
               slope_ci95=[float(lo), float(hi)],
               floor_abs0_mean=float(fl.mean()), floor_abs0_std=float(fl.std(ddof=1)),
               real_slope_ref=args.real_slope, per_video=rows)
    op = out_dir / "aggregate_inject.json"
    with open(op, "w") as f:
        json.dump(out, f, indent=1)
    print(f"n={len(rows)}  slope_gen={sl.mean():.3f}±{sl.std(ddof=1):.3f} "
          f"CI95=[{lo:.3f},{hi:.3f}]  vs real={args.real_slope}")
    print(f"floor|0|_gen={fl.mean():.3f}±{fl.std(ddof=1):.3f}")
    print(f"-> {op}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--limit", type=int, default=24)
    ap.add_argument("--offsets", type=float, nargs="+", default=OFFSETS)
    ap.add_argument("--stride", type=float, default=1.0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--workdir", default="/tmp/desync_work_inject")
    ap.add_argument("--real-slope", type=float, default=0.64)
    ap.add_argument("--aggregate-only", action="store_true")
    args = ap.parse_args()
    if args.aggregate_only:
        aggregate(args)
    else:
        run_shard(args)


if __name__ == "__main__":
    main()
