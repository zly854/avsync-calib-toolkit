"""avalign_offset_calib.json -> Paper B 正文数字（§3.1/§3.3/§4/Fig.2a）。"""
import json, sys
import numpy as np

d = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "avalign_offset_calib.json"))
pv, summ = d["per_video"], d["summary"]
print(f"n_videos={d['n_videos']}  win={d['win']}s  fps={[round(pv[n]['fps'],1) for n in sorted(pv)][:3]}...")

print(f"\n{'δ(s)':>8} {'60s均值':>8} {'±std':>6} | {'4s池均值':>8} {'±std':>6} {'n窗':>5}")
for k in sorted(summ, key=float):
    s = summ[k]
    print(f"{k:>8} {s['full_mean']:>8.3f} {s['full_std']:>6.3f} | "
          f"{s['win_mean']:>8.3f} {s['win_std']:>6.3f} {s['n_win']:>5}")

def g(k, fld="full_mean"): return summ[k][fld]
print("\n== 论文口径对照（旧 3 冒烟片 → 新 RealRef n=14） ==")
print(f"AAC 等价(-0.064) vs 补偿(0):  60s {g('-0.064'):.3f} -> {g('+0.000'):.3f}"
      f"   | 4s {g('-0.064','win_mean'):.3f} -> {g('+0.000','win_mean'):.3f}")
print(f"过补偿(+0.064):               60s {g('+0.064'):.3f}   | 4s {g('+0.064','win_mean'):.3f}")
print(f"双倍延迟(-0.128):             60s {g('-0.128'):.3f}   | 4s {g('-0.128','win_mean'):.3f}")

ks = sorted(summ, key=float)
neg = [k for k in ks if float(k) <= 0][::-1]   # 0 -> -1.0
pos = [k for k in ks if float(k) >= 0]          # 0 -> +1.0
def mono(seq, fld):
    v = [g(k, fld) for k in seq]
    return all(v[i] >= v[i+1] - 1e-9 for i in range(len(v)-1)), [round(x,3) for x in v]
for fld, tag in [("full_mean","60s"), ("win_mean","4s")]:
    mp, vp = mono(pos, fld); mn, vn = mono(neg, fld)
    print(f"{tag} 单调性: 正向 {'✓' if mp else '✗'} {vp} | 负向 {'✓' if mn else '✗'} {vn}")

w0 = np.array([v for n in pv for v in pv[n]["offsets"]["+0.000"]["win_iou"] if v is not None])
print(f"\n4s 窗 @δ=0: n={len(w0)}  极差 [{w0.min():.3f}, {w0.max():.3f}]  IQR "
      f"[{np.percentile(w0,25):.3f}, {np.percentile(w0,75):.3f}]  中位 {np.median(w0):.3f}")
pvm = {n: np.nanmean([v for v in pv[n]["offsets"]["+0.000"]["win_iou"] if v is not None]) for n in pv}
print("per-video 4s 均值极差: [{:.3f}, {:.3f}]".format(min(pvm.values()), max(pvm.values())))
lags = [pv[n].get("aac_roundtrip", {}).get("realized_lag_ms") for n in sorted(pv)]
print("AAC 往返 lag(ms):", [round(x,1) if x is not None else "ERR" for x in lags])
