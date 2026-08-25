"""Table 2 行生成：ext 17 偏移的有符号可分辨计数。

2026-08-24 起改用 moving-block bootstrap（块长 8 窗 ≥ 5 窗重叠依赖范围，
2000 次，95% 分位 CI），替代 calibrate_desync.py 落库的 i.i.d. 窗重采样
（重叠窗强自相关 → i.i.d. CI 反保守，评审轮 R1 指出并证实）。逐窗曲线
直接取 ext json 的 curves 字段，无需重跑标定。旧 i.i.d. 计数以 --iid 复现。
"""
import argparse
import glob
import json

import numpy as np

MAGS = ["0.05", "0.10", "0.20", "0.50", "1.00", "1.50", "2.50", "3.00"]
BLOCK, N_BOOT = 8, 2000


def block_boot_separable(e, z, n_boot=N_BOOT, blk=BLOCK, seed=0):
    """95% moving-block bootstrap CI of mean(e)-mean(z) excludes 0?"""
    rng = np.random.default_rng(seed)

    def resample(x):
        n = len(x)
        nb = int(np.ceil(n / blk))
        starts = rng.integers(0, n, size=(n_boot, nb))
        idx = (starts[:, :, None] + np.arange(blk)[None, None, :]) % n
        return x[idx.reshape(n_boot, -1)[:, :n]]

    diffs = resample(e).mean(axis=1) - resample(z).mean(axis=1)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return lo > 0 or hi < 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iid", action="store_true",
                    help="旧口径：读 calibrate_desync.py 落库的 i.i.d. 判定")
    args = ap.parse_args()

    adv, dly = {m: 0 for m in MAGS}, {m: 0 for m in MAGS}
    n = 0
    for p in sorted(glob.glob("figs/data/ref*_desync_calib_ext.json")):
        j = json.load(open(p))
        n += 1
        if args.iid:
            d = j["resolvable"]
            for m in MAGS:
                if d.get(f"+{m}", {}).get("separable"):
                    adv[m] += 1
                if d.get(f"-{m}", {}).get("separable"):
                    dly[m] += 1
            continue
        c = j["curves"]
        z = np.asarray(c["+0.00"]["expect"])
        for m in MAGS:
            for key, cnt in ((f"+{m}", adv), (f"-{m}", dly)):
                if key in c and block_boot_separable(
                        np.asarray(c[key]["expect"]), z):
                    cnt[m] += 1

    mode = "iid (legacy)" if args.iid else f"block bootstrap (blk={BLOCK})"
    print(f"n = {n}   mode = {mode}")
    print("advanced:", [adv[m] for m in MAGS])
    print("delayed :", [dly[m] for m in MAGS])


if __name__ == "__main__":
    main()
