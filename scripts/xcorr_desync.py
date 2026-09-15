"""XCorrOffset(t)：分段包络互相关的无量程 AV offset 曲线（Synchformer ±2s 饱和的补充）。

动机：S0 在 30s 已 19% 平均饱和，60s 曲线
尾部会假性走平；需要一个量程可配（默认 ±5s）的独立测量交叉验证曲线尾部。

方法（轻依赖，librosa + torchvision 即可）：
- 音频包络：librosa onset_strength（谱通量），→ 重采样到公共率 fr=100Hz；
- 视频包络：相邻帧绝对差的全图均值（帧差能量，16fps → 上采样到 fr）——
  与 AV-Align 的光流峰同源但更便宜，且不做峰化保留连续形状；
- 每段（默认 8s，stride 2s）对两包络去均值互相关，峰位 = 该段 offset，
  抛物线插值到亚采样精度；置信度 = 峰值/次峰值比（>1.2 记 locked）。
- 符号约定与 shift_wav 一致：offset>0 = 音频事件晚于视频（音频滞后）。

解释边界（写论文时要声明）：包络互相关测的是**能量事件对齐**，对 D 类环境底噪
（无离散事件）天然失锁 → locked 率本身就是可测性的报告项，不硬给数。

用法：
  python scripts/xcorr_desync.py --video x.mp4 [--audio x.wav] \
      [--seg 8.0 --stride 2.0 --max-lag 5.0] --out x_xcorr.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

FR = 100.0  # 公共包络采样率 Hz


def audio_envelope(wav_path, fr=FR):
    import librosa
    y, sr = librosa.load(str(wav_path), sr=16000, mono=True)
    hop = 160  # 10ms @16k → 100Hz
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    return env, len(y) / sr


def video_envelope(video_path, fr=FR):
    import torchvision
    rgb, _, info = torchvision.io.read_video(str(video_path), pts_unit="sec",
                                             output_format="TCHW")
    fps = float(info["video_fps"])
    x = rgb.float().mean(1)  # 灰度 [T,H,W]
    diff = (x[1:] - x[:-1]).abs().mean(dim=(1, 2)).numpy()
    diff = np.concatenate([[diff[0]], diff])  # 首帧补齐
    # 上采样到 fr（线性插值足够：帧差能量本来就是 16Hz 带宽）
    t_src = np.arange(len(diff)) / fps
    t_dst = np.arange(0, t_src[-1], 1 / fr)
    return np.interp(t_dst, t_src, diff), len(diff) / fps


def xcorr_offset(a, v, fr=FR, max_lag_s=5.0):
    """返回 (offset_seconds, peak_ratio)。offset>0 = 音频滞后于视频。"""
    n = min(len(a), len(v))
    a = a[:n] - a[:n].mean()
    v = v[:n] - v[:n].mean()
    if a.std() < 1e-9 or v.std() < 1e-9:
        return None, 0.0
    a, v = a / a.std(), v / v.std()
    max_lag = int(round(max_lag_s * fr))
    N = 1 << (2 * n - 1).bit_length()
    xc = np.fft.irfft(np.fft.rfft(a, N) * np.conj(np.fft.rfft(v, N)), N)
    xc = np.concatenate([xc[-max_lag:], xc[:max_lag + 1]])  # lags in [-max_lag, +max_lag]
    k = int(np.argmax(xc))
    # 次峰：抑制主峰 ±0.25s 邻域后再取最大
    guard = int(0.25 * fr)
    xc2 = xc.copy()
    xc2[max(0, k - guard):k + guard + 1] = -np.inf
    ratio = float(xc[k] / xc2.max()) if np.isfinite(xc2.max()) and xc2.max() > 0 else np.inf
    # 抛物线插值
    if 0 < k < len(xc) - 1:
        y0, y1, y2 = xc[k - 1], xc[k], xc[k + 1]
        denom = y0 - 2 * y1 + y2
        frac = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-12 else 0.0
    else:
        frac = 0.0
    lag = (k - max_lag + frac) / fr
    # 音频包络相对视频包络右移 lag → 音频事件晚 lag 秒
    return float(lag), ratio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--audio", default=None, help="无损 wav；默认同名 .wav（严禁 AAC 通路）")
    ap.add_argument("--seg", type=float, default=8.0)
    ap.add_argument("--stride", type=float, default=2.0)
    ap.add_argument("--max-lag", type=float, default=5.0)
    ap.add_argument("--lock-ratio", type=float, default=1.2)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    video = Path(args.video)
    audio = Path(args.audio) if args.audio else video.with_suffix(".wav")
    if not audio.exists():
        raise FileNotFoundError(f"缺少无损音轨 {audio}")

    a_env, a_dur = audio_envelope(audio)
    v_env, v_dur = video_envelope(video)
    dur = min(a_dur, v_dur)
    seg_n = int(args.seg * FR)
    stride_n = int(args.stride * FR)
    n = min(len(a_env), len(v_env))

    t, off, ratio = [], [], []
    for s0 in range(0, n - seg_n + 1, stride_n):
        o, r = xcorr_offset(a_env[s0:s0 + seg_n], v_env[s0:s0 + seg_n],
                            max_lag_s=args.max_lag)
        t.append((s0 + seg_n / 2) / FR)
        off.append(o)
        ratio.append(r)
    if not t:
        raise SystemExit(f"时长 {dur:.1f}s 不足一个 {args.seg}s 段")

    off_a = np.array([np.nan if o is None else o for o in off], float)
    locked = np.isfinite(off_a) & (np.array(ratio) >= args.lock_ratio)
    ol = np.where(locked, off_a, np.nan)
    valid = ol[np.isfinite(ol)]
    tv = np.array(t)[np.isfinite(ol)]
    scal = dict(frac_locked=float(locked.mean()), n_seg=len(t))
    if len(valid) >= 3:
        k = max(1, int(round(0.2 * len(valid))))
        scal["D_end"] = float(np.abs(valid[-k:]).mean() - np.abs(valid[:k]).mean())
        try:
            from scipy.stats import theilslopes
            scal["slope_abs"] = float(theilslopes(np.abs(valid), tv)[0])
        except Exception:
            scal["slope_abs"] = float(np.polyfit(tv, np.abs(valid), 1)[0])
        scal["mean_abs"] = float(np.abs(valid).mean())
        scal["end_offset"] = float(valid[-1])
    out = dict(video=str(video), audio=str(audio), dur=float(dur),
               seg=args.seg, stride=args.stride, max_lag=args.max_lag,
               lock_ratio=args.lock_ratio,
               t=t, offset=[None if not np.isfinite(o) else float(o) for o in off_a],
               peak_ratio=[float(r) if np.isfinite(r) else None for r in ratio],
               locked=locked.tolist(), scalars=scal)
    dst = args.out or str(video.parent / (video.stem + "_xcorr.json"))
    with open(dst, "w") as f:
        json.dump(out, f, indent=1)
    msg = f"[{video.name}] locked={scal['frac_locked']*100:.0f}%"
    if "mean_abs" in scal:
        msg += (f"  |off|均值={scal['mean_abs']:.3f}s  D_end={scal['D_end']:+.3f}s"
                f"  末端={scal['end_offset']:+.3f}s")
    print(msg)
    print(f"  -> {dst}")


if __name__ == "__main__":
    main()
