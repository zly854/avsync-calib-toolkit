"""DeSync 测量核心：Synchformer 滑窗推理，供 desync_curve.py / calibrate_desync.py 复用。

两条硬约束（Phase 0 标定结论，见 EXPERIMENT_LOG 关键结论 1/6）：
  1. 音频必须来自无损 wav —— mp4 的 AAC 轨带 +64 ms 固定 priming 偏置。
  2. DeSync 主估计用 softmax 期望 Σ p_i·grid_i（连续），argmax 仅副产物 ——
     模型输出是 0.2s 量化栅格，argmax 天然产生台阶，会伪造"台阶状崩塌"结论。
"""
import os, subprocess, sys
from pathlib import Path

import numpy as np
import torch

SYNC_DIR = Path(__file__).resolve().parents[1] / "third_party" / "Synchformer"
# Synchformer 内部用相对 sys.path（'.'、'model/modules/feat_extractors/visual'），
# 只在它自己的目录下可导入 → 补成绝对路径，并在建模型前 chdir 过去。
for _p in [SYNC_DIR, SYNC_DIR / "model/modules/feat_extractors/visual",
           SYNC_DIR / "model/modules/feat_extractors",
           SYNC_DIR / "model/modules/feat_extractors/train_clip_src", SYNC_DIR / "scripts"]:
    sys.path.insert(0, str(_p))

VFPS, AFPS, IN_SIZE = 25, 16000, 256
DEFAULT_EXP = "24-01-04T16-39-21"


def build_model(exp_name=DEFAULT_EXP, device="cuda:0"):
    from omegaconf import OmegaConf
    from dataset.transforms import make_class_grid
    from scripts.train_utils import get_model, get_transforms

    os.chdir(SYNC_DIR)
    cfg = OmegaConf.load(SYNC_DIR / f"logs/sync_models/{exp_name}/cfg-{exp_name}.yaml")
    cfg.model.params.afeat_extractor.params.ckpt_path = None
    cfg.model.params.vfeat_extractor.params.ckpt_path = None
    cfg.model.params.transformer.target = cfg.model.params.transformer.target.replace(
        ".modules.feature_selector.", ".sync_model.")

    dev = torch.device(device)
    _, model = get_model(cfg, dev)
    ckpt = torch.load(SYNC_DIR / f"logs/sync_models/{exp_name}/{exp_name}.pt",
                      map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    num_cls = cfg.model.params.transformer.params.off_head_cfg.params.out_features
    grid = make_class_grid(-float(cfg.data.max_off_sec), float(cfg.data.max_off_sec), num_cls)
    tf = get_transforms(cfg, ["test"])["test"]
    return dict(model=model, cfg=cfg, tf=tf, grid=grid.numpy(), device=dev,
                crop_len=float(cfg.data.crop_len_sec), max_off=float(cfg.data.max_off_sec))


def reencode_video_only(src, dst, vfps=VFPS, in_size=IN_SIZE):
    """只重编码视频轨（音频另从 wav 读，绝不经此路）。"""
    if Path(dst).exists():
        return str(dst)
    vf = (f"fps={vfps},scale=iw*{in_size}/'min(iw,ih)':ih*{in_size}/'min(iw,ih)',"
          f"crop='trunc(iw/2)'*2:'trunc(ih/2)'*2")
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src),
                    "-an", "-vf", vf, str(dst)], check=True)
    return str(dst)


def load_streams(video, audio_wav=None, workdir="/tmp/desync_work"):
    """视频帧从 mp4 读，音频从无损 wav 读。"""
    import torchaudio, torchvision
    video = str(Path(video).resolve())
    if audio_wav is None:
        audio_wav = str(Path(video).with_suffix(".wav"))
    audio_wav = str(Path(audio_wav).resolve())
    if not Path(audio_wav).exists():
        raise FileNotFoundError(
            f"缺少无损音轨 {audio_wav}。DeSync 必须从 wav 读音频（mp4 的 AAC 轨带 +64ms 偏置）。")
    os.makedirs(workdir, exist_ok=True)
    tmp = Path(workdir) / (Path(video).stem + f"_{VFPS}fps_{IN_SIZE}.mp4")
    reencode_video_only(video, tmp)
    rgb, _, _ = torchvision.io.read_video(str(tmp), pts_unit="sec", output_format="TCHW")
    wav, sr = torchaudio.load(audio_wav)
    wav = wav.mean(0)
    if sr != AFPS:
        wav = torchaudio.functional.resample(wav, sr, AFPS)
    meta = {"video": {"fps": [float(VFPS)]}, "audio": {"framerate": [float(AFPS)]}}
    return rgb, wav, meta


@torch.no_grad()
def desync_curve(M, rgb, wav, meta, stride=1.0, batch_size=8, path="<mem>"):
    """滑窗算 DeSync(t)。返回 dict(t, expect, argmax, p_max, entropy, saturated)。

    自己按窗切片再喂 transform：链首 EqualifyFromRight(clip_max_len_sec=10) 会把整条流
    截到 ≤10s，靠 transform 自带的 v_start_i_sec 滑窗在长视频上必然 assert 失败。
    """
    crop_len, grid, tf = M["crop_len"], M["grid"], M["tf"]
    vwin, awin = int(round(crop_len * VFPS)), int(round(crop_len * AFPS))
    dur = min(len(rgb) / VFPS, len(wav) / AFPS)
    starts = np.arange(0.0, max(dur - crop_len, 0.0) + 1e-6, stride)
    if len(starts) == 0:
        raise ValueError(f"时长 {dur:.2f}s 短于 Synchformer 所需的 {crop_len}s 窗口")

    expect, amax, pmax, ent, kept = [], [], [], [], []
    for i in range(0, len(starts), batch_size):
        items = []
        for t in starts[i:i + batch_size]:
            vs, as_ = int(round(t * VFPS)), int(round(t * AFPS))
            v_seg, a_seg = rgb[vs:vs + vwin], wav[as_:as_ + awin]
            if v_seg.shape[0] < vwin or a_seg.shape[0] < awin:
                continue
            items.append(tf(dict(video=v_seg, audio=a_seg, meta=meta, path=path, split="test",
                                 targets={"v_start_i_sec": 0.0, "offset_sec": 0.0})))
            kept.append(float(t))
        if not items:
            continue
        from scripts.train_utils import prepare_inputs
        batch = torch.utils.data.default_collate(items)
        aud, vid, _ = prepare_inputs(batch, M["device"])
        with torch.autocast("cuda", enabled=M["cfg"].training.use_half_precision):
            _, logits = M["model"](vid, aud)
        p = torch.softmax(logits.float(), dim=-1).cpu().numpy()
        expect.extend((p * grid[None]).sum(1).tolist())
        amax.extend(grid[p.argmax(1)].tolist())
        pmax.extend(p.max(1).tolist())
        ent.extend((-(p * np.log(p + 1e-12)).sum(1)).tolist())

    return dict(t=(np.asarray(kept) + crop_len / 2.0).tolist(),
                expect=expect, argmax=amax, p_max=pmax, entropy=ent,
                saturated=(np.abs(np.asarray(amax)) >= M["max_off"] - 1e-6).tolist(),
                dur=float(dur), win=crop_len, stride=stride)


def drift_scalars(t, absd):
    """漂移标量：D_end / Theil-Sen 斜率 / TTF(τ)。"""
    t, absd = np.asarray(t), np.asarray(absd)
    n = len(absd)
    k = max(1, int(round(0.2 * n)))
    try:
        from scipy.stats import theilslopes
        slope = float(theilslopes(absd, t)[0])
    except Exception:
        slope = float(np.polyfit(t, absd, 1)[0])

    def ttf(tau):
        idx = np.where(absd > tau)[0]
        return float(t[idx[0]]) if len(idx) else None

    return dict(D_end=float(absd[-k:].mean() - absd[:k].mean()), slope=slope,
                mean_abs=float(absd.mean()),
                TTF_0p2=ttf(0.2), TTF_0p3=ttf(0.3), TTF_0p5=ttf(0.5))
