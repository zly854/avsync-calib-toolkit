"""RealRef 参照集构建：从 Wikimedia Commons 下载真实 AV 视频，裁 60s，产出测量对。

用途（真实同步参照集：噪声底对照 + 校准母体）：
当前"真实视频噪声底 0.222s"建立在 n=1 上，两篇论文都站不住。本脚本构建
n≈14 的多样化真实视频母体（语音/音乐/撞击/环境四类，与 LongAV-48 分层同构）。

选择纪律（写论文 Limitations 用）：
- 全部 Commons 公开许可，逐条记录 license/attribution 进 refset_meta.json；
- 排除无声电影配乐版（同步无定义）；1945 新闻片鼓独奏保留但打 vintage 标记
  （胶片转制同步质量存疑，筛查阶段用 DeSync 读数决定去留）；
- 每条裁**中间** 60s（避开片头字幕/片尾）；不足 65s 的取全长减首尾 2s；
- 输出 refXX.mp4（25fps h264，仅供帧读取）+ refXX.wav（16k mono PCM 测量通路），
  命名 ref00..refNN 按类别排序。

用法：python eval/drift/build_refset.py --out results/RealRef
"""
import argparse
import json
import subprocess
import time
import urllib.parse
import urllib.request
from pathlib import Path

UA = {"User-Agent": "research-calib/0.1 (academic AV-sync metric calibration)"}

# (类别, 搜索词, 标题前缀——用于在搜索结果中锁定精确标题)
CANDIDATES = [
    ("speech",  "interview filetype:video",            "File:Bernard Mabille - Interview"),
    ("speech",  "interview filetype:video",            "File:Interview on extreme weather"),
    ("speech",  "speech lecture filetype:video",       "File:Climate Resilience through Knowledge"),
    ("speech",  "speech lecture filetype:video",       "File:Berenice Cort"),
    ("music",   "violin performance filetype:video",   "File:MHVC-KyokoYonemoto"),
    ("music",   "violin performance filetype:video",   "File:La feria chilpancing"),
    ("music",   "piano recital filetype:video",        "File:Eri & Mari Yoshizawa"),
    ("music",   "drumming filetype:video",             "File:Drumming Basics"),
    ("impact",  "drumming filetype:video",             "File:Luk"),
    ("impact",  "blacksmith forging filetype:video",   "File:RhofLhSchmied"),
    ("impact",  "drumming filetype:video",             "File:Andy Russell Drum Solo"),
    ("impact",  "basketball dribbling filetype:video", "File:Basketball-Basic Types"),
    ("ambient", "rain sound filetype:video",           "File:WV0065"),
    ("ambient", "ocean waves beach filetype:video",    "File:Ocean Shores - Waves"),
]
VINTAGE = {"File:Andy Russell Drum Solo"}


def api(**kw):
    kw.update(action="query", format="json")
    url = "https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode(kw)
    req = urllib.request.Request(url, headers=UA)
    return json.load(urllib.request.urlopen(req, timeout=30))


def resolve(search, prefix):
    r = api(list="search", srsearch=search, srnamespace=6, srlimit=20)
    titles = [h["title"] for h in r["query"]["search"] if h["title"].startswith(prefix)]
    if not titles:
        return None
    time.sleep(3)
    r2 = api(titles=titles[0], prop="videoinfo",
             viprop="url|size|mime|duration|extmetadata")
    page = next(iter(r2["query"]["pages"].values()))
    vi = page["videoinfo"][0]
    md = vi.get("extmetadata", {})
    return dict(title=page["title"], url=vi["url"], duration=vi.get("duration"),
                size=vi.get("size"), mime=vi.get("mime"),
                license=(md.get("LicenseShortName") or {}).get("value"),
                artist=(md.get("Artist") or {}).get("value", "")[:200])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--clip", type=float, default=60.0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    meta = []
    idx = 0
    for cat, search, prefix in CANDIDATES:
        name = f"ref{idx:02d}"
        mp4 = out / f"{name}.mp4"
        if mp4.exists():
            print(f"[{name}] exists, skip");  idx += 1;  continue
        try:
            info = resolve(search, prefix)
            time.sleep(3)
        except Exception as e:
            print(f"[{name}] resolve FAIL {prefix}: {e}");  idx += 1;  continue
        if info is None:
            print(f"[{name}] no match for {prefix}");  idx += 1;  continue

        raw = out / ("raw_" + name + Path(urllib.parse.urlparse(info["url"]).path).suffix)
        print(f"[{name}] {cat} {info['title'][:60]} dur={info['duration']}s "
              f"{(info['size'] or 0)//2**20}MB lic={info['license']}")
        if not raw.exists():
            subprocess.run(["curl", "-sL", "--retry", "3", "-o", str(raw), info["url"]],
                           check=True, timeout=1800)

        dur = float(info["duration"] or 0)
        clip = min(args.clip, max(dur - 4, 10))
        start = max((dur - clip) / 2, 2)
        # 视频 25fps h264（只供帧读取）；音频无损 wav sidecar（测量通路）
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                        "-ss", f"{start:.2f}", "-t", f"{clip:.2f}", "-i", str(raw),
                        "-an", "-vf", "fps=25", "-c:v", "libx264", "-crf", "20", str(mp4)],
                       check=True, timeout=1800)
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                        "-ss", f"{start:.2f}", "-t", f"{clip:.2f}", "-i", str(raw),
                        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
                        str(out / f"{name}.wav")], check=True, timeout=1800)
        raw.unlink()
        meta.append(dict(name=name, category=cat, clip_seconds=clip, clip_start=start,
                         vintage=any(info["title"].startswith(v) for v in VINTAGE), **info))
        with open(out / "refset_meta.json", "w") as f:
            json.dump(meta, f, ensure_ascii=False, indent=1)
        idx += 1

    print(f"done: {len(meta)} clips -> {out}")


if __name__ == "__main__":
    main()
