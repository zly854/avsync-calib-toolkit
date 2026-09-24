"""Reconstruct the fixed reference set from its licensed-source manifest.

No discovery/search is performed. Start times retain the two-decimal rounding
used in the original ffmpeg commands. Source bytes may change upstream; the
historical release did not record source checksums, so byte identity is not claimed.
"""
import argparse
import json
import re
import shlex
import subprocess
from pathlib import Path
from urllib.parse import urlparse


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest', type=Path, default=Path(__file__).resolve().parents[1]/'data/refset_meta.json')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--dry-run', action='store_true', help='Print exact commands without network or writes')
    args = ap.parse_args()
    records = json.loads(args.manifest.read_text())
    names = set()
    for row in records:
        name = row['name']; url = urlparse(row['url'])
        if not re.fullmatch(r'ref\d{2}', name) or name in names:
            ap.error(f'Invalid or duplicate name: {name}')
        names.add(name)
        if url.scheme != 'https' or url.hostname != 'upload.wikimedia.org':
            ap.error(f'Unexpected source URL for {name}')
        if not row.get('license') or not row.get('artist') or row['clip_start'] < 0 or row['clip_seconds'] <= 0:
            ap.error(f'Missing provenance or invalid time span: {name}')
    if not args.dry_run: args.out.mkdir(parents=True, exist_ok=True)
    for row in records:
        name=row['name']; video=args.out/f'{name}.mp4'; audio=args.out/f'{name}.wav'
        if video.is_file() and video.stat().st_size and audio.is_file() and audio.stat().st_size:
            print(f'{name}: both streams exist; skipping'); continue
        raw=args.out/f"raw_{name}{Path(urlparse(row['url']).path).suffix}"
        temp_video=args.out/f'{name}.partial.mp4';temp_audio=args.out/f'{name}.partial.wav'
        common=['ffmpeg','-hide_banner','-loglevel','error','-y','-ss',f"{row['clip_start']:.2f}",'-t',f"{row['clip_seconds']:.2f}",'-i',str(raw)]
        commands=[['curl','--fail','--location','--retry','3','--output',str(raw),row['url']],
            common+['-an','-vf','fps=25','-c:v','libx264','-crf','20',str(temp_video)],
            common+['-vn','-acodec','pcm_s16le','-ar','16000','-ac','1',str(temp_audio)]]
        for command in commands:
            print(shlex.join(command))
            if not args.dry_run: subprocess.run(command,check=True,timeout=1800)
        if not args.dry_run:
            temp_video.replace(video);temp_audio.replace(audio);raw.unlink()
    if not args.dry_run:
        (args.out/'refset_meta.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n')
    print(f'{len(records)} manifest records processed')


if __name__=='__main__':main()
