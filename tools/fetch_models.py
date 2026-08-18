"""ONNX 모델을 GitHub Releases에서 받아 models/onnx/ 에 넣는다.

모델은 저장소에 두지 않는다. Git LFS는 무료 한도가 스토리지·대역폭 각 1GB라
배포가 시작되면 과금되지만, Releases 첨부는 무료이고 대역폭 제한도 없다.

    python tools/fetch_models.py

직접 만들고 싶으면 (torch 필요, VSR에서 sttn.pth 확보 후):
    python tools/export_onnx.py
"""

import argparse
import io
import json
import os
import sys
import urllib.request
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.environ.get("HSE_REPO", "Cench-k/hardsub-eraser")
ASSET = "onnx-models.zip"
NEEDED = ("encoder.onnx", "transformer.onnx", "decoder.onnx")


def latest_asset_url(repo, asset):
    api = f"https://api.github.com/repos/{repo}/releases/latest"
    req = urllib.request.Request(api, headers={"Accept": "application/vnd.github+json",
                                               "User-Agent": "hardsub-eraser"})
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.load(r)
    for a in data.get("assets", []):
        if a["name"] == asset:
            return a["browser_download_url"], data.get("tag_name", "?")
    raise RuntimeError(f"릴리스에 {asset} 이 없습니다 (tag={data.get('tag_name')})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=REPO, help="owner/name")
    ap.add_argument("--url", default=None, help="zip을 직접 지정")
    a = ap.parse_args()

    out = os.path.join(ROOT, "models", "onnx")
    have = [f for f in NEEDED if os.path.isfile(os.path.join(out, f))]
    if len(have) == len(NEEDED):
        print(f"이미 있습니다: {out}")
        return

    url, tag = (a.url, "직접지정") if a.url else latest_asset_url(a.repo, ASSET)
    print(f"내려받는 중 ({tag}): {url}")

    buf = io.BytesIO()
    with urllib.request.urlopen(url, timeout=60) as r:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while chunk := r.read(1 << 20):
            buf.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done/1e6:5.1f} / {total/1e6:.1f} MB", end="")
        print()

    os.makedirs(out, exist_ok=True)
    buf.seek(0)
    with zipfile.ZipFile(buf) as z:
        for n in z.namelist():
            base = os.path.basename(n)
            if base in NEEDED:
                with z.open(n) as src, open(os.path.join(out, base), "wb") as dst:
                    dst.write(src.read())
                print(f"  {base}")

    missing = [f for f in NEEDED if not os.path.isfile(os.path.join(out, f))]
    if missing:
        sys.exit(f"[!] 빠진 파일: {missing}")
    print(f"완료 -> {out}")


if __name__ == "__main__":
    main()
