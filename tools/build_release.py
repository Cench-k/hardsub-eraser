"""배포용 포터블 ZIP을 만든다.

받는 사람이 Python도 ffmpeg도 설치할 필요 없이 압축만 풀고 실행하게 하는 게 목표다.

구성:
  dist/hardsub_eraser/
    runtime/          임베디드 Python + 의존성 + ffmpeg.exe, ffprobe.exe
    hse/  web/        우리 코드
    models/onnx/      STTN 그래프 3개 (67MB) — 작아서 그냥 동봉한다
    실행.bat

torch는 넣지 않는다. onnxruntime-directml 하나로 NVIDIA/AMD/Intel/CPU를 다 커버한다.

    python tools/build_release.py
"""

import argparse
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY_VER = "3.11.9"
PY_URL = f"https://www.python.org/ftp/python/{PY_VER}/python-{PY_VER}-embed-amd64.zip"
PIP_URL = "https://bootstrap.pypa.io/get-pip.py"
# essentials 빌드도 libx264(GPL)를 포함한다. 우리가 GPLv3이므로 동봉 가능.
FFMPEG_URL = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"

# onnxruntime-directml은 여기 넣지 않는다. rapidocr가 CPU 전용 onnxruntime을
# 의존성으로 끌고 오므로, 먼저 이것들을 깔고 → onnxruntime 제거 → directml 설치
# 순서를 지켜야 한다. 순서를 어기면 uninstall이 공유 디렉터리를 지워놓고
# pip 메타데이터는 남아 재설치가 no-op이 되어 'unknown location' 에러가 난다.
PKGS = [
    # headless: 서버 앱이라 GUI 백엔드가 필요 없고 용량이 작다. 버전은 개발환경과 맞춘다.
    "opencv-python-headless==5.0.0.93",
    "numpy==2.4.6",
    "rapidocr-onnxruntime==1.4.4",
    "tqdm==4.70.0",
    "fastapi==0.141.1",
    "uvicorn==0.52.3",
    "python-multipart",
]
ORT_PKG = "onnxruntime-directml==1.24.4"

START_BAT = """@echo off
chcp 65001 >nul
cd /d "%~dp0"
set "PATH=%~dp0runtime;%PATH%"
echo 하드섭 지우개를 시작합니다...
start "" http://127.0.0.1:8756
runtime\\python.exe -m hse.server
pause
"""


def fetch(url, dst):
    if os.path.exists(dst):
        print(f"  (있음) {os.path.basename(dst)}")
        return dst
    print(f"  받는 중 {url}")
    with urllib.request.urlopen(url) as r, open(dst, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r    {done/1e6:6.1f} / {total/1e6:.1f} MB", end="")
        print()
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "dist"))
    ap.add_argument("--cache", default=os.path.join(ROOT, "dist", "_cache"))
    ap.add_argument("--skip-zip", action="store_true")
    a = ap.parse_args()

    app = os.path.join(a.out, "hardsub_eraser")
    runtime = os.path.join(app, "runtime")
    os.makedirs(a.cache, exist_ok=True)
    if os.path.exists(app):
        shutil.rmtree(app)
    os.makedirs(runtime)

    onnx_dir = os.path.join(ROOT, "models", "onnx")
    need = [os.path.join(onnx_dir, f + ".onnx") for f in ("encoder", "transformer", "decoder")]
    missing = [p for p in need if not os.path.isfile(p)]
    if missing:
        sys.exit("ONNX 그래프가 없습니다. 먼저: python tools/export_onnx.py")

    # 1) 임베디드 파이썬
    print("[1/6] 임베디드 Python")
    z = fetch(PY_URL, os.path.join(a.cache, f"python-{PY_VER}-embed.zip"))
    with zipfile.ZipFile(z) as f:
        f.extractall(runtime)

    # 임베디드 배포판은 site 를 꺼둬서 pip 설치분을 못 찾는다. 켜준다.
    pth = next(p for p in os.listdir(runtime) if p.endswith("._pth"))
    pth_path = os.path.join(runtime, pth)
    lines = open(pth_path, encoding="utf-8").read().splitlines()
    out = []
    for ln in lines:
        out.append("import site" if ln.strip() == "#import site" else ln)
    if "import site" not in out:
        out.append("import site")
    out.append("Lib\\site-packages")
    out.append("..")  # hse 패키지를 찾도록 앱 루트 추가
    open(pth_path, "w", encoding="utf-8").write("\n".join(out) + "\n")

    # 2) pip
    print("[2/6] pip")
    gp = fetch(PIP_URL, os.path.join(a.cache, "get-pip.py"))
    py = os.path.join(runtime, "python.exe")
    subprocess.run([py, gp, "--no-warn-script-location", "-q"], check=True)

    # 3) 의존성
    print("[3/6] 의존성 설치 (수백 MB, 시간이 걸립니다)")
    subprocess.run([py, "-m", "pip", "install", "--no-warn-script-location", "-q", *PKGS],
                   check=True)
    # rapidocr가 끌고 온 CPU 전용 onnxruntime을 걷어내고 DirectML판으로 교체한다.
    subprocess.run([py, "-m", "pip", "uninstall", "-y", "-q", "onnxruntime"], check=False)
    subprocess.run([py, "-m", "pip", "install", "--no-warn-script-location", "-q",
                    "--force-reinstall", "--no-deps", ORT_PKG], check=True)

    # 여기서 반드시 확인한다. 조용히 CPU로 떨어지면 사용자 PC에서 10배 느려진다.
    chk = subprocess.run(
        [py, "-c", "import onnxruntime as o; print(';'.join(o.get_available_providers()))"],
        capture_output=True, text=True)
    if "DmlExecutionProvider" not in chk.stdout:
        sys.exit(f"[!] DirectML provider 없음. 런타임이 잘못 깔렸습니다.\n"
                 f"    stdout: {chk.stdout.strip()}\n    stderr: {chk.stderr.strip()[:400]}")
    print(f"  provider 확인: {chk.stdout.strip()}")

    # 4) ffmpeg
    print("[4/6] ffmpeg")
    z = fetch(FFMPEG_URL, os.path.join(a.cache, "ffmpeg.zip"))
    with zipfile.ZipFile(z) as f:
        for n in f.namelist():
            base = os.path.basename(n)
            if base in ("ffmpeg.exe", "ffprobe.exe"):
                with f.open(n) as src, open(os.path.join(runtime, base), "wb") as dst:
                    shutil.copyfileobj(src, dst)

    # 5) 우리 코드
    print("[5/6] 앱 파일")
    shutil.copytree(os.path.join(ROOT, "hse"), os.path.join(app, "hse"),
                    ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(os.path.join(ROOT, "web"), os.path.join(app, "web"))
    os.makedirs(os.path.join(app, "models", "onnx"))
    for p in need:
        shutil.copy2(p, os.path.join(app, "models", "onnx"))
    for f in ("LICENSE", "THIRD-PARTY-NOTICES.md", "README.md"):
        shutil.copy2(os.path.join(ROOT, f), app)
    open(os.path.join(app, "실행.bat"), "w", encoding="utf-8-sig").write(START_BAT)

    # 배포판에는 torch가 없다. 있으면 import 에러만 나므로 아예 뺀다.
    shutil.rmtree(os.path.join(app, "hse", "sttn"), ignore_errors=True)
    for f in ("engine.py",):
        p = os.path.join(app, "hse", f)
        if os.path.exists(p):
            os.remove(p)

    os.makedirs(os.path.join(app, "tools"), exist_ok=True)
    shutil.copy2(os.path.join(ROOT, "tools", "verify_release.py"),
                 os.path.join(app, "tools"))

    size = sum(os.path.getsize(os.path.join(dp, f))
               for dp, _, fs in os.walk(app) for f in fs)
    print(f"[6/6] 완성: {app}  ({size/1e6:.0f} MB)")

    # 배포본 자신의 파이썬으로 스모크 테스트. 개발환경이 섞이지 않도록 env를 비운다.
    print("\n검증 (배포본 런타임으로 실행)")
    env = {k: v for k, v in os.environ.items()
           if k.upper() not in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV")}
    env["PATH"] = runtime + os.pathsep + env.get("PATH", "")
    r = subprocess.run([py, os.path.join(app, "tools", "verify_release.py")], env=env)
    if r.returncode != 0:
        sys.exit("[!] 검증 실패 — 배포하지 말 것")

    if not a.skip_zip:
        zip_path = os.path.join(a.out, "hardsub_eraser-portable-win64")
        print("  압축 중...")
        shutil.make_archive(zip_path, "zip", a.out, "hardsub_eraser")
        print(f"  -> {zip_path}.zip ({os.path.getsize(zip_path + '.zip')/1e6:.0f} MB)")


if __name__ == "__main__":
    main()
