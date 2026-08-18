"""배포본 스모크 테스트. 반드시 배포본의 임베디드 파이썬으로 실행한다.

    dist\\hardsub_eraser\\runtime\\python.exe tools\\verify_release.py

개발 환경이 없는 상태에서 실제로 도는지 확인한다. 'onnxruntime이 조용히
CPU로 떨어지는' 종류의 사고를 빌드 시점에 잡는 게 목적이다.
"""

import os
import subprocess
import sys
import tempfile

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
fails = []


def check(name, fn):
    try:
        msg = fn()
        print(f"  [OK]   {name}" + (f"  {msg}" if msg else ""))
    except Exception as e:  # noqa: BLE001
        print(f"  [FAIL] {name}  {type(e).__name__}: {str(e)[:200]}")
        fails.append(name)


def c_imports():
    import cv2, fastapi, numpy, rapidocr_onnxruntime, uvicorn  # noqa: F401
    return f"cv2 {cv2.__version__}, numpy {numpy.__version__}"


def c_providers():
    import onnxruntime as ort
    av = ort.get_available_providers()
    gpu = [p for p in av if p != "CPUExecutionProvider"]
    if not gpu:
        raise RuntimeError(f"GPU provider 없음 — CPU로만 동작한다: {av}")
    return f"{ort.__version__}  {av}"


def c_models():
    import onnxruntime as ort
    d = os.path.join(APP, "models", "onnx")
    shapes = []
    for n in ("encoder", "transformer", "decoder"):
        p = os.path.join(d, n + ".onnx")
        if not os.path.isfile(p):
            raise FileNotFoundError(p)
        s = ort.InferenceSession(p, providers=["CPUExecutionProvider"])
        shapes.append(f"{n}{s.get_inputs()[0].shape}")
    return " ".join(shapes)


def c_hse():
    sys.path.insert(0, APP)
    import hse.autotune, hse.engine_onnx, hse.pipeline, hse.server  # noqa: F401
    return "hse 로드"


def c_ffmpeg():
    for exe in ("ffmpeg", "ffprobe"):
        r = subprocess.run([exe, "-version"], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"{exe} 실행 실패")
    return subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True
                          ).stdout.splitlines()[0][:60]


def c_batfile():
    """실행.bat이 cmd.exe가 읽을 수 있는 형태인지.

    cmd는 .bat을 OEM 코드페이지(한국어 Windows는 949)로 읽는다. UTF-8 한글이
    들어가면 깨져서 쓰레기 명령으로 실행되고, BOM이 있으면 첫 줄을 못 읽는다.
    실제로 v0.1.0을 이 문제로 다시 냈다.
    """
    hits = [f for f in os.listdir(APP) if f.lower().endswith(".bat")]
    if not hits:
        raise FileNotFoundError("bat 파일이 없다")
    for name in hits:
        raw = open(os.path.join(APP, name), "rb").read()
        if raw[:3] == b"\xef\xbb\xbf":
            raise ValueError(f"{name}: UTF-8 BOM이 있다")
        bad = [b for b in raw if b > 127]
        if bad:
            raise ValueError(f"{name}: 비ASCII 바이트 {len(bad)}개 (cmd가 깨뜨린다)")
        if raw.count(b"\n") != raw.count(b"\r\n"):
            raise ValueError(f"{name}: LF 단독 줄바꿈 (CRLF여야 한다)")
    return " ".join(hits)


def c_endtoend():
    """자막을 태운 짧은 영상을 만들어 실제로 지워 본다."""
    sys.path.insert(0, APP)
    from hse.pipeline import probe, run

    tmp = tempfile.mkdtemp(prefix="hse_verify_")
    src = os.path.join(tmp, "in.mp4")
    dst = os.path.join(tmp, "out.mp4")

    vf = ("drawtext=text='TEST SUBTITLE':fontcolor=white:fontsize=28:borderw=2:"
          "bordercolor=black:x=(w-text_w)/2:y=h-70:enable='lt(t,2)',"
          "drawtext=text='SECOND LINE HERE':fontcolor=white:fontsize=28:borderw=2:"
          "bordercolor=black:x=(w-text_w)/2:y=h-70:enable='gte(t,2)'")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
                    "-i", "testsrc2=size=640x360:rate=15:duration=4",
                    "-vf", vf, "-pix_fmt", "yuv420p", src], check=True)

    run(src, dst, region="0.72,1.0", quiet=True)
    if not os.path.isfile(dst):
        raise RuntimeError("출력 파일이 없다")
    a, b = probe(src), probe(dst)
    if abs(a["n"] - b["n"]) > 2:
        raise RuntimeError(f"프레임 수 불일치 {a['n']} -> {b['n']}")
    return f"{a['w']}x{a['h']} {b['n']}프레임 처리"


if __name__ == "__main__":
    print(f"배포본 검증: {APP}")
    print(f"  python {sys.version.split()[0]}  ({sys.executable})\n")
    check("의존성 import", c_imports)
    check("onnxruntime provider", c_providers)
    check("ONNX 모델", c_models)
    check("hse 패키지", c_hse)
    check("ffmpeg / ffprobe", c_ffmpeg)
    check("실행 배치파일", c_batfile)
    check("전체 파이프라인", c_endtoend)

    print()
    if fails:
        print(f"실패 {len(fails)}건: {', '.join(fails)}")
        sys.exit(1)
    print("전부 통과 — 배포 가능")
