"""로컬 웹 백엔드.

단일 사용자 로컬 앱이므로 작업 상태는 메모리에 둔다. DB 없음.
처리는 워커 스레드에서 돌리고 진행률은 SSE로 흘린다.

    python -m hse.server            # http://127.0.0.1:8756
"""

import json
import os
import queue
import shutil
import tempfile
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .detect import TextDetector
from .pipeline import probe

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK = os.path.join(tempfile.gettempdir(), "hardsub_eraser")
os.makedirs(WORK, exist_ok=True)


@dataclass
class Job:
    id: str
    src: str
    info: dict
    status: str = "ready"           # ready|running|done|error
    stage: str = ""                 # detect|cache|paint
    progress: float = 0.0
    detail: str = ""
    out: str | None = None
    error: str | None = None
    _subs: list = field(default_factory=list, repr=False)

    def public(self):
        d = {k: v for k, v in asdict(self).items() if not k.startswith("_")}
        d["out"] = bool(self.out)
        return d

    def emit(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)
        payload = json.dumps(self.public(), ensure_ascii=False)
        for q in list(self._subs):
            try:
                q.put_nowait(payload)
            except queue.Full:
                pass


JOBS: dict[str, Job] = {}
_detector = None


def detector():
    global _detector
    if _detector is None:
        _detector = TextDetector()
    return _detector


app = FastAPI(title="hardsub_eraser")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])


def _optint(v):
    """빈 값/'auto'는 None으로. None이면 파이프라인이 자동으로 정한다."""
    if v in (None, "", "auto"):
        return None
    return int(v)


def _job(jid) -> Job:
    j = JOBS.get(jid)
    if not j:
        raise HTTPException(404, "job not found")
    return j


# --------------------------------------------------------------- 작업 생성
@app.post("/api/jobs")
async def create_job(file: UploadFile = File(None), path: str = Form(None)):
    """파일 업로드 또는 로컬 경로로 작업을 만든다."""
    jid = uuid.uuid4().hex[:12]
    jdir = os.path.join(WORK, jid)
    os.makedirs(jdir, exist_ok=True)

    if file is not None:
        src = os.path.join(jdir, file.filename or "input.mp4")
        with open(src, "wb") as f:
            shutil.copyfileobj(file.file, f)
    elif path:
        if not os.path.isfile(path):
            raise HTTPException(400, f"파일이 없습니다: {path}")
        src = path
    else:
        raise HTTPException(400, "file 또는 path 중 하나가 필요합니다")

    try:
        info = probe(src)
    except RuntimeError as e:
        raise HTTPException(400, str(e))

    job = Job(id=jid, src=src, info=info)
    JOBS[jid] = job
    return job.public()


@app.get("/api/jobs/{jid}")
def get_job(jid: str):
    return _job(jid).public()


# --------------------------------------------------------------- 프레임 미리보기
@app.get("/api/jobs/{jid}/frame")
def frame(jid: str, n: int = 0, w: int = 720):
    """n번째 프레임을 JPEG로. 브러시 캔버스 배경/타임라인 스크러빙용."""
    j = _job(jid)
    cap = cv2.VideoCapture(j.src)
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, n))
    ok, fr = cap.read()
    cap.release()
    if not ok:
        raise HTTPException(400, "프레임을 읽을 수 없습니다")
    if w and fr.shape[1] > w:
        fr = cv2.resize(fr, (w, int(fr.shape[0] * w / fr.shape[1])))
    ok, buf = cv2.imencode(".jpg", fr, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return Response(buf.tobytes(), media_type="image/jpeg")


# --------------------------------------------------------------- 자막 위치 탐색
@app.get("/api/jobs/{jid}/scan")
def scan(jid: str, samples: int = 24, static_ratio: float = 0.6):
    """자막이 어느 높이에 있는지 찾아 구역을 추천한다.

    단순히 '텍스트가 제일 많이 잡힌 높이'를 고르면 실패한다. 로고나 채널
    워터마크는 매 프레임 나오므로 자막보다 더 자주 잡히기 때문이다(실제로
    세로 숏드라마에서 좌상단 워터마크가 16회, 자막이 14회로 뒤집혔다).

    구분 기준은 시간에 따른 변화다.
      워터마크 — 매 프레임 같은 자리에 같은 크기로 나온다
      자막     — 글자가 바뀌면서 폭과 위치가 계속 달라진다

    그래서 박스 기하를 양자화해 세고, 표본의 static_ratio 이상에서 똑같이
    반복되는 것은 고정 오버레이로 보고 자막 히스토그램에서 뺀다.
    """
    j = _job(jid)
    n, h, w = j.info["n"], j.info["h"], j.info["w"]
    cap = cv2.VideoCapture(j.src)
    det = detector()

    shots = []
    for k in range(samples):
        idx = int(n * (k + 0.5) / samples)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, fr = cap.read()
        if ok:
            shots.append((idx, det.boxes(fr)))
    cap.release()

    # 격자 양자화로는 감지 지터를 못 넘는다(1080p에서 1% = 11px). IoU로 묶는다.
    def iou(a, b):
        ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
        iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
        inter = ix * iy
        ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return inter / ua if ua > 0 else 0.0

    clusters = []
    for idx, boxes in shots:
        for b in boxes:
            for c in clusters:
                if iou(c["rep"], b) > 0.5:
                    c["items"].append((idx, b))
                    break
            else:
                clusters.append({"rep": b, "items": [(idx, b)]})

    thresh = max(2, int(len(shots) * static_ratio))
    for c in clusters:
        frames = {i for i, _ in c["items"]}
        ws = np.array([bx[2] - bx[0] for _, bx in c["items"]], dtype=float)
        cxs = np.array([(bx[0] + bx[2]) / 2 for _, bx in c["items"]], dtype=float)
        # 워터마크는 폭도 가로 위치도 거의 안 변한다. 자막은 글자 수에 따라 폭이 변한다.
        rigid = (ws.std() / max(ws.mean(), 1.0) < 0.06) and (cxs.std() < 0.01 * w)
        c["static"] = len(frames) >= thresh and rigid

    hist = [0] * 20
    subs, marks = [], []
    for c in clusters:
        for idx, b in c["items"]:
            item = {"n": idx, "box": list(b)}
            if c["static"]:
                marks.append(item)
                continue
            subs.append(item)
            for i in range(int(b[1] / h * 20), min(20, int(b[3] / h * 20) + 1)):
                hist[i] += 1

    region = None
    if any(hist):
        peak = max(range(20), key=lambda i: hist[i])
        lo = hi = peak
        while lo > 0 and hist[lo - 1] >= hist[peak] * 0.3:
            lo -= 1
        while hi < 19 and hist[hi + 1] >= hist[peak] * 0.3:
            hi += 1
        region = [round(max(0, lo - 1) / 20, 3), round(min(20, hi + 2) / 20, 3)]

    return {"hist": hist, "boxes": subs[:200], "watermarks": marks[:60],
            "suggested_region": region, "width": w, "height": h,
            "samples": len(shots),
            "static_count": sum(1 for c in clusters if c["static"])}


# --------------------------------------------------------------- 처리 실행
@app.post("/api/jobs/{jid}/run")
def start(jid: str, params: dict):
    j = _job(jid)
    if j.status == "running":
        raise HTTPException(409, "이미 처리 중입니다")

    out = os.path.join(WORK, jid, "output.mp4")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    def worker():
        from .pipeline import run
        try:
            j.emit(status="running", stage="detect", progress=0.0, error=None, out=None)

            def on_progress(stage, done, total, detail=""):
                j.emit(stage=stage, progress=(done / total if total else 0.0), detail=detail)

            run(j.src, out,
                region=params.get("region", "0.70,1.0"),
                det_stride=int(params.get("det_stride", 1)),
                # None을 넘기면 이 기기에 맞춰 자동으로 정한다
                group_size=_optint(params.get("group_size")),
                n_refs=_optint(params.get("n_refs")),
                ref_span=int(params.get("ref_span", 600)),
                feather=int(params.get("feather", 4)),
                mask_pad=int(params.get("mask_pad", 3)),
                crf=int(params.get("crf", 18)),
                backend=params.get("backend", "auto"),
                device=params.get("device", "auto"),
                on_progress=on_progress,
                quiet=True)
            j.emit(status="done", stage="", progress=1.0, out=out, detail="")
        except Exception as e:  # noqa: BLE001 — 워커 스레드라 모두 잡아 상태로 옮긴다
            j.emit(status="error", error=f"{type(e).__name__}: {e}")

    threading.Thread(target=worker, daemon=True).start()
    return j.public()


@app.get("/api/jobs/{jid}/stream")
def stream(jid: str):
    """진행률 SSE."""
    j = _job(jid)
    q: queue.Queue = queue.Queue(maxsize=64)
    j._subs.append(q)

    def gen():
        try:
            yield f"data: {json.dumps(j.public(), ensure_ascii=False)}\n\n"
            last = time.time()
            while True:
                try:
                    yield f"data: {q.get(timeout=15)}\n\n"
                except queue.Empty:
                    yield ": keepalive\n\n"
                if j.status in ("done", "error") and q.empty():
                    if time.time() - last > 1:
                        break
                last = time.time()
        finally:
            if q in j._subs:
                j._subs.remove(q)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


@app.get("/api/jobs/{jid}/source")
def source(jid: str):
    """원본 영상. 비포/애프터 비교에서 왼쪽에 쓴다."""
    j = _job(jid)
    return FileResponse(j.src, media_type="video/mp4")


@app.get("/api/jobs/{jid}/result")
def result(jid: str):
    j = _job(jid)
    if not j.out or not os.path.isfile(j.out):
        raise HTTPException(404, "결과가 아직 없습니다")
    return FileResponse(j.out, media_type="video/mp4",
                        filename=os.path.splitext(os.path.basename(j.src))[0] + "_clean.mp4")


@app.get("/api/backends")
def backends():
    """설치된 실행 provider와 자동 선택 결과."""
    try:
        from .engine_onnx import available_backends, describe, pick_providers
        av = available_backends()
        return {"onnx": av, "active": describe(pick_providers(None)) if av else None}
    except ImportError as e:
        return {"onnx": [], "active": None, "error": str(e)}


_ui = os.path.join(ROOT, "web")
if os.path.isdir(_ui):
    app.mount("/", StaticFiles(directory=_ui, html=True), name="ui")


def main():
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8756, log_level="warning")


if __name__ == "__main__":
    main()
