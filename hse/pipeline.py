"""하드섭 제거 파이프라인.

3패스 구조:
  1) 감지    — 자막 박스를 프레임별로 수집
  2) 밴드 캐시 — 처리 밴드를 432x240으로 줄여 memmap에 저장 (프레임당 311KB)
  3) 처리    — 그룹 단위 STTN + 원본 해상도 합성 + 리먹스

2패스가 필요한 이유: 참조 프레임을 영상 전체 어디서든 뽑으려면 임의 접근이 필요한데,
영상 디코딩은 순차 접근이다. 저해상도 밴드만 캐시하면 임의 접근이 싸진다.
"""

import os
import subprocess
import tempfile
import time

import cv2
import numpy as np
from tqdm import tqdm

from .common import MODEL_H, MODEL_W, composite, to_bgr
from .detect import TextDetector, boxes_to_mask

# 모델 경로는 반드시 절대 경로로 잡는다. 상대 경로로 두면 작업 디렉터리가
# 앱 폴더가 아닐 때 ONNX를 못 찾고 조용히 torch 백엔드로 넘어가는데,
# 배포판에는 torch가 없어서 그대로 실패한다.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_ONNX_DIR = os.path.join(_ROOT, "models", "onnx")
DEFAULT_TORCH_MODEL = os.path.join(_ROOT, "models", "sttn", "sttn.pth")


class Aborted(RuntimeError):
    """사용자가 취소했을 때. 오류와 구분하려고 따로 둔다."""


class Progress:
    """터미널 진행바와 서버 콜백을 동시에 먹인다.

    콜백은 매 프레임 부르지 않는다. SSE로 초당 수십 번 밀어봐야 의미가 없고
    직렬화 비용만 든다.

    취소 확인도 여기서 한다. 모든 단계가 이 update()를 지나가므로
    한 곳만 봐도 어느 단계에서든 멈출 수 있다.
    """

    def __init__(self, stage, total, desc, on_progress=None, quiet=False, every=8,
                 should_abort=None):
        self.stage, self.total, self.on, self.every = stage, total, on_progress, every
        self.bar = None if quiet else tqdm(total=total, desc=desc, unit="f")
        self.should_abort = should_abort
        self.n = 0

    def update(self, k=1, detail=""):
        self.n += k
        if self.bar:
            self.bar.update(k)
        if self.n % self.every == 0:
            if self.should_abort and self.should_abort():
                raise Aborted("사용자가 취소했습니다")
        if self.on and (self.n % self.every == 0 or self.n >= self.total):
            self.on(self.stage, self.n, self.total, detail)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        if self.bar:
            self.bar.close()


def probe(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"영상을 열 수 없습니다: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # `or 25.0` 로는 부족하다. 이미지를 열면 fps 가 -1 로 나오는데 -1 은 참이라
    # 그대로 통과해서 ffmpeg 에 `-r -1.0` 이 넘어가고 인코더가 죽는다.
    # 메타데이터가 깨진 영상에서도 같은 일이 난다.
    if not fps or fps <= 0 or fps > 1000:
        fps = 25.0
    info = dict(
        w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        fps=fps,
        n=max(1, n),
        is_image=(n <= 1),
    )
    cap.release()

    info["size"] = os.path.getsize(path)
    dur = info["n"] / info["fps"] if info["fps"] else 0
    info["duration"] = dur
    info["bitrate"] = int(info["size"] * 8 / dur) if dur > 0 else 0
    info.update(suggest_crf(info))
    return info


def suggest_crf(info):
    """원본 압축 정도를 보고 출력 CRF를 추천한다.

    CRF는 파일에 저장되는 값이 아니라 인코딩 설정이라 되읽을 수 없다. 대신
    픽셀당 비트수(bpp)로 원본이 얼마나 눌려 있는지를 가늠한다.

        bpp = 비트레이트 / (가로 x 세로 x fps)

    원본이 이미 강하게 압축돼 있으면 CRF를 낮게 줘봐야 없는 화질이 생기지 않고
    용량만 몇 배로 불어난다(실측: 47MB 원본이 CRF 18에서 131MB가 됐다).
    """
    px = info["w"] * info["h"] * info["fps"]
    bpp = info["bitrate"] / px if px else 0
    if bpp <= 0:
        return {"bpp": 0, "crf": 18, "quality": "알 수 없음"}
    if bpp < 0.035:
        crf, q = 23, "강하게 압축된 원본"
    elif bpp < 0.060:
        crf, q = 21, "보통으로 압축된 원본"
    elif bpp < 0.100:
        crf, q = 19, "고화질 원본"
    else:
        crf, q = 18, "매우 고화질 원본"
    return {"bpp": round(bpp, 4), "crf": crf, "quality": q}


def parse_region(spec, h):
    """'0.7,1.0' (비율) 또는 '756,1080' (픽셀) 형식."""
    a, b = [float(x) for x in spec.split(",")]
    y0 = int(round(a * h)) if a <= 1.0 else int(a)
    y1 = int(round(b * h)) if b <= 1.0 else int(b)
    return max(0, min(y0, h)), max(0, min(y1, h))


class FFmpegWriter:
    """원본 오디오를 그대로 복사하면서 프레임을 파이프로 넘긴다."""

    def __init__(self, out_path, w, h, fps, audio_from=None, crf=18, preset="medium"):
        cmd = ["ffmpeg", "-y", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", f"{fps}", "-i", "-"]
        if audio_from:
            cmd += ["-i", audio_from]
        cmd += ["-map", "0:v:0"]
        if audio_from:
            cmd += ["-map", "1:a:0?", "-c:a", "copy"]
        cmd += ["-c:v", "libx264", "-crf", str(crf), "-preset", preset,
                "-pix_fmt", "yuv420p", "-shortest", out_path]
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, frame):
        self.p.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self):
        self.p.stdin.close()
        return self.p.wait()


def detect_pass(path, y0, y1, stride, detector, on_progress=None, quiet=False,
                should_abort=None):
    """자막 박스를 전체 프레임 좌표로 수집.

    stride > 1 이면 N프레임마다만 감지한다. 감지는 전체 시간의 20% 정도를
    차지하므로 여기서 아끼는 게 크다.

    건너뛴 프레임은 **앞뒤 샘플의 합집합**으로 채운다. 직전 값만 물려주면
    자막이 새로 뜨는 순간을 최대 N-1 프레임 놓쳐서 글자가 남는다. 합집합이면
    마스크가 조금 넓어질 뿐이라 안전하다.
    """
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sampled, i = {}, 0
    with Progress("detect", total, "1/3 자막 감지", on_progress, quiet,
                  should_abort=should_abort) as p:
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            fr = to_bgr(fr)
            if i % stride == 0:
                sampled[i] = [(x1, ya + y0, x2, yb + y0)
                              for x1, ya, x2, yb in detector.boxes(fr[y0:y1])]
            i += 1
            p.update()
    cap.release()

    n = i
    if stride <= 1:
        return [sampled.get(k, []) for k in range(n)]

    keys = sorted(sampled)
    per_frame = []
    for k in range(n):
        if k in sampled:
            per_frame.append(sampled[k])
            continue
        prev = k - (k % stride)
        nxt = prev + stride
        per_frame.append(sampled.get(prev, []) + sampled.get(nxt, []))
    return per_frame


def load_user_mask(path, w, h):
    """사용자가 그린 마스크를 영상 해상도의 0/1 배열로 만든다.

    UI 캔버스가 영상보다 작을 수 있으므로 크기가 다르면 맞춰 늘린다.
    """
    m = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if m is None:
        raise RuntimeError(f"마스크를 읽을 수 없습니다: {path}")
    if m.ndim == 3:
        m = m[:, :, 3] if m.shape[2] == 4 else cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
    if m.shape[:2] != (h, w):
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
    return (m > 127).astype(np.uint8)


def band_from_mask(mask, w, h, pad=8):
    """마스크를 감싸는 처리 밴드. choose_band와 같은 규칙(가로의 5/18 하한)."""
    ys = np.where(mask.any(axis=1))[0]
    if not len(ys):
        return None
    ymin, ymax = max(0, int(ys[0]) - pad), min(h, int(ys[-1]) + 1 + pad)
    band_h = min(h, max(ymax - ymin, int(w * 5 / 18)))
    center = (ymin + ymax) // 2
    by0 = max(0, center - band_h // 2)
    by1 = min(h, by0 + band_h)
    return max(0, by1 - band_h), by1, ymin, ymax


def choose_band(per_frame, w, h, pad=8):
    """감지 결과를 감싸는 처리 밴드. STTN 입력이 432x240 고정이라 지나치게 얇게
    자르면 세로로 심하게 늘어나므로, 가로의 5/18을 하한으로 둔다."""
    ys = [y for boxes in per_frame for (_, ya, _, yb) in boxes for y in (ya, yb)]
    if not ys:
        return None
    ymin, ymax = max(0, min(ys) - pad), min(h, max(ys) + pad)
    band_h = min(h, max(ymax - ymin, int(w * 5 / 18)))
    center = (ymin + ymax) // 2
    by0 = max(0, center - band_h // 2)
    by1 = min(h, by0 + band_h)
    by0 = max(0, by1 - band_h)
    return by0, by1, ymin, ymax


def cache_pass(path, by0, by1, n, work_dir, on_progress=None, quiet=False,
               should_abort=None):
    """처리 밴드를 432x240으로 줄여 memmap에 저장."""
    mm_path = os.path.join(work_dir, "bands.dat")
    mm = np.memmap(mm_path, dtype=np.uint8, mode="w+", shape=(n, MODEL_H, MODEL_W, 3))
    cap = cv2.VideoCapture(path)
    i = 0
    with Progress("cache", n, "2/3 밴드 캐시", on_progress, quiet,
                  should_abort=should_abort) as p:
        while i < n:
            ok, fr = cap.read()
            if not ok:
                break
            fr = to_bgr(fr)
            mm[i] = cv2.resize(fr[by0:by1], (MODEL_W, MODEL_H), interpolation=cv2.INTER_AREA)
            i += 1
            p.update()
    cap.release()
    mm.flush()
    return mm, mm_path, i


def make_engine(backend, model, onnx_dir, device, group_size, n_refs, ref_span,
                cache_size, log=print):
    """엔진을 만든다. group_size/n_refs가 None이면 이 기기에 맞춰 자동으로 정한다.

    backend='onnx'|'torch'|'auto'. auto는 ONNX 그래프가 있으면 ONNX를 쓴다.
    """
    if backend == "auto":
        backend = "onnx" if os.path.isfile(os.path.join(onnx_dir, "transformer.onnx")) \
            else "torch"

    if backend == "onnx":
        from .engine_onnx import ONNXEngine, describe
        eng = ONNXEngine(model_dir=onnx_dir, group_size=group_size or 5,
                         n_refs=n_refs or 8, ref_span=ref_span, cache_size=cache_size,
                         prefer=None if device in ("auto", "cuda") else device)
        provider = eng.active[0]
        log(f"백엔드: ONNX Runtime / {describe(eng.active)}")
    else:
        try:
            from .engine import STTNEngine
        except ImportError as e:
            raise RuntimeError(
                f"torch 백엔드를 쓸 수 없습니다 ({e}). 배포판에는 torch가 없습니다.\n"
                f"ONNX 그래프를 찾지 못해 여기까지 온 것이라면 다음을 확인하세요: "
                f"{os.path.join(onnx_dir, 'transformer.onnx')}") from e
        dev = "cuda" if device in ("auto", "cuda") else device
        eng = STTNEngine(model, device=dev, group_size=group_size or 5,
                         n_refs=n_refs or 8, ref_span=ref_span, cache_size=cache_size)
        provider = "CUDAExecutionProvider" if dev.startswith("cuda") else "CPUExecutionProvider"
        log(f"백엔드: PyTorch / {dev}")

    if group_size is None or n_refs is None:
        # t는 양쪽 엔진 모두 동적이므로 세션을 다시 만들 필요 없이 값만 바꿔 잰다.
        def factory(g, r):
            eng.group_size, eng.n_refs = g, r
            eng.reset_cache()
            return eng

        from .autotune import autotune
        g, r, msg = autotune(factory, provider=provider, verbose=log is print)
        eng.group_size, eng.n_refs = g, r
        eng.reset_cache()
        log(f"자동 설정: {msg}")

    return eng


def run(inp, out, region="0.70,1.0", det_stride=1, device="auto", group_size=None,
        n_refs=None, ref_span=600, feather=4, model=None, crf=18,
        mask_pad=3, det_side_len=960, cache_size=64, work_dir=None,
        backend="auto", onnx_dir=None, on_progress=None, quiet=False,
        mask_path=None, mask_mode="detect", should_abort=None):
    """mask_path: 사용자가 그린 마스크(PNG). 주면 region 대신 이걸 쓴다.

    mask_mode
      'detect' — 그린 영역 '안에서' 자막을 프레임마다 찾아 지운다.
                 자막이 없는 프레임이 참조로 살아나므로 품질이 가장 좋다.
      'static' — 그린 영역 전체를 모든 프레임에서 지운다.
                 로고·고정 워터마크처럼 늘 같은 자리에 있는 것에 쓴다.
    """
    model = model or DEFAULT_TORCH_MODEL
    onnx_dir = onnx_dir or DEFAULT_ONNX_DIR
    log = (lambda *a: None) if quiet else print

    info = probe(inp)
    w, h, fps = info["w"], info["h"], info["fps"]
    n_total = info["n"]

    user = load_user_mask(mask_path, w, h) if mask_path else None
    if user is not None and not user.any():
        raise RuntimeError("마스크가 비어 있습니다. 지울 영역을 칠해 주세요.")

    # ---- 처리 밴드와 프레임별 마스크 결정 ----
    if user is not None and mask_mode == "static":
        # 그린 영역을 모든 프레임에서 지운다. 감지 패스가 필요 없다.
        band = band_from_mask(user, w, h)
        by0, by1, ymin, ymax = band
        band_h = by1 - by0
        log(f"입력: {w}x{h} @ {fps:.3f}fps, {n_total} frames | 사용자 마스크 (전체 프레임 적용)")
        log(f"마스크 범위 y={ymin}~{ymax} -> 처리 밴드 y={by0}~{by1} "
            f"({w}x{band_h} -> {MODEL_W}x{MODEL_H})")

        band_mask = user[by0:by1]
        small_band = cv2.resize(band_mask, (MODEL_W, MODEL_H),
                                interpolation=cv2.INTER_NEAREST)
        n = n_total

        def full_mask(i):
            return band_mask

        def small_mask(i):
            return small_band

        # 모든 프레임이 같으므로 참조 선택은 시간축으로 고르게만 퍼지면 된다
        mask_area = np.full(n, int(small_band.sum()), dtype=np.int64)
    else:
        if user is not None:
            # 그린 영역을 탐색 범위로 삼는다
            ys = np.where(user.any(axis=1))[0]
            y0, y1 = max(0, int(ys[0]) - 4), min(h, int(ys[-1]) + 5)
            log(f"입력: {w}x{h} @ {fps:.3f}fps, {n_total} frames | "
                f"사용자 마스크 안에서 감지 y={y0}~{y1}")
        else:
            y0, y1 = parse_region(region, h)
            log(f"입력: {w}x{h} @ {fps:.3f}fps, {n_total} frames | 감지 구역 y={y0}~{y1}")

        t0 = time.time()
        per_frame = detect_pass(inp, y0, y1, det_stride,
                                TextDetector(limit_side_len=det_side_len),
                                on_progress=on_progress, quiet=quiet,
                                should_abort=should_abort)
        if user is not None:
            # 그린 영역 밖으로 삐져나온 감지는 버린다
            per_frame = [[b for b in boxes
                          if user[max(0, b[1]):b[3], max(0, b[0]):b[2]].any()]
                         for boxes in per_frame]
        n = len(per_frame)
        log(f"감지 완료: {sum(1 for b in per_frame if b)}/{n} 프레임 ({time.time()-t0:.1f}s)")

        band = choose_band(per_frame, w, h)
        if band is None:
            raise RuntimeError("지정한 영역에서 텍스트를 찾지 못했습니다. "
                               "영역을 조정하거나 '영역 전체 지우기'로 바꿔 보세요.")
        by0, by1, ymin, ymax = band
        band_h = by1 - by0
        log(f"자막 범위 y={ymin}~{ymax} -> 처리 밴드 y={by0}~{by1} "
            f"({w}x{band_h} -> {MODEL_W}x{MODEL_H})")

        # 마스크는 박스에서 즉석 생성한다(사각형 몇 개 채우는 비용). 저장하지 않는다.
        sx, sy = MODEL_W / w, MODEL_H / band_h

        def boxes_of(i):
            return [(x1, ya - by0, x2, yb - by0) for x1, ya, x2, yb in per_frame[i]]

        def full_mask(i):
            return boxes_to_mask(boxes_of(i), band_h, w, pad=mask_pad)

        def small_mask(i):
            b = [(int(x1 * sx), int(ya * sy), int(np.ceil(x2 * sx)), int(np.ceil(yb * sy)))
                 for x1, ya, x2, yb in boxes_of(i)]
            return boxes_to_mask(b, MODEL_H, MODEL_W, pad=1)

        mask_area = np.array([small_mask(i).sum() for i in range(n)], dtype=np.int64)
    tmp = work_dir or tempfile.mkdtemp(prefix="hse_")
    os.makedirs(tmp, exist_ok=True)

    mm, mm_path, n_cached = cache_pass(inp, by0, by1, n, tmp,
                                       on_progress=on_progress, quiet=quiet,
                                should_abort=should_abort)
    n = min(n, n_cached)

    engine = make_engine(backend, model, onnx_dir, device, group_size, n_refs,
                         ref_span, cache_size, log=log)
    group_size = engine.group_size

    # 사진 한 장을 넣으면 사진이 나와야 한다. mp4 로 감싸면 쓸 수가 없다.
    single = info.get("is_image") or n <= 1
    writer = None if single else FFmpegWriter(out, w, h, fps, audio_from=inp, crf=crf)
    if single and not out.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
        out = os.path.splitext(out)[0] + ".png"
    cap = cv2.VideoCapture(inp)

    t1 = time.time()
    buf, idxs, n_painted = [], [], 0

    def flush():
        nonlocal buf, idxs, n_painted
        if not buf:
            return
        if mask_area[idxs].any():
            outs = engine.process_group(idxs, n, lambda i: mm[i], small_mask, mask_area)
            for f, i, o in zip(buf, idxs, outs):
                f[by0:by1] = composite(f[by0:by1], full_mask(i), o, feather=feather)
            n_painted += len(buf)
        for f in buf:
            if writer:
                writer.write(f)
            else:
                cv2.imwrite(out, f)      # 사진 한 장
        buf, idxs = [], []

    with Progress("paint", n, "3/3 인페인팅", on_progress, quiet,
                  should_abort=should_abort) as p:
        for i in range(n):
            ok, fr = cap.read()
            if not ok:
                break
            buf.append(to_bgr(fr))
            idxs.append(i)
            p.update(detail=f"{i+1}/{n} 프레임")
            if len(buf) >= group_size:
                flush()
        flush()

    cap.release()
    rc = writer.close() if writer else 0
    dt = time.time() - t1

    del mm
    try:
        os.remove(mm_path)
    except OSError:
        pass

    log(f"\n완료 (ffmpeg rc={rc}) -> {out}")
    log(f"인페인팅 {n_painted}/{n} 프레임 | {dt:.1f}s ({n/max(dt,1e-9):.1f} fps)")

    # torch가 설치돼 있을 때만 VRAM을 보고한다(배포판엔 없다).
    if not quiet:
        try:
            import torch
            if torch.cuda.is_available() and torch.cuda.max_memory_allocated():
                log(f"GPU 피크 VRAM: {torch.cuda.max_memory_allocated()/1024**2:.0f} MB")
        except ImportError:
            pass

    if rc != 0:
        raise RuntimeError(f"ffmpeg 인코딩 실패 (rc={rc})")
    return out
