"""감지 단계 계측: 입력 스케일/스레드에 따른 속도."""
import time

import cv2
import numpy as np
from rapidocr_onnxruntime import RapidOCR

img = cv2.imread("samples/test_f.png")
H, W = img.shape[:2]
band = img[int(H * 0.70):, :]
print(f"프레임 {W}x{H} -> 감지 밴드 {band.shape[1]}x{band.shape[0]}")

e = RapidOCR()

# 내부 설정 확인
for attr in ("text_det", "det_op", "text_score"):
    o = getattr(e, attr, None)
    if o is None:
        continue
    for k in ("limit_side_len", "limit_type", "thresh", "box_thresh"):
        if hasattr(o, k):
            print(f"  {attr}.{k} = {getattr(o, k)}")
    sess = getattr(o, "session", None) or getattr(o, "infer_session", None)
    if sess is not None and hasattr(sess, "get_inputs"):
        print(f"  {attr} onnx input shape = {sess.get_inputs()[0].shape}")

def timeit(im, n=8):
    e(im, use_det=True, use_cls=False, use_rec=False)  # warmup
    t = time.time()
    for _ in range(n):
        r, _ = e(im, use_det=True, use_cls=False, use_rec=False)
    return (time.time() - t) / n, (len(r) if r else 0)

print(f"\n{'입력':>14} {'ms/frame':>10} {'boxes':>6}")
for scale in (1.0, 0.75, 0.5):
    im = band if scale == 1.0 else cv2.resize(band, None, fx=scale, fy=scale)
    dt, nb = timeit(im)
    print(f"{im.shape[1]}x{im.shape[0]:<9} {dt*1000:>10.0f} {nb:>6}")

# 전체 프레임 대비 밴드 크롭 효과
dt, nb = timeit(img)
print(f"{'(전체프레임)':>14} {dt*1000:>10.0f} {nb:>6}")

import onnxruntime
print(f"\nonnxruntime {onnxruntime.__version__}, providers={onnxruntime.get_available_providers()}")
