"""두 출력 영상이 실질적으로 같은지 비교 (백엔드 교체 검증용)."""
import sys

import cv2
import numpy as np

a, b = cv2.VideoCapture(sys.argv[1]), cv2.VideoCapture(sys.argv[2])
mx, tot, n = 0, 0.0, 0
while True:
    ok1, fa = a.read()
    ok2, fb = b.read()
    if not (ok1 and ok2):
        break
    d = np.abs(fa.astype(np.int16) - fb.astype(np.int16))
    mx = max(mx, int(d.max()))
    tot += float(d.mean())
    n += 1
a.release()
b.release()
print(f"{n} 프레임 비교")
print(f"  최대 픽셀 차이: {mx}")
print(f"  평균 픽셀 차이: {tot/max(n,1):.3f}   (h.264 재인코딩만으로도 2~4는 나온다)")
