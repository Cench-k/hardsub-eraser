"""사용자 마스크 경로 검증.

UI에서 브러시로 칠한 것과 같은 마스크를 만들어 두 모드를 모두 돌려본다.
  detect — 칠한 영역 '안에서' 자막만 찾아 지운다
  static — 칠한 영역을 모든 프레임에서 지운다
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hse.pipeline import probe, run  # noqa: E402

SRC = "samples/v2.mp4"
info = probe(SRC)
w, h = info["w"], info["h"]
print(f"입력 {w}x{h}, {info['n']} 프레임")

# 실제 자막은 y=1240~1324 근처. 브러시로 대충 칠한 것처럼 만든다.
m = np.zeros((h, w), np.uint8)
cv2.line(m, (int(w * 0.18), 1282), (int(w * 0.82), 1282), 255, 120)   # 굵은 획
cv2.circle(m, (int(w * 0.5), 1282), 70, 255, -1)
os.makedirs("samples", exist_ok=True)
cv2.imwrite("samples/_mask.png", m)
print(f"마스크 생성: 칠한 픽셀 {int((m>127).sum()):,} ({(m>127).mean()*100:.2f}%)")

for mode in ("detect", "static"):
    print(f"\n{'='*52}\n모드: {mode}\n{'='*52}")
    try:
        run(SRC, f"samples/_mask_{mode}.mp4", mask_path="samples/_mask.png",
            mask_mode=mode, quiet=False)
    except Exception as e:
        print(f"[실패] {type(e).__name__}: {e}")
