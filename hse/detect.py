"""자막 텍스트 영역 감지.

PaddleOCR 대신 rapidocr-onnxruntime(같은 DBNet 계열을 ONNX로 변환)을 쓴다.
PaddlePaddle 의존성이 사라지고, 감지 단계는 CPU로 돌려 3GB VRAM을 전부
인페인팅에 넘길 수 있다. 인식(rec)은 끄고 감지(det)만 사용한다.
"""

import numpy as np
from rapidocr_onnxruntime import RapidOCR


def _to_rect(item):
    """rapidocr가 돌려주는 다양한 형태를 (x1,y1,x2,y2) 정수 사각형으로 정규화."""
    box = item
    # rec를 켠 경우 (box, text, score) 튜플로 나온다
    if isinstance(item, (list, tuple)) and len(item) == 3 and not np.isscalar(item[1]):
        box = item[0]
    pts = np.asarray(box, dtype=np.float32).reshape(-1, 2)
    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)
    return int(x1), int(y1), int(x2), int(y2)


class TextDetector:
    """rapidocr 기본값은 limit_type='min', limit_side_len=736 이라
    짧은 변을 736까지 '확대'한다. 852x144 자막 밴드가 4350x736으로 부풀려져
    프레임당 719ms가 걸렸다. limit_type='max'로 긴 변을 제한하면 28ms로 떨어진다.
    """

    def __init__(self, limit_side_len=960, limit_type="max"):
        self.engine = RapidOCR(det_limit_side_len=limit_side_len, det_limit_type=limit_type)

    def boxes(self, img):
        """img(BGR) 안의 텍스트 사각형 목록을 반환."""
        try:
            res, _ = self.engine(img, use_det=True, use_cls=False, use_rec=False)
        except TypeError:
            # 구버전 시그니처 대응
            res, _ = self.engine(img)
        if not res:
            return []
        out = []
        for item in res:
            try:
                out.append(_to_rect(item))
            except Exception:
                continue
        return out


def boxes_to_mask(boxes, h, w, pad=3):
    """사각형 목록 -> 이진 마스크. pad만큼 여유를 줘 글자 외곽선까지 덮는다."""
    m = np.zeros((h, w), dtype=np.uint8)
    for x1, y1, x2, y2 in boxes:
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)
        if x2 > x1 and y2 > y1:
            m[y1:y2, x1:x2] = 1
    return m
