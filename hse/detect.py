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

    """box_thresh/unclip_ratio 도 rapidocr 기본값(0.5/1.6)에서 낮추고 넓힌다.
    DBNet 은 한 줄을 띄어쓰기 단위로 쪼개는데, 두 글자짜리 끝 단어는 면적이 작아
    점수가 0.5 를 못 넘고 통째로 버려진다. 그러면 8글자 중 6글자만 지워진다.
    """

    def __init__(self, limit_side_len=960, limit_type="max", box_thresh=0.3,
                 unclip_ratio=2.0):
        self.engine = RapidOCR(det_limit_side_len=limit_side_len, det_limit_type=limit_type)
        self.box_thresh, self.unclip_ratio = box_thresh, unclip_ratio

    def boxes(self, img):
        """img(BGR) 안의 텍스트 사각형 목록을 반환. 같은 줄의 박스는 하나로 합친다."""
        try:
            # 호출 인자로 넘긴다. rapidocr 는 kwargs 가 하나라도 오면 이 두 값을
            # 기본값으로 덮어쓰므로 생성자 설정만 믿을 수 없다.
            res, _ = self.engine(img, use_det=True, use_cls=False, use_rec=False,
                                 box_thresh=self.box_thresh, unclip_ratio=self.unclip_ratio)
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
        return merge_lines(out)


def merge_lines(boxes, gap=1.0):
    """세로로 절반 이상 겹치고 가로 틈이 글자 높이(x gap) 이하인 박스를 합친다.

    단어 사이 빈틈까지 마스크로 덮어 띄어쓰기 자리에 글자 조각이 남지 않게 한다.
    틈 제한이 있어 같은 높이의 먼 워터마크와는 합쳐지지 않는다.
    """
    boxes = [list(b) for b in boxes]
    merged = True
    while merged:
        merged = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                ha, hb = a[3] - a[1], b[3] - b[1]
                overlap = min(a[3], b[3]) - max(a[1], b[1])
                hgap = max(a[0], b[0]) - min(a[2], b[2])
                if overlap >= 0.5 * min(ha, hb) and hgap <= gap * max(ha, hb):
                    boxes[i] = [min(a[0], b[0]), min(a[1], b[1]),
                                max(a[2], b[2]), max(a[3], b[3])]
                    del boxes[j]
                    merged = True
                    break
            if merged:
                break
    return [tuple(b) for b in boxes]


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
