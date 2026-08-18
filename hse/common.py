"""백엔드에 무관한 공통 부분. numpy/cv2만 쓰고 torch를 import하지 않는다.

배포판은 onnxruntime만 싣고 torch는 빼기 때문에, 파이프라인이 torch에
의존하지 않도록 여기로 분리했다.
"""

import cv2
import numpy as np

# STTN 입력 크기. network_sttn.py의 patchsize=[(108,60),(36,20),(18,10),(9,5)]가
# 이 크기를 전제로 하므로 변경 불가.
MODEL_W, MODEL_H = 432, 240


def pick_refs(center, total, n_refs, span, mask_area, exclude):
    """참조 프레임 선택.

    center 주변 span 범위를 n_refs개 구간으로 나누고, 각 구간에서 마스크 면적이
    가장 작은(= 배경이 가장 많이 드러난) 프레임을 고른다. 시간축으로 고르게 퍼지면서
    가장 깨끗한 프레임을 쓰게 된다. 단순 고정 간격 샘플링보다 낫다.

    반환 개수는 항상 정확히 n_refs다. t = group_size + n_refs가 상수여야
    ONNX 정적 shape로 내보낼 수 있고 그룹마다 VRAM 사용량도 일정해진다.
    """
    lo = max(0, center - span // 2)
    hi = min(total, center + span // 2)
    if hi - lo < n_refs:
        lo, hi = 0, total

    refs = []
    edges = np.linspace(lo, hi, n_refs + 1).astype(int)
    for a, b in zip(edges[:-1], edges[1:]):
        cands = [i for i in range(a, b) if i not in exclude]
        if cands:
            refs.append(min(cands, key=lambda i: mask_area[i]))

    refs = sorted(set(refs))
    if not refs:
        refs = [int(np.argmin(mask_area))]
    base = list(refs)
    while len(refs) < n_refs:  # 중복 참조는 어텐션상 무해하다
        refs.append(base[len(refs) % len(base)])
    return refs[:n_refs]


def composite(orig_band, mask_full, small_out, feather=4):
    """432x240 결과를 원본 해상도 밴드의 마스크 영역에만 합성.

    밴드 전체를 덮어쓰지 않는 것이 핵심이다. 덮어쓰면 자막과 무관한 픽셀까지
    432x240 왕복을 거쳐 뭉개진다.
    """
    H, W = orig_band.shape[:2]
    if not mask_full.any():
        return orig_band
    up = cv2.resize(small_out, (W, H), interpolation=cv2.INTER_CUBIC)

    alpha = mask_full.astype(np.uint8)
    if feather > 0:
        # 원래 마스크가 alpha=1로 완전히 덮이도록 먼저 팽창시킨 뒤 블러.
        # 페더 전이 구간이 자막 바깥에 생겨 글자가 되살아나지 않는다.
        alpha = cv2.dilate(alpha, np.ones((3, 3), np.uint8), iterations=feather)
        k = feather * 2 + 1
        alpha = cv2.GaussianBlur(alpha.astype(np.float32), (k, k), 0)
    alpha = np.clip(alpha, 0, 1).astype(np.float32)[:, :, None]

    return (orig_band.astype(np.float32) * (1 - alpha)
            + up.astype(np.float32) * alpha).astype(np.uint8)
