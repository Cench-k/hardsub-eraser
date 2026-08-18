"""마스크 밖 픽셀이 정말 보존됐는지 확인."""
import cv2
import numpy as np

a = cv2.imread("samples/a_before.png").astype(np.int16)
b = cv2.imread("samples/a_after.png").astype(np.int16)
d = np.abs(a - b).max(axis=2)

H, W = d.shape
print(f"프레임 {W}x{H}")
print(f"변경된 픽셀(차이>8): {(d > 8).sum():,} / {d.size:,} ({(d > 8).mean()*100:.2f}%)")

rows = np.where((d > 8).any(axis=1))[0]
if len(rows):
    print(f"변경 발생 y 범위: {rows.min()} ~ {rows.max()}  (처리 밴드는 244~480)")

# 밴드 안이지만 마스크 밖인 영역의 화질이 유지됐는지: 밴드 상단부(자막 없음)
band_top = d[244:360, :]
print(f"밴드 내 자막 없는 구간(y244~360) 최대 차이: {band_top.max()}  평균: {band_top.mean():.2f}")

# h.264 재인코딩 자체의 차이를 감안한 기준선: 밴드 밖 영역
outside = d[:244, :]
print(f"밴드 밖(y0~244) 최대 차이: {outside.max()}  평균: {outside.mean():.2f}   <- 재인코딩 노이즈 기준선")

cv2.imwrite("samples/a_diff.png", np.clip(d * 6, 0, 255).astype(np.uint8))
print("\n차분맵 -> samples/a_diff.png (6배 증폭)")
