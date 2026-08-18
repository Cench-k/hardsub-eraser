"""정답이 있는 품질 평가.

자막이 없는 구간에 자막을 합성해 넣고(=정답을 아는 상태) 지운 뒤,
원본과 비교해 PSNR/SSIM을 낸다. 눈대중 대신 파라미터를 수치로 튜닝하기 위한 도구.

    python tools/eval_quality.py samples/clean.mp4 --y 700 --h 170
"""

import argparse
import os
import subprocess
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hse.pipeline import run  # noqa: E402

FONT = "C\\:/Windows/Fonts/malgun.ttf"
LINES = ["이건 합성으로 넣은 시험용 자막입니다", "배경이 얼마나 복원되는지 봅니다",
         "글자가 사라진 자리의 질감이 관건", "정답 영상과 직접 비교합니다",
         "참조 프레임이 넓을수록 유리하다"]


def burn(src, dst, y, seg=2.0):
    """seg초마다 다른 문장을 태워 넣는다(실제 자막처럼 바뀌게)."""
    draws = []
    for i, txt in enumerate(LINES):
        draws.append(
            f"drawtext=fontfile='{FONT}':text='{txt}':fontcolor=white:fontsize=52:"
            f"borderw=3:bordercolor=black:x=(w-text_w)/2:y={y}:"
            f"enable='between(t,{i*seg},{(i+1)*seg})'")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src,
                    "-vf", ",".join(draws), "-c:v", "libx264", "-crf", "16",
                    "-pix_fmt", "yuv420p", "-an", dst], check=True)


def ssim(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu_a, mu_b = cv2.GaussianBlur(a, (11, 11), 1.5), cv2.GaussianBlur(b, (11, 11), 1.5)
    sa = cv2.GaussianBlur(a * a, (11, 11), 1.5) - mu_a ** 2
    sb = cv2.GaussianBlur(b * b, (11, 11), 1.5) - mu_b ** 2
    sab = cv2.GaussianBlur(a * b, (11, 11), 1.5) - mu_a * mu_b
    m = ((2 * mu_a * mu_b + C1) * (2 * sab + C2)) / ((mu_a ** 2 + mu_b ** 2 + C1) * (sa + sb + C2))
    return m


def compare(orig_path, burned_path, erased_path):
    """자막이 있던 픽셀에서만 원본 대비 PSNR/SSIM을 낸다."""
    co, cb, ce = (cv2.VideoCapture(p) for p in (orig_path, burned_path, erased_path))
    se, ss, npx, nf = 0.0, 0.0, 0, 0
    while True:
        ok1, fo = co.read()
        ok2, fb = cb.read()
        ok3, fe = ce.read()
        if not (ok1 and ok2 and ok3):
            break
        # 자막이 칠해진 픽셀 = 원본과 태운 영상이 다른 곳
        m = (np.abs(fo.astype(np.int16) - fb.astype(np.int16)).max(axis=2) > 24)
        m = cv2.dilate(m.astype(np.uint8), np.ones((5, 5), np.uint8), 2).astype(bool)
        if m.sum() < 100:
            continue
        se += ((fo.astype(np.float64) - fe.astype(np.float64)) ** 2)[m].sum()
        npx += m.sum() * 3
        g_o = cv2.cvtColor(fo, cv2.COLOR_BGR2GRAY)
        g_e = cv2.cvtColor(fe, cv2.COLOR_BGR2GRAY)
        ss += ssim(g_o, g_e)[m].mean()
        nf += 1
    for c in (co, cb, ce):
        c.release()
    mse = se / max(npx, 1)
    return (10 * np.log10(255 ** 2 / mse) if mse > 0 else 99), ss / max(nf, 1), nf


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="자막이 없는 원본 클립")
    ap.add_argument("--y", type=int, default=700, help="자막을 태울 y 위치")
    ap.add_argument("--h", type=int, default=180, help="처리 구역 높이")
    ap.add_argument("--n-refs", type=int, default=8)
    ap.add_argument("--ref-span", type=int, default=600)
    ap.add_argument("--group-size", type=int, default=5)
    ap.add_argument("--det-stride", type=int, default=1)
    ap.add_argument("--backend", default="torch")
    a = ap.parse_args()

    burned = "samples/_eval_burned.mp4"
    erased = "samples/_eval_erased.mp4"
    if not os.path.exists(burned):
        burn(a.src, burned, a.y)
        print(f"자막 합성 완료 -> {burned}")

    h = cv2.VideoCapture(a.src).get(cv2.CAP_PROP_FRAME_HEIGHT)
    region = f"{max(0, a.y - 30)},{a.y + a.h}"
    print(f"처리 구역: y={region} (전체 높이 {int(h)})")

    run(burned, erased, region=region, n_refs=a.n_refs, ref_span=a.ref_span,
        group_size=a.group_size, backend=a.backend, det_stride=a.det_stride)

    psnr, s, nf = compare(a.src, burned, erased)
    print(f"\n=== 자막이 있던 영역만 원본과 비교 ({nf} 프레임) ===")
    print(f"  PSNR  {psnr:6.2f} dB   (높을수록 좋음, 30 이상이면 양호)")
    print(f"  SSIM  {s:6.4f}         (1.0이 완벽)")
