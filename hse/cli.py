import argparse
import os
import sys

from .pipeline import run

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(argv=None):
    p = argparse.ArgumentParser("hardsub-eraser", description="영상 하드섭 제거 (STTN, 100% 로컬)")
    p.add_argument("input")
    p.add_argument("-o", "--output", default=None)
    p.add_argument("--region", default="0.70,1.0",
                   help="자막 탐색 구역 y0,y1. 0~1이면 비율, 그 이상이면 픽셀 (기본 하단 30%%)")
    p.add_argument("--det-stride", type=int, default=1, help="N프레임마다 감지 (1=전 프레임)")
    p.add_argument("--det-side-len", type=int, default=960, help="감지 입력 긴 변 상한")

    # 트랜스포머에 들어가는 프레임 수 t = group-size + n-refs. VRAM은 t에만 비례한다.
    # 지정하지 않으면 이 기기의 VRAM을 재고 실제로 돌려 보며 자동으로 정한다.
    p.add_argument("--group-size", type=int, default=None, help="한 번에 복원할 프레임 수 (기본 자동)")
    p.add_argument("--n-refs", type=int, default=None, help="참조 프레임 수 (기본 자동)")
    p.add_argument("--ref-span", type=int, default=600,
                   help="참조를 뽑아올 시간 범위(프레임). 넓을수록 깨끗한 배경을 찾을 확률↑")
    p.add_argument("--cache-size", type=int, default=64, help="인코더 특징 캐시 프레임 수")

    p.add_argument("--feather", type=int, default=4, help="마스크 경계 페더링 강도(px)")
    p.add_argument("--mask-pad", type=int, default=3, help="감지 박스 여유(px)")
    p.add_argument("--crf", type=int, default=18, help="출력 화질 (낮을수록 고화질)")
    p.add_argument("--backend", default="auto", choices=["auto", "onnx", "torch"],
                   help="auto: ONNX가 준비돼 있으면 ONNX, 없으면 torch")
    p.add_argument("--device", default="auto",
                   help="auto | cuda | dml(DirectML) | cpu")
    p.add_argument("--model", default=os.path.join(ROOT, "models", "sttn", "sttn.pth"))
    p.add_argument("--onnx-dir", default=os.path.join(ROOT, "models", "onnx"))
    p.add_argument("--work-dir", default=None, help="밴드 캐시 위치 (기본: 임시 폴더)")
    a = p.parse_args(argv)

    if not os.path.isfile(a.input):
        sys.exit(f"입력 파일이 없습니다: {a.input}")
    out = a.output or os.path.splitext(a.input)[0] + "_clean.mp4"

    run(a.input, out, region=a.region, det_stride=a.det_stride, device=a.device,
        group_size=a.group_size, n_refs=a.n_refs, ref_span=a.ref_span, feather=a.feather,
        model=a.model, crf=a.crf, mask_pad=a.mask_pad, det_side_len=a.det_side_len,
        cache_size=a.cache_size, work_dir=a.work_dir, backend=a.backend,
        onnx_dir=a.onnx_dir)


if __name__ == "__main__":
    main()
