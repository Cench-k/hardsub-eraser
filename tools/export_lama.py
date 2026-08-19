"""LaMa(big-lama)를 ONNX로 내보낸다. 사진 인페인팅용.

STTN 은 영상 모델이다. 여러 프레임의 배경을 참조해 채우는 게 강점이라
사진 한 장에는 참조할 게 없어 주변 픽셀만으로 추론한다. 사진은 LaMa 가 맞다.

입력
  image [1,3,H,W]  float 0~1
  mask  [1,1,H,W]  0 또는 1 (1인 곳을 채운다)
H, W 는 8의 배수여야 한다. 동적 축으로 내보내 어떤 크기든 받게 한다.

    python tools/export_lama.py
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/big-lama/big-lama.pt")
    ap.add_argument("--out-dir", default="models/onnx")
    ap.add_argument("--opset", type=int, default=17)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    out = os.path.join(a.out_dir, "lama.onnx")

    m = torch.jit.load(a.model, map_location="cpu").eval()

    img = torch.rand(1, 3, 256, 256)
    msk = torch.zeros(1, 1, 256, 256)
    msk[:, :, 100:150, 100:160] = 1.0

    torch.onnx.export(
        m, (img, msk), out, opset_version=a.opset,
        input_names=["image", "mask"], output_names=["out"],
        dynamic_axes={"image": {2: "h", 3: "w"},
                      "mask": {2: "h", 3: "w"},
                      "out": {2: "h", 3: "w"}})
    print(f"lama.onnx  {os.path.getsize(out)/1e6:.1f} MB")

    # ---------- 검증: 내보낼 때와 다른 크기로도 맞는지 ----------
    import onnxruntime as ort
    sess = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
    print(f"입력 shape: {[i.shape for i in sess.get_inputs()]}")

    print("\n=== PyTorch 대비 최대 절대오차 ===")
    ok = True
    rng = np.random.default_rng(0)
    for h, w in ((256, 256), (320, 512), (704, 904)):
        im = torch.from_numpy(rng.random((1, 3, h, w), dtype=np.float32))
        mk = torch.zeros(1, 1, h, w)
        mk[:, :, h // 3:h // 3 + 40, w // 3:w // 3 + 60] = 1.0
        with torch.inference_mode():
            ref = m(im, mk).numpy()
        got = sess.run(None, {"image": im.numpy(), "mask": mk.numpy()})[0]
        e = np.abs(ref - got).max()
        ok &= e < 1e-3
        print(f"  {w}x{h}  {e:.3e}{'' if e < 1e-3 else '   << 불일치'}")

    print("\n동적 해상도 정상" if ok else "\n[!] 해상도를 바꾸면 결과가 어긋난다")


if __name__ == "__main__":
    main()
