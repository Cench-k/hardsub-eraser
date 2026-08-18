"""STTN을 ONNX 세 그래프로 내보낸다.

  encoder.onnx      [B,3,240,432]     -> [B,256,60,108]     동적 B
  transformer.onnx  [t,256,60,108]    -> [t,256,60,108]     동적 t
  decoder.onnx      [G,256,60,108]    -> [G,3,240,432]      동적 G

세 개로 나눈 이유:

  encoder를 분리 — 참조 프레임의 인코딩 결과를 그룹 간에 캐시하기 위해서.
                   합치면 매 그룹마다 참조를 다시 인코딩해야 한다.
  decoder를 분리 — 트랜스포머는 t개를 처리하지만 실제로 필요한 출력은
                   group_size개뿐이다. 합쳐 두면 참조 프레임까지 디코딩해
                   t/group_size 배(보통 2.6배)의 낭비가 생긴다.

t가 동적이라 VRAM에 맞춰 n_refs를 자유롭게 고를 수 있다(hse/autotune.py).
트랜스포머 내부가 `t = bt // b` 같은 shape 파생 정수로 reshape을 하는데도
동적 t가 되는지는 아래 검증에서 여러 t로 직접 대조해 확인한다.

마스크는 입력으로 받지 않는다. network_sttn.py:149의
`scores.masked_fill(m, -1e9)`가 결과를 버리기 때문에 어텐션 마스크는 출력에
영향이 없다(tools/check_maskfill.py 참고). 실제 마스킹은 인코더 입력에서 끝난다.

    python tools/export_onnx.py
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hse.common import MODEL_H, MODEL_W  # noqa: E402
from hse.sttn.network_sttn import InpaintGenerator  # noqa: E402

FH, FW = MODEL_H // 4, MODEL_W // 4  # 60, 108


class Transformer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.transformer = model.transformer

    def forward(self, feat):
        m = torch.zeros(feat.shape[0], 1, FH, FW, dtype=feat.dtype, device=feat.device)
        return self.transformer({"x": feat, "m": m, "b": 1, "c": feat.size(1)})["x"]


class Decoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.decoder = model.decoder

    def forward(self, enc):
        return torch.tanh(self.decoder(enc))


def _real_feat(model, t, sample="samples/test_f.png"):
    """검증용 특징맵은 실제 인코더 출력을 쓴다.

    randn을 넣으면 어텐션이 414,720차원 내적을 하면서 softmax가 포화되어
    torch/ORT의 미세한 부동소수점 차이가 크게 증폭된다(실사용과 무관한 수치).
    """
    import cv2

    img = cv2.imread(sample)
    if img is None:
        yy, xx = np.mgrid[0:MODEL_H, 0:MODEL_W].astype(np.float32)
        base = np.stack([(np.sin(xx / 30) + np.cos(yy / 20)) * 60 + 128] * 3, -1)
    else:
        h = img.shape[0]
        base = cv2.resize(img[int(h * 0.7):], (MODEL_W, MODEL_H)).astype(np.float32)

    frames = [cv2.cvtColor(np.roll(base, i * 3, axis=1).astype(np.uint8), cv2.COLOR_BGR2RGB)
              for i in range(t)]
    x = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float().div_(255).mul_(2).sub_(1)
    m = torch.zeros(t, 1, MODEL_H, MODEL_W)
    m[:, :, 100:150, 100:330] = 1.0
    with torch.no_grad():
        return model.encoder(x * (1 - m))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/sttn/sttn.pth")
    ap.add_argument("--out-dir", default="models/onnx")
    ap.add_argument("--opset", type=int, default=17)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    model = InpaintGenerator()
    sd = torch.load(a.model, map_location="cpu")
    model.load_state_dict(sd["netG"] if "netG" in sd else sd)
    model.eval()

    paths = {k: os.path.join(a.out_dir, k + ".onnx")
             for k in ("encoder", "transformer", "decoder")}

    torch.onnx.export(
        model.encoder, (torch.randn(2, 3, MODEL_H, MODEL_W),), paths["encoder"],
        opset_version=a.opset, input_names=["frames"], output_names=["feat"],
        dynamic_axes={"frames": {0: "b"}, "feat": {0: "b"}})

    torch.onnx.export(
        Transformer(model).eval(), (torch.randn(13, 256, FH, FW),), paths["transformer"],
        opset_version=a.opset, input_names=["feat"], output_names=["enc"],
        dynamic_axes={"feat": {0: "t"}, "enc": {0: "t"}})

    torch.onnx.export(
        Decoder(model).eval(), (torch.randn(5, 256, FH, FW),), paths["decoder"],
        opset_version=a.opset, input_names=["enc"], output_names=["out"],
        dynamic_axes={"enc": {0: "g"}, "out": {0: "g"}})

    for k, p in paths.items():
        print(f"{k+'.onnx':<20} {os.path.getsize(p)/1e6:>6.1f} MB")

    # ---------- 검증 ----------
    import onnxruntime as ort
    sess = {k: ort.InferenceSession(p, providers=["CPUExecutionProvider"])
            for k, p in paths.items()}

    print("\n=== PyTorch 대비 최대 절대오차 ===")
    x = torch.randn(3, 3, MODEL_H, MODEL_W)
    with torch.no_grad():
        ref = model.encoder(x).numpy()
    got = sess["encoder"].run(None, {"frames": x.numpy()})[0]
    print(f"  encoder(B=3)      {np.abs(ref - got).max():.3e}")

    tf, dec = Transformer(model).eval(), Decoder(model).eval()
    # 내보낼 때 쓴 t(13) 말고 다른 값들로 확인해야 동적 t가 진짜인지 알 수 있다
    ok = True
    for t in (7, 13, 25):
        feat = _real_feat(model, t)
        with torch.no_grad():
            ref_e = tf(feat).numpy()
        got_e = sess["transformer"].run(None, {"feat": feat.numpy()})[0]
        e = np.abs(ref_e - got_e).max()
        ok &= e < 1e-2
        print(f"  transformer(t={t:<2})  {e:.3e}{'' if e < 1e-2 else '   << 불일치'}")

    enc = torch.randn(4, 256, FH, FW)
    with torch.no_grad():
        ref_d = dec(enc).numpy()
    got_d = sess["decoder"].run(None, {"enc": enc.numpy()})[0]
    print(f"  decoder(G=4)      {np.abs(ref_d - got_d).max():.3e}")

    print("\n동적 축 정상" if ok else "\n[!] t를 바꾸면 결과가 어긋난다 — 정적 그래프를 써야 한다")


if __name__ == "__main__":
    main()
