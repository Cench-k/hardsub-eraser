"""core 그래프를 t 동적으로 내보낼 수 있는지 시험.

되면 설정마다 65MB짜리 그래프를 따로 싣지 않아도 되고, VRAM 자동 조정이
n_refs를 자유롭게 고를 수 있게 된다.

우려: 트랜스포머가 `t = bt // b`, `out_w = w // width` 처럼 shape에서 파생된
정수로 reshape을 한다. 기존 TorchScript 추적기는 이걸 상수로 구워버린다.
torch 2.x의 dynamo 익스포터는 심볼릭으로 잡을 수 있으므로 그것부터 시도한다.
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hse.common import MODEL_H, MODEL_W  # noqa: E402
from hse.sttn.network_sttn import InpaintGenerator  # noqa: E402

FH, FW = MODEL_H // 4, MODEL_W // 4
OUT = "models/onnx/_probe_dynamic.onnx"


class Core(nn.Module):
    """디코딩 개수도 인자로 받으면 그래프가 복잡해지므로 전체를 디코딩하고
    호출부에서 잘라 쓴다. 어차피 group_size는 t에 비해 작다."""

    def __init__(self, model):
        super().__init__()
        self.transformer = model.transformer
        self.decoder = model.decoder

    def forward(self, feat):
        m = torch.zeros(feat.shape[0], 1, FH, FW, dtype=feat.dtype, device=feat.device)
        enc = self.transformer({"x": feat, "m": m, "b": 1, "c": feat.size(1)})["x"]
        return torch.tanh(self.decoder(enc))


def main():
    model = InpaintGenerator()
    sd = torch.load("models/sttn/sttn.pth", map_location="cpu")
    model.load_state_dict(sd["netG"] if "netG" in sd else sd)
    model.eval()
    core = Core(model).eval()

    feat13 = torch.randn(13, 256, FH, FW)
    os.makedirs("models/onnx", exist_ok=True)

    ok = False
    try:
        print("[1] dynamo 익스포터 + 동적 t 시도")
        torch.onnx.export(
            core, (feat13,), OUT, dynamo=True,
            input_names=["feat"], output_names=["out"],
            dynamic_shapes={"feat": {0: torch.export.Dim("t", min=2, max=64)}})
        ok = True
        print("    내보내기 성공")
    except Exception as e:
        print(f"    실패: {type(e).__name__}: {str(e)[:200]}")

    if not ok:
        try:
            print("[2] 기존 추적기 + dynamic_axes 시도")
            torch.onnx.export(
                core, (feat13,), OUT, opset_version=17,
                input_names=["feat"], output_names=["out"],
                dynamic_axes={"feat": {0: "t"}, "out": {0: "t"}})
            ok = True
            print("    내보내기 성공 (단, 상수가 구워졌을 수 있으니 아래 검증 필수)")
        except Exception as e:
            print(f"    실패: {type(e).__name__}: {str(e)[:200]}")

    if not ok:
        print("\n결론: 동적 t 불가. 설정별로 정적 그래프를 따로 내보내야 한다.")
        return

    # ---- 진짜 되는지: 내보낼 때와 다른 t로 돌려 PyTorch와 대조 ----
    import onnxruntime as ort
    sess = ort.InferenceSession(OUT, providers=["CPUExecutionProvider"])
    print(f"\n    onnx 입력 shape: {sess.get_inputs()[0].shape}")

    print("\n=== 다른 t로 검증 (PyTorch 대비 최대 절대오차) ===")
    all_ok = True
    for t in (9, 13, 21):
        f = torch.randn(t, 256, FH, FW) * 0.3  # 포화 방지를 위해 작은 스케일
        with torch.no_grad():
            ref = core(f).numpy()
        try:
            got = sess.run(None, {"feat": f.numpy()})[0]
            err = np.abs(ref - got).max()
            mark = "OK" if err < 1e-2 else "불일치"
            print(f"  t={t:>3}  {err:.3e}  {mark}")
            all_ok &= err < 1e-2
        except Exception as e:
            print(f"  t={t:>3}  실행 실패: {str(e)[:120]}")
            all_ok = False

    print("\n결론:", "동적 t 사용 가능" if all_ok else
          "그래프는 나왔지만 t가 실제로는 고정 — 정적 그래프를 써야 한다")


if __name__ == "__main__":
    main()
