"""network_sttn.py:149 `scores.masked_fill(m, -1e9)` 는 결과를 버린다
(대입도 없고 in-place `masked_fill_`도 아니다). 즉 어텐션 마스크가 무효인지 확인.

같은 feat에 전혀 다른 mask를 넣어 출력이 동일하면 무효가 맞다.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hse.sttn.network_sttn import InpaintGenerator  # noqa: E402

m = InpaintGenerator()
sd = torch.load("models/sttn/sttn.pth", map_location="cpu")
m.load_state_dict(sd["netG"] if "netG" in sd else sd)
m.eval()

torch.manual_seed(0)
feat = torch.randn(13, 256, 60, 108)
m_zero = torch.zeros(13, 1, 240, 432)
m_ones = torch.ones(13, 1, 240, 432)
m_rand = (torch.rand(13, 1, 240, 432) > 0.5).float()

with torch.no_grad():
    a = m.infer(feat, m_zero)
    b = m.infer(feat, m_ones)
    c = m.infer(feat, m_rand)

print(f"mask=0 vs mask=1    최대차: {(a - b).abs().max().item():.3e}")
print(f"mask=0 vs mask=랜덤 최대차: {(a - c).abs().max().item():.3e}")
print()
print("0이면 어텐션 마스크는 출력에 영향이 없다는 뜻.")
print("실제 마스킹은 인코더 입력의 feats*(1-mask)에서 일어난다.")
