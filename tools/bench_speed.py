"""group_size 최적점 재확인. 단발 측정은 노이즈가 섞이므로 여러 번 반복한다."""
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hse.autotune import probe  # noqa: E402
from hse.engine_onnx import ONNXEngine  # noqa: E402


def make(g, r):
    return ONNXEngine(group_size=g, n_refs=r, ref_span=600)


print(f"{'group':>6} {'refs':>5} {'t':>4}   {'s/frame (3회 중앙값)':>22} {'fps':>7} {'대비':>7}")
res = {}
for g, r in ((5, 8), (6, 8), (7, 8), (8, 8), (9, 8), (10, 8), (8, 10)):
    vals = []
    for _ in range(3):
        per, ok, err = probe(make, g, r, budget_s=99, rounds=3)
        if per != float("inf"):
            vals.append(per)
    if not vals:
        print(f"{g:>6} {r:>5} {g+r:>4}   실패")
        continue
    m = statistics.median(vals)
    res[(g, r)] = m
    base = res.get((5, 8), m)
    spread = f"{min(vals):.3f}~{max(vals):.3f}"
    print(f"{g:>6} {r:>5} {g+r:>4} {m:>10.3f}  ({spread}) {1/m:>7.2f} {base/m:>6.2f}x")

best = min(res, key=res.get)
print(f"\n최적: group={best[0]} refs={best[1]}  ({1/res[best]:.2f} fps, "
      f"기본 대비 {res[(5,8)]/res[best]:.2f}x)")
