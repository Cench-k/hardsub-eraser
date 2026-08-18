"""자동 조정 동작 확인. 탐지 -> 추천 -> 실측 -> 하향까지 전부 돌려본다."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hse.autotune import TABLE, autotune, detect_vram, estimate_mb, probe, suggest  # noqa: E402
from hse.engine_onnx import ONNXEngine, describe, pick_providers  # noqa: E402

mb, src = detect_vram()
print(f"VRAM 탐지: {mb} MB (출처: {src})\n")

print("가상 GPU별 추천값:")
for v in (0, 1024, 2048, 3072, 4096, 6144, 8192, 12288, 24576):
    g, r = suggest(v or None)
    print(f"  {v:>6} MB -> group={g} refs={r}  t={g+r:<3} 예상 {estimate_mb(g,r):>5} MB")
print(f"  CPU     -> group={suggest(8192,'CPUExecutionProvider')[0]} "
      f"refs={suggest(8192,'CPUExecutionProvider')[1]}")

prov = pick_providers(None)
print(f"\n실행 provider: {describe(prov)}\n")


def make(g, r):
    return ONNXEngine(model_dir="models/onnx", group_size=g, n_refs=r, ref_span=600)


print("=== 실측 probe (설정별 프레임당 시간) ===")
for g, r in ((5, 8), (5, 14), (6, 16), (8, 24)):
    per, ok, err = probe(make, g, r, budget_s=1.2)
    tag = "OK" if ok else (err or "예산 초과")
    val = f"{per:.3f} s/frame" if per != float("inf") else "실패"
    print(f"  group={g:<2} refs={r:<3} t={g+r:<3} {val:>16}   {tag}")

print("\n=== autotune() 최종 ===")
g, r, msg = autotune(make, provider=prov[0])
print(f"\n선택: group_size={g}, n_refs={r}")
print(msg)
