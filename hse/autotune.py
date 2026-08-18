"""임의 사용자 GPU에 맞춰 파라미터를 정한다.

배포판은 torch가 없으므로 `torch.cuda`로 VRAM을 물어볼 수 없다. 그래서
여러 경로를 순서대로 시도한다.

여기서 중요한 건 Windows의 고약한 성질이다. VRAM이 모자라도 드라이버가
시스템 램(공유 GPU 메모리)으로 넘겨버려서 **OOM이 나지 않고 조용히 10배
느려진다**. 실제로 GTX 1060 3GB에서 t를 17로 올렸더니 예외 하나 없이
프레임당 5.7초가 나왔다. 따라서 용량만 보고 정하면 안 되고,
실제로 한 번 돌려서 속도를 재는 편이 확실하다(probe).
"""

import os
import subprocess
import sys
import time

import numpy as np

from .common import MODEL_H, MODEL_W

# t = group_size + n_refs 기준 대략적인 소요 VRAM (GTX 1060 3GB 실측 외삽)
#   t=9  -> 1.0GB,  t=13 -> 1.7GB   (측정값)
# 여유분(디스플레이·드라이버)을 빼고 전체의 70%만 쓸 수 있다고 본다.
_MB_PER_T = 170
_BASE_MB = 500

# (최소 VRAM MB, 시작 group_size, n_refs)
#
# n_refs 는 품질을 좌우하므로 VRAM 여유에 따라 정한다.
# group_size 는 속도를 좌우하는데 최적점이 GPU마다 다르고 절벽처럼 꺾이므로
# 표에 박아두지 않고 아래 hill-climb 으로 실제로 찾는다.
#
# GTX 1060 3GB 실측: group 을 5에서 10으로 올리면 참조 수가 같은데도 1.41배
# 빨라진다(3.99 -> 5.64 fps). 트랜스포머가 t 프레임을 처리해 group 개를 내놓으므로
# group 을 키우면 같은 비용이 더 많은 출력에 분산된다. 단 t=20 을 넘으면
# 공유 메모리로 밀려 0.18 -> 0.45 s/frame 으로 폭락한다.
TABLE = [
    (7000, 6, 16),
    (5000, 6, 12),
    (3500, 6, 10),
    (2500, 6, 8),
    (1600, 4, 5),
    (0,    3, 3),
]

CPU_PROFILE = (4, 4)  # CPU는 t를 작게. 어차피 느리다.
MAX_GROUP = 32


def _nvidia_smi():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=6)
        if out.returncode == 0:
            vals = [int(x) for x in out.stdout.split() if x.strip().isdigit()]
            if vals:
                return max(vals)
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return None


def _windows_registry():
    """드라이버가 기록해 둔 실제 VRAM 크기.

    WMI의 Win32_VideoController.AdapterRAM은 32비트라 4GB에서 잘리므로 쓰지 않는다.
    레지스트리의 qwMemorySize는 64비트라 정확하다.
    """
    if sys.platform != "win32":
        return None
    try:
        import winreg
        base = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
        best = None
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, base) as k:
            for i in range(16):
                try:
                    sub = winreg.EnumKey(k, i)
                except OSError:
                    break
                if not sub.isdigit():
                    continue
                try:
                    with winreg.OpenKey(k, sub) as sk:
                        v, _ = winreg.QueryValueEx(sk, "HardwareInformation.qwMemorySize")
                        mb = int(v) // (1024 * 1024)
                        if mb > (best or 0):
                            best = mb
                except OSError:
                    continue
        return best
    except Exception:  # noqa: BLE001 — 탐지 실패는 치명적이지 않다
        return None


def _torch():
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
    except Exception:  # noqa: BLE001
        pass
    return None


def detect_vram():
    """(MB, 출처). 못 찾으면 (None, 'unknown')."""
    for fn, name in ((_nvidia_smi, "nvidia-smi"), (_torch, "torch"),
                     (_windows_registry, "registry")):
        mb = fn()
        if mb:
            return mb, name
    return None, "unknown"


def suggest(vram_mb, provider="", usable=0.70):
    """VRAM에서 (group_size, n_refs)를 고른다."""
    if provider.startswith("CPU"):
        return CPU_PROFILE
    if not vram_mb:
        return (5, 8)  # 모르면 중간값. probe가 뒤에서 걸러준다.
    budget = vram_mb * usable
    for need, g, r in TABLE:
        if budget >= need * usable:
            return g, r
    return TABLE[-1][1], TABLE[-1][2]


def estimate_mb(group_size, n_refs):
    return _BASE_MB + (group_size + n_refs) * _MB_PER_T


def probe(make_engine, group_size, n_refs, frames=None, budget_s=1.2, rounds=2):
    """설정 하나를 실제로 돌려 프레임당 시간을 잰다.

    용량 계산만으로는 공유 메모리 스필을 못 잡는다. 예외가 아니라 속도로 드러나기
    때문이다. 합성 프레임으로 그룹 두 번을 돌려보고 budget_s를 넘으면 실패로 본다.
    """
    t = group_size + n_refs
    n = max(t + 2, 2 * group_size)
    if frames is None:
        rng = np.random.default_rng(0)
        base = rng.integers(0, 255, (MODEL_H, MODEL_W, 3), dtype=np.uint8)
        frames = [np.roll(base, i * 4, axis=1) for i in range(n)]

    mask = np.zeros((MODEL_H, MODEL_W), np.uint8)
    mask[150:200, 80:350] = 1
    area = np.full(n, int(mask.sum()), dtype=np.int64)

    try:
        eng = make_engine(group_size, n_refs)
        eng.process_group(list(range(group_size)), n,
                          lambda i: frames[i % n], lambda i: mask, area)  # 워밍업
        t0 = time.time()
        for k in range(rounds):
            g = [(k * group_size + j) % n for j in range(group_size)]
            eng.process_group(g, n, lambda i: frames[i % n], lambda i: mask, area)
        per_frame = (time.time() - t0) / (rounds * group_size)
        return per_frame, per_frame <= budget_s, None
    except Exception as e:  # noqa: BLE001 — OOM 종류가 백엔드마다 달라 모두 잡는다
        return float("inf"), False, f"{type(e).__name__}: {str(e)[:120]}"


def autotune(make_engine, provider="", verbose=True, budget_s=1.2, use_cache=True,
             slack=1.35):
    """탐지 -> 추천 -> 실측 -> 필요하면 하향. 반환 (group_size, n_refs, 설명문).

    추천값이 예산 안에 들어와도 곧바로 받아들이지 않는다. 한 단계 낮은 설정이
    확연히 빠르면(slack배 이상) 그쪽을 쓴다. 참조를 늘려도 품질 이득은 작은데
    (실측 n_refs 4->8에서 +0.29dB) 속도 손해는 크기 때문이다.
    """
    vram, src = detect_vram()
    key = f"{provider or 'unknown'}|{vram or 0}"
    if use_cache:
        hit = _load_cache().get(key)
        if hit:
            g, r = hit["group_size"], hit["n_refs"]
            if verbose:
                print(f"이전 측정값 사용: group={g} refs={r} (초기화하려면 {cache_path()} 삭제)")
            return g, r, f"{hit.get('msg', '')} [캐시]"

    head = f"GPU 메모리 {vram} MB ({src})" if vram else "GPU 메모리 탐지 실패"

    # CPU는 재보나 마나 느리다. 최소 프로필로 바로 간다.
    if str(provider).startswith("CPU"):
        g, r = CPU_PROFILE
        msg = f"CPU 실행 — 최소 설정 group={g} refs={r}"
        if verbose:
            print(msg)
        _save_cache(key, g, r, msg)
        return g, r, msg

    g0, r = suggest(vram, provider)
    if verbose:
        print(f"{head} -> 참조 {r}개로 시작, group 최적점을 찾습니다")

    # 1) 시작점이 도는지 확인. 안 되면 표를 따라 내려간다.
    best = None
    for gg, rr in [(g0, r)] + [(a, b) for _, a, b in TABLE if a + b < g0 + r]:
        per, ok, err = probe(make_engine, gg, rr, budget_s=budget_s)
        if verbose:
            v = f"{per:.3f} s/frame" if per != float("inf") else "실패"
            print(f"  group={gg:<3} refs={rr:<3} {v}  {'OK' if ok else (err or '예산 초과')}")
        if ok:
            best = (gg, rr, per)
            break
    if best is None:
        g, r = TABLE[-1][1], TABLE[-1][2]
        return g, r, f"{head}, 측정 실패 — 최소 설정 group={g} refs={r}"

    # 2) group_size 를 키우며 처리량이 좋아지는 동안 계속 올린다.
    #    최적점은 GPU마다 다르고 VRAM 한계에서 절벽처럼 꺾이므로 표로 못 정한다.
    #    참조 수는 건드리지 않으므로 품질은 그대로다.
    g, r, per = best
    step = 3
    while g + step <= MAX_GROUP:
        p2, ok2, _ = probe(make_engine, g + step, r, budget_s=budget_s)
        gain = per / p2 if (ok2 and p2 > 0) else 0
        if verbose:
            v = f"{p2:.3f} s/frame" if p2 != float("inf") else "실패"
            print(f"  group={g+step:<3} refs={r:<3} {v}" +
                  (f"  {gain:.2f}x" if gain else "  (더 느림)"))
        if gain < 1.03:          # 3% 미만이면 멈춘다
            break
        g, per = g + step, p2

    msg = f"{head}, group={g} refs={r} ({per:.3f}s/frame, {1/per:.1f} fps)"
    _save_cache(key, g, r, msg)
    return g, r, msg


def cache_path():
    d = os.path.join(os.path.expanduser("~"), ".hardsub_eraser")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, "autotune.json")


def _load_cache():
    try:
        import json
        with open(cache_path(), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _save_cache(key, g, r, msg):
    try:
        import json
        d = _load_cache()
        d[key] = {"group_size": g, "n_refs": r, "msg": msg}
        with open(cache_path(), "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=1)
    except OSError:
        pass
