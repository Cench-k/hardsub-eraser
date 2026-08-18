"""ONNX Runtime 기반 STTN 엔진. 배포용 실행 경로.

torch(약 3GB, NVIDIA 전용) 대신 onnxruntime 하나로
NVIDIA(CUDA) / AMD·Intel(DirectML) / CPU를 모두 커버한다.

세 그래프를 쓴다(tools/export_onnx.py 참고):
  encoder     — 프레임별. 참조 프레임 결과를 캐시해 그룹 간에 재사용한다.
  transformer — t = group_size + n_refs 개를 한 번에. t는 동적이라
                VRAM에 맞춰 n_refs를 바꿔도 그래프를 다시 만들 필요가 없다.
  decoder     — 실제로 필요한 group_size 개만. 참조까지 디코딩하면 낭비다.

STTNEngine(torch)과 process_group 시그니처가 같아 바꿔 끼울 수 있다.
"""

import os

import cv2
import numpy as np
import onnxruntime as ort

from .common import pick_refs

# 빠른 순서. DirectML은 DX12를 쓰므로 NVIDIA에서도 되지만 보통 CUDA보다 느리다.
PROVIDER_ORDER = ["CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"]
GRAPHS = ("encoder", "transformer", "decoder")


def available_backends():
    avail = set(ort.get_available_providers())
    return [p for p in PROVIDER_ORDER if p in avail]


def pick_providers(prefer=None):
    """prefer가 주어지면 우선 적용, 없거나 사용 불가면 가장 빠른 것으로 자동 선택."""
    avail = available_backends()
    if not avail:
        return ["CPUExecutionProvider"]
    if prefer:
        key = {"cuda": "CUDAExecutionProvider", "dml": "DmlExecutionProvider",
               "directml": "DmlExecutionProvider", "cpu": "CPUExecutionProvider"}.get(
                   str(prefer).lower(), prefer)
        if key in avail:
            return [key] + [p for p in avail if p != key]
    return avail


def describe(providers):
    return {"CUDAExecutionProvider": "NVIDIA CUDA",
            "DmlExecutionProvider": "DirectML (AMD/Intel/NVIDIA)",
            "CPUExecutionProvider": "CPU"}.get(providers[0], providers[0])


class ONNXEngine:
    def __init__(self, model_dir=None, group_size=5, n_refs=8, ref_span=600,
                 cache_size=64, prefer=None, intra_threads=0):
        # 상대 경로 기본값은 작업 디렉터리에 따라 깨진다. 패키지 위치 기준으로 잡는다.
        model_dir = model_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "onnx")
        self.group_size = group_size
        self.n_refs = n_refs
        self.ref_span = ref_span
        self.cache_size = cache_size

        paths = {k: os.path.join(model_dir, k + ".onnx") for k in GRAPHS}
        missing = [p for p in paths.values() if not os.path.isfile(p)]
        if missing:
            raise FileNotFoundError(
                f"ONNX 그래프 없음: {', '.join(os.path.basename(p) for p in missing)}\n"
                f"  python tools/export_onnx.py")

        self.providers = pick_providers(prefer)
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if intra_threads:
            so.intra_op_num_threads = intra_threads

        self.sess = {k: ort.InferenceSession(p, so, providers=self.providers)
                     for k, p in paths.items()}
        # 실제로 붙은 provider (요청한 게 없으면 CPU로 조용히 떨어진다)
        self.active = self.sess["encoder"].get_providers()
        self._cache = {}

    def _encode(self, idxs, get_small, get_mask):
        todo = [i for i in idxs if i not in self._cache]
        if todo:
            imgs = np.stack([cv2.cvtColor(get_small(i), cv2.COLOR_BGR2RGB) for i in todo])
            x = imgs.transpose(0, 3, 1, 2).astype(np.float32) / 255.0 * 2.0 - 1.0
            m = np.stack([get_mask(i) for i in todo])[:, None].astype(np.float32)
            feats = self.sess["encoder"].run(
                None, {"frames": np.ascontiguousarray(x * (1 - m))})[0]
            for k, i in enumerate(todo):
                self._cache[i] = feats[k]

        out = np.stack([self._cache[i] for i in idxs])

        # 축출은 결과를 뽑은 뒤에. 이번 호출에 쓰인 프레임은 버리지 않는다.
        keep = set(idxs)
        overflow = len(self._cache) - max(self.cache_size, len(keep))
        if overflow > 0:
            for i in [k for k in self._cache if k not in keep][:overflow]:
                del self._cache[i]
        return out

    def process_group(self, group, total, get_small, get_mask, mask_area):
        want = len(group)
        group = list(group) + [group[-1]] * (self.group_size - len(group))
        refs = pick_refs(group[len(group) // 2], total, self.n_refs, self.ref_span,
                         mask_area, exclude=set(group))
        idxs = list(group) + refs

        feat = self._encode(idxs, get_small, get_mask)
        enc = self.sess["transformer"].run(None, {"feat": np.ascontiguousarray(feat)})[0]
        # 참조 프레임은 디코딩하지 않는다
        out = self.sess["decoder"].run(
            None, {"enc": np.ascontiguousarray(enc[: self.group_size])})[0]

        out = ((out + 1) / 2).clip(0, 1) * 255
        out = out.transpose(0, 2, 3, 1).astype(np.uint8)  # RGB
        return [cv2.cvtColor(o, cv2.COLOR_RGB2BGR) for o in out[:want]]

    def reset_cache(self):
        self._cache.clear()
