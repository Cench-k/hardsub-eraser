"""STTN 추론 엔진 (PyTorch). 개발/연구용 경로.

배포 실행 경로는 engine_onnx.ONNXEngine 이고, 이 파일은 ONNX 내보내기의
기준 구현이자 수치 비교 대상으로 남긴다. process_group 시그니처는 양쪽이 같다.

Phase 0의 청크 방식은 청크 안에서만 참조를 뽑았다. 청크를 키우면 참조 구간이
넓어지지만 인코더 특징이 프레임당 6.6MB(256x60x108x4)씩 상주해서, 120프레임이면
796MB가 붙박이로 잡히고 Windows 공유 메모리로 밀려나 10배 이상 느려졌다.

여기서는 청크 개념을 버리고 트랜스포머에 들어가는 프레임 수 t를 상수로 고정한다.

    t = group_size(출력 프레임) + n_refs(참조 프레임)

메모리는 t에만 비례하므로 참조를 영상 전체 어디서든 뽑아도 비용이 같다.
"""

import cv2
import numpy as np
import torch

from .common import MODEL_H, MODEL_W, composite, pick_refs  # noqa: F401  (재수출)
from .sttn.network_sttn import InpaintGenerator


class STTNEngine:
    def __init__(self, model_path, device="cuda", group_size=5, n_refs=8, ref_span=600,
                 cache_size=64):
        self.device = torch.device(device)
        self.model = InpaintGenerator().to(self.device)
        sd = torch.load(model_path, map_location="cpu")
        self.model.load_state_dict(sd["netG"] if "netG" in sd else sd)
        self.model.eval()
        self.group_size = group_size
        self.n_refs = n_refs
        self.ref_span = ref_span
        # 참조 프레임은 그룹이 넘어가도 상당수 재사용되므로 인코딩 결과를 캐시한다.
        # 프레임당 6.6MB이므로 64개면 약 420MB.
        self.cache_size = cache_size
        self._cache = {}
        self._dummy_mask = None

    def _encode(self, idxs, get_small, get_mask):
        """필요한 프레임만 인코딩하고 캐시. 반환: [len(idxs),256,60,108]"""
        todo = [i for i in idxs if i not in self._cache]
        if todo:
            imgs = np.stack([cv2.cvtColor(get_small(i), cv2.COLOR_BGR2RGB) for i in todo])
            msks = np.stack([get_mask(i) for i in todo])[:, None]
            x = torch.from_numpy(imgs).permute(0, 3, 1, 2).float().div_(255).mul_(2).sub_(1)
            m = torch.from_numpy(msks).float()
            x, m = x.to(self.device), m.to(self.device)
            with torch.no_grad():
                feats = self.model.encoder(x * (1 - m))
            for k, i in enumerate(todo):
                self._cache[i] = feats[k]
            del x, m, feats

        out = torch.stack([self._cache[i] for i in idxs])

        # 축출은 결과를 뽑은 뒤에. 이번 호출에 쓰인 프레임은 절대 버리지 않는다.
        keep = set(idxs)
        overflow = len(self._cache) - max(self.cache_size, len(keep))
        if overflow > 0:
            for i in [k for k in self._cache if k not in keep][:overflow]:
                del self._cache[i]
        return out

    @torch.no_grad()
    def process_group(self, group, total, get_small, get_mask, mask_area):
        """group(연속 프레임 인덱스)의 인페인팅 결과를 432x240 BGR 리스트로 반환."""
        # 마지막 그룹은 group_size보다 짧을 수 있다. 마지막 프레임을 복제해 채워
        # t를 항상 group_size + n_refs로 유지하고 결과에서 여분을 버린다.
        want = len(group)
        group = list(group) + [group[-1]] * (self.group_size - len(group))

        refs = pick_refs(group[len(group) // 2], total, self.n_refs, self.ref_span,
                         mask_area, exclude=set(group))
        idxs = list(group) + refs

        feats = self._encode(idxs, get_small, get_mask)

        # infer()에 넘기는 마스크는 값이 무의미하다. network_sttn.py:149의
        # `scores.masked_fill(m, -1e9)`가 대입도 in-place도 아니라 결과를 버리기 때문.
        # (tools/check_maskfill.py에서 최대차 0.000e+00 확인. 원본 researchmm/STTN부터
        #  그렇고 모델이 그 상태로 학습됐으므로 고치지 않는다.)
        # 실제 마스킹은 _encode()의 feats*(1-mask)에서 이미 끝났다. shape만 맞으면 된다.
        if self._dummy_mask is None or self._dummy_mask.shape[0] != len(idxs):
            self._dummy_mask = torch.zeros(len(idxs), 1, MODEL_H, MODEL_W, device=self.device)

        enc = self.model.infer(feats, self._dummy_mask)
        out = torch.tanh(self.model.decoder(enc[: self.group_size]))
        out = out.add_(1).div_(2).clamp_(0, 1).mul_(255)
        out = out.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)  # RGB

        del feats, enc
        return [cv2.cvtColor(o, cv2.COLOR_RGB2BGR) for o in out[:want]]

    def reset_cache(self):
        self._cache.clear()
