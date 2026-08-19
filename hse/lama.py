"""LaMa 기반 이미지 인페인팅.

STTN 은 영상 모델이다. 여러 프레임의 배경을 참조해 채우는 게 강점이라 사진 한 장에는
참조할 게 없어 주변 픽셀만으로 추론한다. 사진은 LaMa 가 맞다.

LaMa 는 ONNX 로 못 내보낸다. FFC(푸리에 합성곱) 블록이 aten::fft_rfftn 을 쓰는데
ONNX 익스포터가 지원하지 않고, 배포된 가중치가 TorchScript 라 dynamo 익스포터도
받지 못한다(ScriptModule 미지원). 그래서 이 경로만 torch 를 쓴다.
torch 가 없으면 파이프라인이 STTN 으로 넘어간다.

마스크 주변만 잘라서 처리한다. 전체 해상도를 그대로 넣으면 큰 사진에서 메모리가
터지고, 어차피 필요한 건 마스크 근처뿐이다.
"""

import cv2
import numpy as np

PAD_TO = 8          # LaMa 는 8의 배수 입력을 요구한다
CONTEXT = 96        # 마스크 주변으로 확보할 문맥 여유(px)
MAX_SIDE = 1600     # 잘라낸 조각의 긴 변 상한. 넘으면 줄여서 처리하고 되돌린다


def available():
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


class LamaInpainter:
    def __init__(self, model_path, device=None):
        import torch
        if device in (None, "auto"):
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch = torch
        self.device = torch.device(device)
        self.model = torch.jit.load(model_path, map_location=self.device).eval()

    def _run(self, img_bgr, mask01):
        """잘라낸 조각 하나를 처리. 입력/출력 모두 BGR uint8."""
        torch = self.torch
        h, w = img_bgr.shape[:2]
        ph = (PAD_TO - h % PAD_TO) % PAD_TO
        pw = (PAD_TO - w % PAD_TO) % PAD_TO
        if ph or pw:
            img_bgr = cv2.copyMakeBorder(img_bgr, 0, ph, 0, pw, cv2.BORDER_REFLECT)
            mask01 = cv2.copyMakeBorder(mask01, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=0)

        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = torch.from_numpy(rgb).permute(2, 0, 1)[None].to(self.device)
        m = torch.from_numpy((mask01 > 0).astype(np.float32))[None, None].to(self.device)

        with torch.inference_mode():
            out = self.model(x, m)
        out = out[0].permute(1, 2, 0).float().cpu().numpy()
        out = np.clip(out * 255, 0, 255).astype(np.uint8)
        return cv2.cvtColor(out[:h, :w], cv2.COLOR_RGB2BGR)

    def __call__(self, img_bgr, mask01, feather=3):
        """마스크 영역만 채워 넣은 이미지를 돌려준다."""
        if not mask01.any():
            return img_bgr
        H, W = img_bgr.shape[:2]
        ys, xs = np.where(mask01 > 0)
        y0 = max(0, int(ys.min()) - CONTEXT)
        y1 = min(H, int(ys.max()) + 1 + CONTEXT)
        x0 = max(0, int(xs.min()) - CONTEXT)
        x1 = min(W, int(xs.max()) + 1 + CONTEXT)

        crop = img_bgr[y0:y1, x0:x1]
        cmask = mask01[y0:y1, x0:x1]

        # 조각이 너무 크면 줄여서 처리하고 되돌린다. 메모리 상한을 지키기 위한 것.
        ch, cw = crop.shape[:2]
        scale = min(1.0, MAX_SIDE / max(ch, cw))
        if scale < 1.0:
            small = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            smask = cv2.resize(cmask, (small.shape[1], small.shape[0]),
                               interpolation=cv2.INTER_NEAREST)
            filled = cv2.resize(self._run(small, smask), (cw, ch),
                                interpolation=cv2.INTER_CUBIC)
        else:
            filled = self._run(crop, cmask)

        # 마스크 픽셀만 교체한다. 나머지는 원본 그대로 둔다.
        alpha = cmask.astype(np.uint8)
        if feather > 0:
            alpha = cv2.dilate(alpha, np.ones((3, 3), np.uint8), iterations=feather)
            k = feather * 2 + 1
            alpha = cv2.GaussianBlur(alpha.astype(np.float32), (k, k), 0)
        alpha = np.clip(alpha, 0, 1).astype(np.float32)[:, :, None]

        out = img_bgr.copy()
        out[y0:y1, x0:x1] = (crop.astype(np.float32) * (1 - alpha)
                             + filled.astype(np.float32) * alpha).astype(np.uint8)
        return out
