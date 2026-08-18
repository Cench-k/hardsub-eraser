# 서드파티 고지

hardsub_eraser는 **GNU General Public License v3.0** (`LICENSE`)으로 배포된다.
GPLv3을 택한 이유는 ffmpeg 때문이다 — 아래 참조.

## 모델 가중치

| 구성요소 | 출처 | 라이선스 |
|---|---|---|
| STTN (`models/sttn/sttn.pth`) | [researchmm/STTN](https://github.com/researchmm/STTN) | MIT |
| LaMa (`models/big-lama/big-lama.pt`) | [advimman/lama](https://github.com/advimman/lama) | Apache 2.0 |
| PP-OCR 텍스트 감지 | PaddleOCR (rapidocr 재배포판) | Apache 2.0 |

두 가중치와 STTN 네트워크 정의 코드는
[YaoFANGUK/video-subtitle-remover](https://github.com/YaoFANGUK/video-subtitle-remover)
(Apache 2.0)에서 가져왔다. 해당 저장소는 가중치를 GitHub 50MB 제한 때문에 분할해
배포하며, 이 프로젝트는 `models/`에 재조립해 둔다.

`hse/sttn/network_sttn.py`, `hse/sttn/spectral_norm.py`는 위 저장소에서 복사한 뒤
import 경로만 수정했다(Apache 2.0 §4에 따른 변경 고지).

## 파이썬 의존성

| 패키지 | 라이선스 |
|---|---|
| PyTorch, torchvision | BSD-3-Clause |
| onnxruntime / onnxruntime-directml | MIT |
| opencv-python | Apache 2.0 |
| numpy | BSD-3-Clause |
| rapidocr-onnxruntime | Apache 2.0 |
| tqdm | MPL-2.0 / MIT |

## ffmpeg — GPLv3을 택한 이유

이 프로젝트는 ffmpeg로 디먹스/리먹스와 H.264 인코딩을 한다.
일반적으로 배포되는 ffmpeg 풀 빌드(gyan.dev 등)는 **libx264(GPLv2+)** 등
GPL 구성요소를 포함하므로 **GPL 빌드**다. 이를 함께 배포하는 저작물은
GPL 조건을 따라야 한다.

따라서 이 프로젝트 전체를 GPLv3으로 공개한다. 이 선택의 결과:

- ffmpeg 바이너리를 그대로 동봉해도 된다.
- 소스를 공개해야 하며, 파생 저작물도 GPLv3이어야 한다.
- 상용 클로즈드소스 제품에 이 코드를 포함할 수 없다.

**클로즈드소스로 전환하려면** ffmpeg를 분리해야 한다. 선택지:
1. LGPL 빌드로 교체하고 libx264 대신 OpenH264(BSD, Cisco가 특허료 부담) 또는
   하드웨어 인코더(NVENC/QSV/AMF) 사용
2. ffmpeg를 동봉하지 않고 사용자가 직접 설치하게 하거나 첫 실행 시 내려받게 함

두 경우 모두 H.264 자체의 특허(MPEG-LA) 문제는 별도로 검토해야 한다.
