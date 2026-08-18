# hardsub_eraser · 하드섭 지우개

[![release](https://img.shields.io/github/v/release/Cench-k/hardsub-eraser)](https://github.com/Cench-k/hardsub-eraser/releases/latest)
[![license](https://img.shields.io/badge/license-GPLv3-blue)](LICENSE)
![platform](https://img.shields.io/badge/windows-x64-lightgrey)

영상에 박힌 하드코딩 자막을 지우는 로컬 도구. 100% 오프라인, API 키 불필요.
AniEraser(media.io) 같은 인페인팅 도구를 하드섭 제거에 특화해 만든 것.

> **[포터블 판 내려받기](https://github.com/Cench-k/hardsub-eraser/releases/latest)** (264MB)
> — 압축 풀고 `실행.bat`. Python도 ffmpeg도 설치할 필요 없습니다.
> 경로가 너무 깊으면 Windows 260자 제한에 걸리니 바탕화면이나 `C:\` 근처에 푸세요.

NVIDIA·AMD·Intel GPU 모두 지원하고, GPU가 없어도 CPU로 동작합니다.
VRAM 크기에 맞춰 설정을 알아서 맞춥니다.

**라이선스 GPLv3** ([LICENSE](LICENSE), 이유는 [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md))

## 쓰는 법

**포터블 판** — 압축을 풀고 `실행.bat`. Python도 ffmpeg도 설치할 필요 없다.

**소스에서 실행:**

```powershell
start.bat                                          # 서버 + 브라우저
.\.venv\Scripts\python.exe -m hse.cli in.mp4 -o out.mp4
.\.venv\Scripts\python.exe -m hse.cli in.mp4 --region 0.6,0.75
```

자막이 화면 어디 있는지 모르면 UI의 **자막 자동 탐색**을 쓰면 구역을 잡아준다.
CLI 기본값은 하단 30%(`0.70,1.0`)인데, 세로 숏드라마는 자막이 화면 65% 근처에
오는 경우가 많아 `--region`을 직접 줘야 한다.

## 설치 (소스에서)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip uninstall -y onnxruntime
.\.venv\Scripts\python.exe -m pip install onnxruntime-directml

.\.venv\Scripts\python.exe tools\fetch_models.py     # ONNX 모델 (67MB)
```

`onnxruntime`을 지웠다 다시 까는 이유는 `requirements.txt` 주석 참고.
ffmpeg는 PATH에 있어야 한다.

**모델 가중치는 저장소에 없다.** Git LFS는 무료 한도가 스토리지·대역폭 각 1GB뿐이라
배포가 시작되면 과금되지만, Releases 첨부는 무료이고 대역폭 제한도 없기 때문이다.
`tools/fetch_models.py`가 최신 릴리스에서 받아온다.

직접 만들려면 torch와 원본 `sttn.pth`가 필요하다
([VSR](https://github.com/YaoFANGUK/video-subtitle-remover)의 `backend/models/sttn-det/`):

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe tools\export_onnx.py
```

## 배포판 만들기

```powershell
.\.venv\Scripts\python.exe tools\build_release.py
```

임베디드 Python + 의존성 + ffmpeg + ONNX 모델을 묶어 `dist/`에 포터블 ZIP을 만든다
(폴더 594MB, ZIP 263MB). torch는 넣지 않는다.

빌드 끝에 `tools/verify_release.py`가 **배포본 자신의 파이썬으로** 자동 실행되어
의존성·provider·모델·ffmpeg·전체 파이프라인을 확인한다. 실패하면 빌드가 중단된다.
개발 환경이 섞여 통과하는 일을 막으려고 `PYTHONPATH`/`PYTHONHOME`/`VIRTUAL_ENV`를
비우고 실행한다.

빌드하며 밟은 함정들:

- **설치 순서.** `rapidocr-onnxruntime`이 CPU 전용 `onnxruntime`을 의존성으로
  끌고 온다. 먼저 나머지를 깔고 → `onnxruntime` 제거 → `onnxruntime-directml` 설치
  순서를 지켜야 한다. 순서를 어기면 uninstall이 공유 디렉터리를 지워놓고 pip
  메타데이터는 남아, 재설치가 no-op이 되면서
  `ImportError: cannot import name 'GraphOptimizationLevel' from 'onnxruntime' (unknown location)`
  이 난다. 그래서 빌드가 provider를 직접 확인하고 없으면 중단한다.
- **경로 기본값.** 모델 경로를 상대 경로로 두면 작업 디렉터리가 앱 폴더가 아닐 때
  ONNX를 못 찾고 조용히 torch 백엔드로 넘어간다. 배포판엔 torch가 없어 그대로
  실패한다. `hse/pipeline.py`가 패키지 위치 기준 절대 경로를 쓴다.
- **긴 경로.** 압축을 아주 깊은 경로에 풀면 Windows 260자 제한에 걸린다
  (`Expand-Archive` 실패). `C:\` 근처나 바탕화면 정도에 푸는 것을 권한다.

모델 파일을 GitHub에 올릴 때 **Git LFS는 쓰지 말 것** — 무료 한도가 스토리지·대역폭
각 1GB뿐이라 배포가 시작되면 과금된다. **Releases 첨부**는 무료이고 대역폭 제한도 없다.

## 구성

| 역할 | 기술 |
|---|---|
| 자막 감지 | rapidocr-onnxruntime (DBNet, CPU) |
| 영상 인페인팅 | STTN — ONNX 또는 PyTorch |
| 실행 백엔드 | onnxruntime: CUDA / DirectML / CPU 자동 선택 |
| 디먹스·리먹스 | ffmpeg (원본 오디오 무손실 복사) |
| UI | FastAPI + 바닐라 HTML (빌드 단계 없음) |

```
hse/common.py       백엔드 무관 공통부 (torch를 import하지 않는다)
hse/engine.py       PyTorch 엔진 — ONNX 내보내기의 기준 구현
hse/engine_onnx.py  ONNX 엔진 — 배포 실행 경로
hse/pipeline.py     3패스 파이프라인
hse/server.py       로컬 웹 API
web/index.html      UI 전체 (단일 파일)
```

## 동작 방식

**3패스.** ① 자막 감지 → ② 처리 밴드를 432×240으로 줄여 memmap에 캐시 → ③ 그룹 단위
인페인팅 + 원본 해상도 합성 + 리먹스. ②가 있는 이유는 참조 프레임을 영상 전체
아무 데서나 뽑으려면 임의 접근이 필요한데 영상 디코딩은 순차적이기 때문이다.

**참조 프레임 희소 샘플링.** 트랜스포머에 들어가는 프레임 수를 상수로 고정한다.

```
t = group_size(출력) + n_refs(참조)
```

메모리가 t에만 비례하므로 참조를 수백 프레임 밖에서 가져와도 비용이 같다.
참조는 시간축으로 고르게 나눈 구간마다 **마스크 면적이 가장 작은**(배경이 가장 많이
드러난) 프레임을 고른다.

## 자동 설정

`--group-size`/`--n-refs`를 주지 않으면 이 기기에 맞춰 알아서 정한다
([hse/autotune.py](hse/autotune.py)).

```
GPU 메모리 3072 MB (nvidia-smi) -> 우선 시도 group=5 refs=8 (예상 2710 MB)
  시험 group=5 refs=8: 0.249 s/frame  OK
  시험 group=4 refs=5: 0.365 s/frame  OK
자동 설정: group=5 refs=8 (0.25s/frame)
```

VRAM 용량만 보고 정하지 않고 **실제로 두 그룹을 돌려 속도를 잰다.** Windows는
VRAM이 모자라면 OOM을 내는 대신 시스템 램으로 넘겨 조용히 10배 느려지기 때문에,
예외로는 잡히지 않고 속도로만 드러난다. 한 단계 낮은 설정이 확연히 빠를 때만
갈아탄다(참조를 늘려도 품질 이득은 작고 속도 손해는 크다). 결과는
`~/.hardsub_eraser/autotune.json`에 캐시된다.

VRAM 탐지는 nvidia-smi → torch → Windows 레지스트리 순으로 시도한다.
WMI의 `Win32_VideoController.AdapterRAM`은 32비트라 4GB에서 잘리므로 쓰지 않고,
레지스트리의 `qwMemorySize`를 읽는다.

## 측정 (GTX 1060 3GB, 1080×1920 300프레임)

| 구성 | 속도 | 피크 VRAM | PSNR | SSIM |
|---|---|---|---|---|
| Phase 0 (청크 방식) | 2.6 fps | 784 MB | — | — |
| n_refs=4 | 5.4 fps | 1009 MB | 23.49 dB | 0.8657 |
| **n_refs=8 (자동 선택값)** | **4.1 fps** | **1680 MB** | **23.78 dB** | **0.8673** |

PSNR/SSIM은 `tools/eval_quality.py`로 잰 값이다. 자막 없는 구간에 자막을 합성해 넣고
지운 뒤 원본과 비교하므로 정답이 있는 평가다. 눈대중 대신 이걸로 튜닝한다.

**백엔드 비교** (같은 영상, 같은 PSNR/SSIM):

| 백엔드 | 속도 | 배포 용량 | 지원 |
|---|---|---|---|
| PyTorch CUDA | 4.5 fps | 약 3GB | NVIDIA 전용 |
| ONNX DirectML | 4.0 fps | 약 300MB | NVIDIA·AMD·Intel |

출력 차이는 평균 픽셀 0.265 — h.264 재인코딩 노이즈(2~4)보다 작다.
ONNX 변환 오차는 encoder 5.0e-6, transformer 1.0e-4, decoder 9.5e-5.

**그래프를 셋으로 나눈 효과.** 처음엔 트랜스포머와 디코더를 한 그래프로 묶었는데,
그때는 t=17만 돼도 공유 메모리로 밀려나 5.7 s/frame으로 무너졌다. 디코더를 떼어내
참조 프레임을 디코딩하지 않게 하자 t=32도 1.18 s/frame으로 버틴다.
t가 동적이라 설정마다 그래프를 따로 실을 필요도 없다.

## 알아둘 것

### 어텐션 마스크는 무효다
`hse/sttn/network_sttn.py:149`의 `scores.masked_fill(m, -1e9)`는 대입도 in-place도
아니라 결과가 버려진다. 즉 어텐션 마스크가 출력에 아무 영향이 없다
(`tools/check_maskfill.py`에서 최대차 0.000e+00 확인, ONNX 익스포터도 이 입력을
죽은 코드로 제거했다). 원본 researchmm/STTN부터 그런 코드이고 **모델이 그 상태로
학습됐으므로 고치면 안 된다.** 실제 마스킹은 인코더 입력의 `feats*(1-mask)`에서 일어난다.

### rapidocr 기본값이 이미지를 확대한다
기본값 `limit_type='min'`, `limit_side_len=736`은 **짧은** 변을 736까지 늘린다.
852×144 자막 밴드가 약 4350×736이 되어 프레임당 719ms가 걸렸다. `'max'`로 바꾸면
**28ms (26배)**, 감지 결과는 동일. 밴드를 얇게 자를수록 느려지는 역설이 생긴다.

### 워터마크가 자막보다 자주 잡힌다
"텍스트가 제일 많이 나온 높이"로 자막 위치를 찾으면 실패한다. 로고·채널 워터마크는
매 프레임 나오기 때문이다(실측: 워터마크 16회 vs 자막 14회로 뒤집힘).
`/api/jobs/{id}/scan`은 박스를 IoU로 묶은 뒤 **폭과 가로 위치가 거의 변하지 않는**
클러스터를 고정 오버레이로 보고 제외한다. 자막은 글자 수에 따라 폭이 변한다.

### 마스크 영역만 합성한다
[VSR 원본](https://github.com/YaoFANGUK/video-subtitle-remover)은 432×240으로
축소했다 확대한 밴드를 통째로 덮어써서 자막과 무관한 픽셀까지 뭉갠다.
여기서는 마스크 픽셀만 교체한다. `tools/check_diff.py` 검증:

```
밴드 내 자막 없는 구간  최대차 14  평균 3.47
밴드 밖 (미처리 영역)   최대차 15  평균 3.23   <- 재인코딩 노이즈 기준선
```

통계적으로 동일 = 화질 손실 없음.

## 한계

- 같은 자막이 오래 머무르고 배경이 움직이면 옅은 얼룩이 남는다. 참조 구간을 넓혀
  많이 줄였지만 완전히 없애진 못했다.
- 처리 시간은 실시간의 약 7배. 1분 30fps 영상에 약 7분.
- 배경이 격하게 움직이는 구간은 번짐이 생길 수 있다(STTN 공통 한계).

## 도구

```
tools/eval_quality.py       정답 있는 PSNR/SSIM 평가 (파라미터 튜닝용)
tools/export_onnx.py        ONNX 재생성 + 수치 동등성 검증
tools/build_release.py      포터블 배포본 빌드
tools/verify_release.py     배포본 스모크 테스트 (빌드가 자동 실행)
tools/check_autotune.py     VRAM 탐지·추천·실측 확인
tools/check_maskfill.py     어텐션 마스크 무효 확인
tools/check_diff.py         마스크 밖 픽셀 보존 확인
tools/compare_videos.py     두 출력 영상 비교
tools/bench_det.py          감지 속도 측정
tools/try_dynamic_export.py 동적 t 내보내기 가능 여부 확인 (조사용)
```

## 다음

- 이미지 모드 (LaMa, 가중치는 이미 `models/big-lama/`에 있음 — ONNX 변환 필요)
- 브러시로 임의 영역 마스킹 (현재는 구역 밴드만)
- 움직이는 객체 추적 (SAM2-tiny)
- macOS / Linux 패키징 (현재 build_release.py는 Windows 전용)
