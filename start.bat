@echo off
REM 하드섭 지우개 실행. 서버를 띄우고 브라우저를 연다.
setlocal
cd /d "%~dp0"

set PY=.venv\Scripts\python.exe
if not exist "%PY%" (
  echo [!] .venv 가 없습니다. 먼저 설치하세요:
  echo     py -3.11 -m venv .venv
  echo     .venv\Scripts\python.exe -m pip install -r requirements.txt
  echo     .venv\Scripts\python.exe -m pip uninstall -y onnxruntime
  echo     .venv\Scripts\python.exe -m pip install onnxruntime-directml
  pause
  exit /b 1
)

where ffmpeg >nul 2>&1
if errorlevel 1 (
  echo [!] ffmpeg 를 PATH 에서 찾을 수 없습니다. 설치 후 다시 실행하세요.
  pause
  exit /b 1
)

start "" http://127.0.0.1:8756
"%PY%" -m hse.server
