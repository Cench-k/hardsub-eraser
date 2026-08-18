@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

REM Korean text is intentionally NOT used in this file.
REM cmd.exe reads .bat in the OEM codepage (949 on Korean Windows),
REM so UTF-8 Korean here becomes mojibake and cmd tries to run it.
REM Keep this file pure ASCII with CRLF line endings.

set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" goto nopy

where ffmpeg >nul 2>&1
if errorlevel 1 goto noffmpeg

echo Starting hardsub eraser ... http://127.0.0.1:8756
start "" http://127.0.0.1:8756
"%PY%" -m hse.server
goto done

:nopy
echo [!] .venv not found. Run setup first:
echo       py -3.11 -m venv .venv
echo       .venv\Scripts\python.exe -m pip install -r requirements.txt
echo       .venv\Scripts\python.exe -m pip uninstall -y onnxruntime
echo       .venv\Scripts\python.exe -m pip install onnxruntime-directml
pause
exit /b 1

:noffmpeg
echo [!] ffmpeg not found in PATH. Install ffmpeg and try again.
pause
exit /b 1

:done
