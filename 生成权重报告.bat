@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
cd /d "%~dp0"

set "OUT=%~dp0weights_report.txt"
set "PY="
for %%P in (
  "C:\Users\小米\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe"
  "C:\Users\小米\AppData\Local\Programs\Python\Python312\python.exe"
  "C:\Users\小米\AppData\Local\Programs\Python\Python311\python.exe"
  "C:\Users\小米\AppData\Local\Programs\Python\Python310\python.exe"
) do if not defined PY if exist %%P set "PY=%%~P"
if not defined PY for /f "delims=" %%P in ('where python 2^>nul') do if not defined PY set "PY=%%P"
if not defined PY for /f "delims=" %%P in ('where py 2^>nul') do if not defined PY set "PY=%%P"

echo ============================================
echo   GoAI weight inspection
echo ============================================
echo.
echo script dir : %~dp0
echo report     : %OUT%
echo python     : %PY%
echo.

if not exist "%~dp0inspect_weights.py" (
  echo [ERROR] inspect_weights.py is missing from %~dp0
  echo         Both files must sit in the same folder.
  goto :end
)

if not defined PY (
  echo [ERROR] No Python interpreter found.
  echo         Tell me this and I will use another approach.
  goto :end
)

echo Collecting ... this takes a few seconds.
echo.

> "%OUT%" echo GoAI weight inspection report
>>"%OUT%" echo python: %PY%
>>"%OUT%" echo.

>>"%OUT%" echo ===== nn_weights_v2.bin =====
"%PY%" "%~dp0inspect_weights.py" "C:\Users\小米\Desktop\goai\nn_weights_v2.bin" >>"%OUT%" 2>&1
>>"%OUT%" echo.
>>"%OUT%" echo ===== nn_weights_v3.bin =====
"%PY%" "%~dp0inspect_weights.py" "C:\Users\小米\Desktop\goai\nn_weights_v3.bin" >>"%OUT%" 2>&1
>>"%OUT%" echo.
>>"%OUT%" echo ===== nn_weights_v2.bin.pt =====
"%PY%" "%~dp0inspect_weights.py" "C:\Users\小米\Desktop\goai\nn_weights_v2.bin.pt" >>"%OUT%" 2>&1
>>"%OUT%" echo.
>>"%OUT%" echo ===== nn_weights_v3.bin.pt =====
"%PY%" "%~dp0inspect_weights.py" "C:\Users\小米\Desktop\goai\nn_weights_v3.bin.pt" >>"%OUT%" 2>&1

echo Done.
echo.
echo The report is here:
echo   %OUT%
echo.
echo It should have opened in Notepad. Use Ctrl+A then Ctrl+C to copy it,
echo or just tell me and I will read the file directly.

if exist "%OUT%" start "" notepad "%OUT%"

:end
echo.
pause
