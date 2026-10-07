@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

set "PY=C:\Users\小米\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe"
if not exist "%PY%" set "PY=python"

echo ============================================
echo   Inspect GoAI weight files
echo ============================================
echo.

"%PY%" inspect_weights.py "C:\Users\小米\Desktop\goai\nn_weights_v2.bin" "C:\Users\小米\Desktop\goai\nn_weights_v3.bin"
echo.
echo ------------------ .pt checkpoints ------------------
echo.
"%PY%" inspect_weights.py "C:\Users\小米\Desktop\goai\nn_weights_v2.bin.pt" "C:\Users\小米\Desktop\goai\nn_weights_v3.bin.pt"
echo.
echo ============================================
echo   Copy the whole output above and send it back.
echo ============================================
pause
