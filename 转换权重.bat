@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo ============================================
echo   权重转换 - nn_weights_*.bin -> 引擎可用格式
echo ============================================
echo.

set "PY="
for %%P in (
  "C:\Users\小米\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe"
  "C:\Users\小米\AppData\Local\Programs\Python\Python313\python.exe"
  "C:\Users\小米\AppData\Local\Programs\Python\Python312\python.exe"
  "C:\Users\小米\AppData\Local\Programs\Python\Python311\python.exe"
) do if not defined PY if exist %%P set "PY=%%~P"
if not defined PY for /f "delims=" %%P in ('where python 2^>nul') do if not defined PY set "PY=%%P"

if not defined PY (
  echo [错误] 找不到 Python 解释器。
  pause
  exit /b 1
)
echo Python: %PY%
echo.

set "SRC=C:\Users\小米\Desktop\goai\nn_weights_v3.bin"
if not exist "%SRC%" set "SRC=C:\Users\小米\Desktop\goai\nn_weights_v2.bin"
if not exist "%SRC%" (
  echo [错误] 找不到输入权重：
  echo   C:\Users\小米\Desktop\goai\nn_weights_v3.bin
  echo   C:\Users\小米\Desktop\goai\nn_weights_v2.bin
  pause
  exit /b 1
)

echo 输入：%SRC%
echo 输出：%~dp0nn_weights.bin
echo.
"%PY%" "%~dp0转换权重.py" "%SRC%" "%~dp0nn_weights.bin"
set "RC=%ERRORLEVEL%"
echo.
if not "%RC%"=="0" (
  echo [注意] 转换或自检未通过（返回码 %RC%），请看上面的输出。
) else (
  echo [完成] nn_weights.bin 已就位，引擎会直接加载它。
)
echo.
echo 下一步：跑「构建并自检.bat」，看 "network I/O contract" 一行是否为 ok。
echo.
pause
