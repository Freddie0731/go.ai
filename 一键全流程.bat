@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
cd /d D:\GoAI3\GoAI3\GoAI3

rem ---------------------------------------------------------------------------
rem 源码在 D:\GoAI3\GoAI3\GoAI3\ ，但 Visual Studio 把可执行文件输出到上一层：
rem     D:\GoAI3\GoAI3\x64\Debug\GoAI3.exe
rem nn_weights.bin 也在那里（引擎从"当前工作目录"找权重）。
rem 所以这里必须编译到同一个目录，否则你运行的还是旧 exe。
rem ---------------------------------------------------------------------------
set "SRCDIR=%~dp0"
set "OUTDIR=D:\GoAI3\GoAI3\x64\Debug"
set "LOG=%~dp0一键结果.txt"
set "VCVARS="

> "%LOG%" echo GoAI3 构建与自检
>>"%LOG%" echo 时间: %DATE% %TIME%
>>"%LOG%" echo 源码: %SRCDIR%
>>"%LOG%" echo 输出: %OUTDIR%
>>"%LOG%" echo.

echo ============================================
echo   GoAI3 构建 + 自检
echo ============================================
echo  源码: %SRCDIR%
echo  输出: %OUTDIR%
echo  日志: %LOG%
echo.

if not exist "%OUTDIR%" (
  echo [错误] 输出目录不存在：%OUTDIR%
  echo        先在 Visual Studio 里编译过一次，或手动创建该目录。
  >>"%LOG%" echo [错误] 输出目录不存在
  goto :end
)

set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if exist "%VSWHERE%" (
  for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2^>nul`) do (
    if exist "%%i\VC\Auxiliary\Build\vcvars64.bat" set "VCVARS=%%i\VC\Auxiliary\Build\vcvars64.bat"
  )
)

if not defined VCVARS (
  echo [错误] 找不到 MSVC（vswhere 没结果）。
  echo        请用 Visual Studio 打开 D:\GoAI3\GoAI3\GoAI3.slnx 按 F7 编译，
  echo        编译完再回来运行本脚本的自检部分。
  >>"%LOG%" echo [错误] 找不到 MSVC
  goto :end
)

echo [1/4] 载入 MSVC 环境 ...
>>"%LOG%" echo ===== MSVC =====
>>"%LOG%" echo %VCVARS%
call "%VCVARS%" >>"%LOG%" 2>&1

echo [2/4] 编译 go_ai.cpp -^> %OUTDIR%\GoAI3.exe ...
>>"%LOG%" echo.
>>"%LOG%" echo ===== 编译 =====
pushd "%OUTDIR%"
cl /nologo /O2 /EHsc /std:c++17 /W3 /DNDEBUG /Fe:GoAI3.exe "%SRCDIR%go_ai.cpp" /link gdiplus.lib >>"%LOG%" 2>&1
set "BLD=!ERRORLEVEL!"
popd
echo      编译返回码：!BLD!
>>"%LOG%" echo 编译返回码: !BLD!

if not "!BLD!"=="0" (
  echo.
  echo [失败] 编译没过。错误已经写进日志，请让助手读 %LOG%
  goto :end
)

echo [3/4] 自检 ...
>>"%LOG%" echo.
>>"%LOG%" echo ===== 自检 =====
pushd "%OUTDIR%"
GoAI3.exe --selftest >>"%LOG%" 2>&1
set "RC=!ERRORLEVEL!"
echo.
GoAI3.exe --selftest
echo.
echo      自检返回码：!RC!
>>"%LOG%" echo 自检返回码: !RC!
echo.
echo [4/4] 快速基准（约 15 秒，用来确认搜索真的能出着法）...
>>"%LOG%" echo.
>>"%LOG%" echo ===== 基准 =====
GoAI3.exe --benchmark >>"%LOG%" 2>&1
echo.
GoAI3.exe --benchmark
popd

:end
echo.
echo ============================================
echo   完整日志：%LOG%
echo   回来跟助手说「读一键结果.txt」即可，不用复制粘贴。
echo ============================================
echo.
pause
exit /b 0
