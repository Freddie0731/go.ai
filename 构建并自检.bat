@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
cd /d "%~dp0"

set "LOG=%~dp0build_log.txt"
set "VCVARS="

> "%LOG%" echo GoAI3 build log
>>"%LOG%" echo date: %DATE% %TIME%
>>"%LOG%" echo dir : %~dp0
>>"%LOG%" echo.

echo ============================================
echo   GoAI3 - build ^& self test
echo ============================================
echo.
echo Everything below is also written to:
echo   %LOG%
echo.

set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"

if exist "%VSWHERE%" (
  for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2^>nul`) do (
    if exist "%%i\VC\Auxiliary\Build\vcvars64.bat" set "VCVARS=%%i\VC\Auxiliary\Build\vcvars64.bat"
  )
)

if defined VCVARS (
  echo [1/3] compiler: MSVC
  echo         !VCVARS!
  >>"%LOG%" echo [compiler] MSVC
  >>"%LOG%" echo vcvars: !VCVARS!
  >>"%LOG%" echo.
  call "!VCVARS!" >>"%LOG%" 2>&1
  echo [2/3] compiling ...
  cl /nologo /O2 /EHsc /std:c++17 /W3 /DNDEBUG /Fe:GoAI3.exe go_ai.cpp /link gdiplus.lib gdi32.lib user32.lib >>"%LOG%" 2>&1
  set "BLD=!ERRORLEVEL!"
) else (
  where g++ >nul 2>nul
  if errorlevel 1 (
    echo [ERROR] No compiler found.
    >>"%LOG%" echo [ERROR] Neither MSVC ^(vswhere^) nor g++ found.
    echo         Install "Desktop development with C++" or add MinGW g++ to PATH.
    goto :end
  )
  echo [1/3] compiler: MinGW g++
  >>"%LOG%" echo [compiler] MinGW g++
  >>"%LOG%" echo.
  echo [2/3] compiling ...
  g++ -O2 -std=c++17 -pthread go_ai.cpp -o GoAI3.exe -lgdiplus -lgdi32 -luser32 -static >>"%LOG%" 2>&1
  set "BLD=!ERRORLEVEL!"
)

rem ---------------------------------------------------------------------------
rem 不要信任 %ERRORLEVEL%：它可能被上面的 call 覆盖成 0，导致编译失败却报成功。
rem 因此这里做两道判断：先看退出码，再看产物文件是否真的生成。
rem ---------------------------------------------------------------------------
set "PRODUCED=1"
if not exist "GoAI3.exe" set "PRODUCED=0"

if not "!BLD!"=="0" (
  echo.
  echo [FAILED] compile returned !BLD!
  >>"%LOG%" echo.
  >>"%LOG%" echo [FAILED] compile exit code !BLD!
  goto :end
)

if "!PRODUCED!"=="0" (
  echo.
  echo [FAILED] compile reported success but GoAI3.exe was not created.
  echo          Common cause: the old GoAI3.exe is still running and locked, so
  echo          the linker cannot overwrite it. Close it or run:
  echo              taskkill /IM GoAI3.exe /F
  >>"%LOG%" echo.
  >>"%LOG%" echo [FAILED] GoAI3.exe missing after cl reported success.
  >>"%LOG%" echo Likely a locked/running GoAI3.exe. Try: taskkill /IM GoAI3.exe /F
  goto :end
)

echo [3/3] running self test ...
>>"%LOG%" echo.
>>"%LOG%" echo ===== self test =====
GoAI3.exe --selftest >>"%LOG%" 2>&1
set "RC=!ERRORLEVEL!"

echo.
echo ----- self test output -----
GoAI3.exe --selftest
echo ----------------------------
echo.
if not "!RC!"=="0" (
  echo [WARN] self test reported failures ^(exit !RC!^)
) else (
  echo [OK] build + self test finished
)

:end
echo.
echo ============================================
echo   Full log: %LOG%
echo   Tell the assistant "read build_log.txt" - no copy/paste needed.
echo ============================================
echo.
pause
exit /b 0
