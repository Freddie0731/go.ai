@echo off
rem ============================================================
rem  GoAI3 - one-shot build + self test
rem
rem  All-ASCII content on purpose: cmd.exe reads .bat files using
rem  the system ANSI code page, so non-ASCII bytes here would be
rem  mis-decoded. The log is written as ANSI and then converted to
rem  UTF-8 by PowerShell, so compiler diagnostics containing CJK
rem  text are preserved for the assistant to read.
rem
rem  Output name is GoAI3_new.exe on purpose: the old GoAI3.exe may
rem  still be mapped by a running process, which makes the linker
rem  fail with "the process cannot access the file".
rem ============================================================
chcp 65001 >nul
setlocal enabledelayedexpansion
cd /d "%~dp0"

set "SRC=%~dp0go_ai.cpp"
set "OUTDIR=D:\GoAI3\GoAI3\x64\Debug"
set "LOG=%~dp0build_log.txt"
set "RAW=%~dp0build_log.raw.txt"
set "EXE=%OUTDIR%\GoAI3_new.exe"
set "VCVARS="

if not exist "%SRC%" (
  echo [FAILED] source not found: %SRC%
  goto :end
)
if not exist "%OUTDIR%" (
  echo [FAILED] output directory not found: %OUTDIR%
  goto :end
)

rem --- locate MSVC ---
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if exist "%VSWHERE%" (
  for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2^>nul`) do (
    if exist "%%i\VC\Auxiliary\Build\vcvars64.bat" set "VCVARS=%%i\VC\Auxiliary\Build\vcvars64.bat"
  )
)
if not defined VCVARS if exist "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" set "VCVARS=C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
if not defined VCVARS if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" set "VCVARS=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"

rem --- fresh log ---
> "%RAW%" echo GoAI3 build and self test
>>"%RAW%" echo date: %DATE% %TIME%
>>"%RAW%" echo source: %SRC%
>>"%RAW%" echo output: %EXE%

echo ============================================
echo   GoAI3 - build ^& self test
echo ============================================
echo   source: %SRC%
echo   output: %EXE%
echo   log   : %LOG%
echo.

if not defined VCVARS (
  where g++ >nul 2>nul
  if errorlevel 1 (
    echo [FAILED] no compiler found ^(neither MSVC via vswhere nor g++ on PATH^)
    >>"%RAW%" echo [FAILED] no compiler found
    goto :finish
  )
  echo [1/3] compiler: MinGW g++
  >>"%RAW%" echo [compiler] MinGW g++
  echo [2/3] compiling ...
  g++ -O2 -std=c++17 -pthread "%SRC%" -o "%EXE%" -lgdiplus -lgdi32 -luser32 -static >>"%RAW%" 2>&1
  set "BLD=!ERRORLEVEL!"
) else (
  echo [1/3] compiler: MSVC
  echo         !VCVARS!
  >>"%RAW%" echo [compiler] MSVC
  >>"%RAW%" echo vcvars: !VCVARS!
  call "!VCVARS!" >nul 2>&1
  echo [2/3] compiling ...
  cl /nologo /O2 /EHsc /std:c++17 /W3 /DNDEBUG /Fe:"%EXE%" "%SRC%" /link gdiplus.lib gdi32.lib user32.lib >>"%RAW%" 2>&1
  set "BLD=!ERRORLEVEL!"
)

echo       compiler exit code: !BLD!

rem --- do not trust the exit code: verify the artifact exists ---
if not exist "%EXE%" (
  echo.
  echo [FAILED] compilation did not produce %EXE%
  >>"%RAW%" echo.
  >>"%RAW%" echo [FAILED] compilation did not produce %EXE% ^(exit code !BLD!^)
  goto :finish
)

if not "!BLD!"=="0" (
  echo.
  echo [WARN] compiler reported exit code !BLD! but the executable exists.
)

echo [3/3] running self test ...
>>"%RAW%" echo.
>>"%RAW%" echo ===== self test =====
rem The engine looks for nn_weights.bin relative to the CURRENT DIRECTORY, and
rem the weights live next to the executable, so run it from there.
pushd "%OUTDIR%"
"%EXE%" --selftest >>"%RAW%" 2>&1
set "RC=!ERRORLEVEL!"
echo.

echo ----- self test output -----

"%EXE%" --selftest
set "RC=!ERRORLEVEL!"
popd
echo ----------------------------
echo       self test exit code: !RC!
>>"%RAW%" echo self test exit code: !RC!

if "!RC!"=="0" (
  echo.
  echo [OK] build and self test both succeeded
) else (
  echo.
  echo [FAILED] build succeeded but self test failed ^(exit !RC!^)
)

:finish
rem --- convert the ANSI log to UTF-8 so non-ASCII diagnostics survive ---
powershell -NoProfile -Command "Get-Content -LiteralPath '%RAW%' -Encoding Default | Set-Content -LiteralPath '%LOG%' -Encoding UTF8" >nul 2>&1

:end
echo.
echo ============================================
echo   Full log: %LOG%
echo   Tell the assistant "read build_log.txt"
echo ============================================
echo.
pause
exit /b 0
