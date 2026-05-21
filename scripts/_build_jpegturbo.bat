@echo off
setlocal EnableExtensions
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
if errorlevel 1 (echo vcvars64.bat failed& exit /b 1)

set PATH=C:\Program Files\NASM;%PATH%
set ROOT=C:\Users\95151\Documents\Python_Projects\20260521_HDR_GM_JPG
set INSTALL=C:\msvcinstalls

echo === nasm version ===
nasm -v
if errorlevel 1 (echo nasm not found on PATH& exit /b 1)

if not exist "%ROOT%\libjpeg-turbo" (
  echo === cloning libjpeg-turbo ===
  cd /d "%ROOT%"
  git clone --depth 1 https://github.com/libjpeg-turbo/libjpeg-turbo.git
  if errorlevel 1 (echo git clone failed& exit /b 1)
)

cd /d "%ROOT%\libjpeg-turbo"
if exist build rmdir /s /q build

echo === configuring libjpeg-turbo (Ninja, Release) ===
cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="%INSTALL%" -S . -B build
if errorlevel 1 (echo configure failed& exit /b 1)

echo === building libjpeg-turbo ===
cmake --build build --config Release --target install
if errorlevel 1 (echo build/install failed& exit /b 1)

echo === verifying install ===
if exist "%INSTALL%\bin\jpeg62.dll" (echo OK: jpeg62.dll present) else (echo MISSING: %INSTALL%\bin\jpeg62.dll & exit /b 1)
dir /b "%INSTALL%\bin"
dir /b "%INSTALL%\lib"
echo === DONE ===
