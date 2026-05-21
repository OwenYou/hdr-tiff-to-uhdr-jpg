@echo off
setlocal EnableExtensions
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
if errorlevel 1 (echo vcvars64.bat failed& exit /b 1)

set PATH=C:\Program Files\NASM;%PATH%
set ROOT=C:\Users\95151\Documents\Python_Projects\20260521_HDR_GM_JPG
set INSTALL=C:\msvcinstalls

cd /d "%ROOT%\libultrahdr"
if exist build rmdir /s /q build

echo === configuring libultrahdr (Ninja, Release, against C:\msvcinstalls) ===
cmake -G Ninja ^
      -DCMAKE_BUILD_TYPE=Release ^
      -DCMAKE_PREFIX_PATH="%INSTALL%" ^
      -DCMAKE_INSTALL_PREFIX="%INSTALL%" ^
      -DBUILD_SHARED_LIBS=ON ^
      -DUHDR_BUILD_EXAMPLES=ON ^
      -DUHDR_BUILD_TESTS=OFF ^
      -DUHDR_WRITE_XMP=ON ^
      -S . -B build
if errorlevel 1 (echo configure failed& exit /b 1)

echo === building libultrahdr ===
cmake --build build --config Release
if errorlevel 1 (echo build failed& exit /b 1)

echo === build outputs ===
dir /b build\*.dll build\*.lib build\*.exe 2>nul
echo === DONE ===
