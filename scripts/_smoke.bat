@echo off
setlocal EnableExtensions
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
cd /d C:\Users\95151\Documents\Python_Projects\20260521_HDR_GM_JPG

echo === dumpbin /dependents uhdr.dll ===
dumpbin /dependents uhdr.dll
echo.
echo === dumpbin /dependents jpeg62.dll ===
dumpbin /dependents jpeg62.dll
echo.
echo === ultrahdr_app.exe (no args = usage) ===
libultrahdr\build\ultrahdr_app.exe
echo --- ultrahdr_app exit=%ERRORLEVEL%
echo.
echo === python ctypes load test ===
uv run python -c "import uhdr_ctypes; print('OK sizeof', __import__('ctypes').sizeof(uhdr_ctypes.UhdrRawImage))"
echo --- python exit=%ERRORLEVEL%
