@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
echo VCVARS_RC=%ERRORLEVEL%
echo --- where cmake ---
where cmake
echo --- where cl ---
where cl
echo --- where ninja ---
where ninja
echo --- cmake --version ---
cmake --version
echo --- cl version (stderr) ---
cl 2>&1 | findstr /R "Microsoft.*Compiler"
