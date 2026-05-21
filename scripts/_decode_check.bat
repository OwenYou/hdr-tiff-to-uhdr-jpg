@echo off
setlocal EnableExtensions
cd /d C:\Users\95151\Documents\Python_Projects\20260521_HDR_GM_JPG\test_files
copy /Y ..\libultrahdr\build\ultrahdr_app.exe . >nul
copy /Y ..\uhdr.dll . >nul
copy /Y ..\jpeg62.dll . >nul
ultrahdr_app.exe -m 1 -j Still_2026-05-20_175538_1.7.1.ultrahdr.jpg
echo --- decode exit=%ERRORLEVEL%
dir /b Still_2026-05-20_175538_1.7.1.ultrahdr*
