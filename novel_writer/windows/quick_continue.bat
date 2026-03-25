@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0quick_continue.ps1" %*
exit /b %ERRORLEVEL%
