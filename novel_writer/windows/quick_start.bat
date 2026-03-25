@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0quick_start.ps1" %*
exit /b %ERRORLEVEL%
