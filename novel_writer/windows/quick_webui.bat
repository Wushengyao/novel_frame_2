@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0quick_webui.ps1" %*
exit /b %ERRORLEVEL%
