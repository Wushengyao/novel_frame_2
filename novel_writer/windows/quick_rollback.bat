@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0quick_rollback.ps1" %*
exit /b %ERRORLEVEL%
