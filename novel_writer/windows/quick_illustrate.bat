@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0quick_illustrate.ps1" %*
exit /b %ERRORLEVEL%
