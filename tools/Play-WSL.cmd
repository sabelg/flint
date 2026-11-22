@echo off
wsl.exe --cd "%~dp0." --exec bash ./run-wsl.sh %*
exit /b %ERRORLEVEL%
