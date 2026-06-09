@echo off
cd /d "%~dp0"

set PY=D:\miniconda\envs\stock\pythonw.exe
if exist "%PY%" goto :run

set PY=%USERPROFILE%\miniconda3\envs\stock\pythonw.exe
if exist "%PY%" goto :run

set PY=%LOCALAPPDATA%\miniconda3\envs\stock\pythonw.exe
if exist "%PY%" goto :run

set PY=D:\ProgramData\miniconda3\envs\stock\pythonw.exe
if exist "%PY%" goto :run

set PY=C:\miniconda3\envs\stock\pythonw.exe
if exist "%PY%" goto :run

set PY=D:\miniconda\envs\stock\python.exe
if exist "%PY%" goto :run

echo [ERROR] Could not find stock conda python
pause
exit /b 1

:run
start "" "%PY%" "%~dp0start_trade_advisor.py"
echo Started. Close this window or press any key.
pause
