@echo off
setlocal
cd /d %~dp0

REM Prefer py launcher (Windows)
where py >nul 2>nul
if %errorlevel%==0 (
  py -3 installer_gui_extended.py
  goto :eof
)

REM Fallback to python
where python >nul 2>nul
if %errorlevel%==0 (
  python installer_gui_extended.py
  goto :eof
)

echo Python not found. Install Python 3.10+ from https://www.python.org/downloads/
pause
