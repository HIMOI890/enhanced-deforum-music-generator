@echo off
setlocal

echo Enhanced Deforum Music Generator
echo ==================================

REM Root folder of this script
set "ROOT=%~dp0"

REM Prefer venv python explicitly (most reliable)
set "PY=%ROOT%venv\Scripts\python.exe"

REM Activate venv if present (optional but helpful)
if exist "%ROOT%venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call "%ROOT%venv\Scripts\activate.bat"
) else (
    echo [WARN] venv not found at "%ROOT%venv"
)

REM Load .env key=value lines (best-effort)
if exist "%ROOT%.env" (
    for /f "usebackq delims=" %%A in ("%ROOT%.env") do (
        set "%%A"
    )
)

REM Ensure folders
if not exist "%ROOT%data" mkdir "%ROOT%data"
if not exist "%ROOT%data\models" mkdir "%ROOT%data\models"
if not exist "%ROOT%data\cache" mkdir "%ROOT%data\cache"
if not exist "%ROOT%data\logs" mkdir "%ROOT%data\logs"
if not exist "%ROOT%output" mkdir "%ROOT%output"
if not exist "%ROOT%output\packages" mkdir "%ROOT%output\packages"
if not exist "%ROOT%output\analysis" mkdir "%ROOT%output\analysis"
if not exist "%ROOT%output\previews" mkdir "%ROOT%output\previews"

echo.
echo Starting Gradio UI...
echo Using Python: "%PY%"
echo.

REM Always prefer venv python if it exists
if exist "%PY%" (
    "%PY%" -m enhanced_deforum_music_generator ui --port 7860
) else (
    echo [WARN] venv python not found. Falling back to PATH python.
    python -m enhanced_deforum_music_generator ui --port 7860
)

echo.
echo Application stopped.
pause
