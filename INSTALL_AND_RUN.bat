@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

echo ============================================================
echo ARKAAI - SNAPDRAGON FINAL ONE-CLICK SETUP
echo ============================================================
echo.
echo This package installs the complete ArkaAI software stack.
echo On Snapdragon Windows ARM64 it uses Qualcomm GenieX + AI Hub models
echo and runs locally. The installer downloads third-party runtime and
echo model assets as needed; the submission source stays together here.
echo.

set "PYEXE="
for /f "delims=" %%P in ('where py 2^>nul') do if not defined PYEXE set "PYEXE=%%P"
if not defined PYEXE for /f "delims=" %%P in ('where python 2^>nul') do if not defined PYEXE set "PYEXE=%%P"
if not defined PYEXE (
    echo Python not found. Trying winget...
    where winget >nul 2>nul
    if errorlevel 1 goto :python_missing
    winget install --id Python.Python.3.13 -e --accept-package-agreements --accept-source-agreements
    if errorlevel 1 goto :python_failed
    for /f "delims=" %%P in ('where python 2^>nul') do if not defined PYEXE set "PYEXE=%%P"
    if not defined PYEXE if exist "%LocalAppData%\Programs\Python\Python313\python.exe" set "PYEXE=%LocalAppData%\Programs\Python\Python313\python.exe"
    if not defined PYEXE if exist "%ProgramFiles%\Python313\python.exe" set "PYEXE=%ProgramFiles%\Python313\python.exe"
)
if not defined PYEXE goto :python_failed

if not exist ".venv\Scripts\python.exe" (
    echo Creating private Python environment...
    "%PYEXE%" -m venv .venv
    if errorlevel 1 goto :venv_failed
)
set "PYEXE=%CD%\.venv\Scripts\python.exe"
set "PIP_DISABLE_PIP_VERSION_CHECK=1"

echo.
echo [1/4] Installing application dependencies...
"%PYEXE%" -m pip install --upgrade pip
"%PYEXE%" -m pip install -r requirements_ArkaAI.txt
if errorlevel 1 goto :pip_failed

for /f "delims=" %%A in ('"%PYEXE%" -c "import platform; print(platform.machine())"') do set "PYARCH=%%A"
echo Python architecture: %PYARCH%
if /I "%PYARCH%"=="ARM64" (
    echo Installing on-device QNN support for semantic/vision helpers...
    "%PYEXE%" -m pip uninstall -y onnxruntime >nul 2>nul
    "%PYEXE%" -m pip install -U onnxruntime-qnn
    if errorlevel 1 echo WARNING: onnxruntime-qnn install failed; app will continue with compatible fallbacks.
)

echo.
echo [2/4] Installing Qualcomm GenieX and downloading local AI models...
"%PYEXE%" setup_models.py
if errorlevel 1 goto :model_failed

echo.
echo [3/4] Verifying all ArkaAI Python modules...
"%PYEXE%" -m py_compile app.py model_router.py semantic_embeddings.py snapdragon_backend.py setup_models.py preflight_check.py
if errorlevel 1 goto :code_failed
"%PYEXE%" preflight_check.py
if errorlevel 1 goto :code_failed
"%PYEXE%" -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='SamLowe/roberta-base-go_emotions-onnx')" >nul 2>nul

echo.
echo [4/4] Starting ArkaAI...
"%PYEXE%" app.py
if errorlevel 1 (
    echo.
    echo ArkaAI stopped. Keep this window open to inspect the error.
    pause
)
exit /b 0

:python_missing
 echo ERROR: Python was not found and winget is unavailable.
 pause
 exit /b 1
:python_failed
 echo ERROR: Python installation failed.
 pause
 exit /b 1
:venv_failed
 echo ERROR: Could not create the private Python environment.
 pause
 exit /b 1
:pip_failed
 echo ERROR: Python dependencies could not be installed.
 pause
 exit /b 1
:model_failed
 echo ERROR: Local AI runtime/model setup failed. Check internet, storage and Snapdragon hardware.
 pause
 exit /b 1
:code_failed
 echo ERROR: A Python module failed the syntax check.
 pause
 exit /b 1
