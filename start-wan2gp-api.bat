@echo off
setlocal

REM --- Wan2GP API server startup for Windows ---
REM Edit these paths and token before use.

set "WANGP_DIR=C:\Wan2GP"
set "PYTHON_EXE=%WANGP_DIR%\venv\Scripts\python.exe"
set "WAN2GP_TOKEN=REPLACE_WITH_YOUR_TOKEN"
set "WAN2GP_OUTPUTS_ROOT=%WANGP_DIR%\outputs"
set "WAN2GP_JOB_DB=%USERPROFILE%\.wan2gp\jobs.sqlite"
set "WAN2GP_LOG_PROMPTS=0"
set "WAN2GP_CORS_ORIGINS="
set "HOST=0.0.0.0"
set "PORT=8100"

if not exist "%WANGP_DIR%" (
  echo [ERROR] Wan2GP folder not found: %WANGP_DIR%
  exit /b 1
)

if not exist "%PYTHON_EXE%" (
  echo [ERROR] Python executable not found: %PYTHON_EXE%
  exit /b 1
)

if not exist "%USERPROFILE%\.wan2gp" mkdir "%USERPROFILE%\.wan2gp"

cd /d "%WANGP_DIR%"

echo Starting Wan2GP API on %HOST%:%PORT%
echo Output root: %WAN2GP_OUTPUTS_ROOT%

"%PYTHON_EXE%" agent_api.py serve --host %HOST% --port %PORT%

endlocal
