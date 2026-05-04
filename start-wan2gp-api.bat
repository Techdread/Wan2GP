@echo off
setlocal

REM --- Wan2GP API server startup for Windows ---
REM Uses the local venv and WAN2GP_TOKEN from the process, user, or machine environment.

for %%I in ("%~dp0.") do set "WANGP_DIR=%%~fI"
set "PYTHON_EXE=%WANGP_DIR%\venv\Scripts\python.exe"
set "WAN2GP_OUTPUTS_ROOT=%WANGP_DIR%\outputs"
set "WAN2GP_JOB_DB=%USERPROFILE%\.wan2gp\jobs.sqlite"
set "WAN2GP_LOG_PROMPTS=0"
set "WAN2GP_CORS_ORIGINS="
set "HOST=0.0.0.0"
set "PORT=8100"

if not defined WAN2GP_TOKEN (
  for /f "usebackq delims=" %%T in (`powershell -NoProfile -Command "$t=[Environment]::GetEnvironmentVariable('WAN2GP_TOKEN','User'); if (-not $t) { $t=[Environment]::GetEnvironmentVariable('WAN2GP_TOKEN','Machine') }; if ($t) { [Console]::Out.Write($t) }"`) do set "WAN2GP_TOKEN=%%T"
)

if not defined WAN2GP_TOKEN (
  echo [ERROR] WAN2GP_TOKEN is not set in the process, user, or machine environment.
  exit /b 1
)

if not exist "%WANGP_DIR%" (
  echo [ERROR] Wan2GP folder not found: %WANGP_DIR%
  exit /b 1
)

if not exist "%PYTHON_EXE%" (
  echo [ERROR] Python executable not found: %PYTHON_EXE%
  exit /b 1
)

if not exist "%USERPROFILE%\.wan2gp" mkdir "%USERPROFILE%\.wan2gp"
if not exist "%WAN2GP_OUTPUTS_ROOT%" mkdir "%WAN2GP_OUTPUTS_ROOT%"

cd /d "%WANGP_DIR%"

echo Starting Wan2GP API on %HOST%:%PORT%
echo Output root: %WAN2GP_OUTPUTS_ROOT%

"%PYTHON_EXE%" agent_api.py serve --host %HOST% --port %PORT% --profile 4 --attention sage2

endlocal
