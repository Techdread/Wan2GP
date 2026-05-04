$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$logPath = Join-Path $root "wan2gp-api-task.log"
$launcher = Join-Path $root "start-wan2gp-api.bat"

Set-Location -LiteralPath $root
Add-Content -LiteralPath $logPath -Value "[$(Get-Date -Format o)] Starting Wan2GP API task"

$existing = netstat -ano | Select-String -Pattern "0\.0\.0\.0:8100|127\.0\.0\.1:8100|\[::\]:8100"
if ($existing -and ($existing -match "LISTENING")) {
    Add-Content -LiteralPath $logPath -Value "[$(Get-Date -Format o)] Port 8100 is already listening; leaving existing server in place"
    exit 0
}

& $launcher *>> $logPath
