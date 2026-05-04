param(
    [string]$TaskName = "Wan2GP API",
    [string]$User = "$env:USERDOMAIN\$env:USERNAME"
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$script = Join-Path $root "start-wan2gp-api-task.ps1"

if (-not (Test-Path -LiteralPath $script)) {
    throw "Missing task wrapper: $script"
}

$action = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$script`"" `
    -WorkingDirectory $root

$trigger = New-ScheduledTaskTrigger -AtLogOn -User $User

$principal = New-ScheduledTaskPrincipal `
    -UserId $User `
    -LogonType Interactive `
    -RunLevel Highest

$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -MultipleInstances IgnoreNew `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 1)

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $action `
    -Trigger $trigger `
    -Principal $principal `
    -Settings $settings `
    -Description "Starts the Wan2GP LAN API node from $root at user logon." `
    -Force | Out-Null

Get-ScheduledTask -TaskName $TaskName |
    Select-Object TaskName, State,
        @{Name = "UserId"; Expression = { $_.Principal.UserId }},
        @{Name = "RunLevel"; Expression = { $_.Principal.RunLevel }},
        @{Name = "LogonType"; Expression = { $_.Principal.LogonType }}
