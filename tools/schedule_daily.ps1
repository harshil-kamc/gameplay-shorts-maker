# Registers a Windows Scheduled Task that runs the pipeline once a day.
# Usage:   .\tools\schedule_daily.ps1            (defaults to 9:00 AM)
#          .\tools\schedule_daily.ps1 -At 7am
# Remove:  Unregister-ScheduledTask -TaskName "GameplayShortsMaker" -Confirm:$false
param([string]$At = "9:00AM")

$runner = Join-Path $PSScriptRoot "run_daily.ps1"
$action  = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$runner`""
$trigger = New-ScheduledTaskTrigger -Daily -At $At
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable

Register-ScheduledTask -TaskName "GameplayShortsMaker" -Action $action `
    -Trigger $trigger -Settings $settings -Force `
    -Description "Daily GTA 6 gameplay shorts pipeline (Gameplay Shorts Maker)" | Out-Null

Write-Host "Registered 'GameplayShortsMaker' to run daily at $At."
Write-Host "It makes 5 gameplay Shorts/day from config.yaml -> scheduler.sources."
Write-Host "Test it now without waiting:  .\tools\run_daily.ps1"
