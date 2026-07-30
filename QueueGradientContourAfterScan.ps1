param(
    [Parameter(Mandatory = $true)]
    [int]$ScanPid
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$OutputRoot = Join-Path $RepoRoot "Output\GradientPhaseSpaceScan"
$QueueLog = Join-Path $OutputRoot "Logs\GHZ_muon_contour_queue.log"
$ScanLog = Join-Path $OutputRoot "Logs\GHZ_gradient_phase_space_scan.log"
$MinimaCsv = Join-Path $OutputRoot "muon\Data\GHZ\scan\local_minima.csv"
$ContourStdout = Join-Path $OutputRoot "Logs\GHZ_gradient_phase_space_contour.stdout.log"
$ContourStderr = Join-Path $OutputRoot "Logs\GHZ_gradient_phase_space_contour.stderr.log"
$PythonExe = "C:\Users\sFerm\AppData\Local\Python\bin\python.exe"

New-Item -ItemType Directory -Force -Path (Split-Path $QueueLog) | Out-Null

function Write-QueueLog {
    param([string]$Message)
    $Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -LiteralPath $QueueLog -Value "[$Timestamp] $Message"
}

Write-QueueLog "Waiting for muon GHZ scan PID $ScanPid."
try {
    Wait-Process -Id $ScanPid
}
catch {
    Write-QueueLog "Could not wait for scan PID $ScanPid`: $($_.Exception.Message)"
    exit 1
}

Write-QueueLog "Scan PID $ScanPid exited; validating completed scan artifacts."
if (-not (Test-Path -LiteralPath $MinimaCsv)) {
    Write-QueueLog "Contour stage not started: missing $MinimaCsv"
    exit 2
}
if (-not (Test-Path -LiteralPath $ScanLog)) {
    Write-QueueLog "Contour stage not started: missing completion log $ScanLog"
    exit 3
}

$MinimumCount = (Import-Csv -LiteralPath $MinimaCsv).Count
if ($MinimumCount -lt 1) {
    Write-QueueLog "Contour stage not started: local_minima.csv has no rows."
    exit 4
}

Write-QueueLog (
    "Starting the selected pre-cluster muon contour stages after validating " +
    "$MinimumCount GHZ raw minima; GHZ runs before W and clustering will not " +
    "be started."
)
$ContourProcess = Start-Process `
    -FilePath $PythonExe `
    -ArgumentList "GradientPhaseSpaceContour.py" `
    -WorkingDirectory $RepoRoot `
    -RedirectStandardOutput $ContourStdout `
    -RedirectStandardError $ContourStderr `
    -WindowStyle Hidden `
    -PassThru
Write-QueueLog "Contour parent PID $($ContourProcess.Id) started."
$ContourProcess.WaitForExit()
Write-QueueLog "Contour parent PID $($ContourProcess.Id) exited with code $($ContourProcess.ExitCode)."
exit $ContourProcess.ExitCode
