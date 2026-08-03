$ErrorActionPreference = "Stop"

$pythonExe = "C:\Users\sFerm\AppData\Local\Python\bin\python.exe"
Set-Location -LiteralPath $PSScriptRoot

$started = Get-Date -Format "yyyy-MM-dd HH:mm:ss zzz"
Write-Output "[$started] Starting pairwise minimum scans: CEP, CPGAMMA, CEGAMMA"
Write-Output "Species order within each objective: electron, muon"
Write-Output "Contours will start only if the complete minimum-scan stage succeeds."

& $pythonExe GradientPhaseSpaceScan.py
if ($LASTEXITCODE -ne 0) {
    throw "GradientPhaseSpaceScan.py failed with exit code $LASTEXITCODE"
}

$contourStarted = Get-Date -Format "yyyy-MM-dd HH:mm:ss zzz"
Write-Output "[$contourStarted] All minimum scans completed; starting contours."
& $pythonExe GradientPhaseSpaceContour.py
if ($LASTEXITCODE -ne 0) {
    throw "GradientPhaseSpaceContour.py failed with exit code $LASTEXITCODE"
}

$finished = Get-Date -Format "yyyy-MM-dd HH:mm:ss zzz"
Write-Output "[$finished] Pairwise minimum scans and contours completed successfully."
