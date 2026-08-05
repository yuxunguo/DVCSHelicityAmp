$ErrorActionPreference = "Stop"

$pythonExe = "C:\Users\sFerm\AppData\Local\Python\bin\python.exe"
Set-Location -LiteralPath $PSScriptRoot

$started = Get-Date -Format "yyyy-MM-dd HH:mm:ss zzz"
Write-Output "[$started] Starting electron-first pairwise production."

& $pythonExe RunPairwiseElectronThenMuon.py
if ($LASTEXITCODE -ne 0) {
    throw "RunPairwiseElectronThenMuon.py failed with exit code $LASTEXITCODE"
}

$finished = Get-Date -Format "yyyy-MM-dd HH:mm:ss zzz"
Write-Output "[$finished] Electron-first pairwise production completed successfully."
