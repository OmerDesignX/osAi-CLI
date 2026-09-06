[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ([Environment]::OSVersion.Platform -ne [PlatformID]::Win32NT) {
  throw "Run this script on Windows 10 or Windows 11."
}

$projectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$buildScript = Join-Path $projectRoot "releaseScripts\common\build_release.py"

if ($env:OSAI_RELEASE_PYTHON) {
  & $env:OSAI_RELEASE_PYTHON $buildScript
}
elseif (Get-Command py.exe -ErrorAction SilentlyContinue) {
  & py.exe -3.13 $buildScript
}
elseif (Get-Command python.exe -ErrorAction SilentlyContinue) {
  & python.exe $buildScript
}
else {
  throw "Python 3.10-3.13 was not found."
}

if ($LASTEXITCODE -ne 0) {
  throw "The osAi release build failed with exit code $LASTEXITCODE."
}
