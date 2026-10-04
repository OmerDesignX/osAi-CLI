[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ([Environment]::OSVersion.Platform -ne [PlatformID]::Win32NT) {
  throw "Run this script on Windows 10 or Windows 11."
}

$projectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$buildScript = Join-Path $projectRoot "releaseScripts\common\build_release.py"

$releasePython = $env:OSAI_RELEASE_PYTHON
$releasePythonArgs = @()
if (-not $releasePython) {
  $venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
  if (Test-Path -LiteralPath $venvPython -PathType Leaf) {
    $releasePython = $venvPython
  }
}
if (-not $releasePython -and (Get-Command py.exe -ErrorAction SilentlyContinue)) {
  try {
    $installed = (& py.exe -0p 2>$null) -join "`n"
  }
  catch {
    $installed = ""
  }
  foreach ($version in @("3.13", "3.12", "3.11", "3.10")) {
    if ($installed -match "(?m)^\s*-(?:V:)?$([regex]::Escape($version))\b") {
      $releasePython = "py.exe"
      $releasePythonArgs = @("-$version")
      break
    }
  }
}
if (-not $releasePython) {
  $candidate = Get-Command python.exe -ErrorAction SilentlyContinue
  if ($candidate -and $candidate.Source -notlike '*\WindowsApps\*') {
    $releasePython = $candidate.Source
  }
}
if (-not $releasePython) {
  throw "Python 3.10-3.13 was not found."
}

& $releasePython @releasePythonArgs $buildScript
if ($LASTEXITCODE -ne 0) {
  throw "The osAi release build failed with exit code $LASTEXITCODE."
}
