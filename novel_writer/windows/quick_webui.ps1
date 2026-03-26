Param(
	[string]$BindHost = "0.0.0.0",
	[int]$Port = 8008
)

$ErrorActionPreference = "Stop"
if ($PSVersionTable.PSVersion.Major -ge 7) {
	$PSNativeCommandUseErrorActionPreference = $false
}

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
. (Join-Path $ScriptDir "script_common.ps1")

$pythonExe = Resolve-PythonExe
Set-Location $ProjectRoot
& $pythonExe (Join-Path $ProjectRoot "webui.py") --host $BindHost --port "$Port"
