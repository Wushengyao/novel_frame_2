Param(
	[string]$BindHost = "0.0.0.0",
	[int]$Port = 8008
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir

function Resolve-PythonExe {
	if ($env:NOVEL_PYTHON_EXE -and (Test-Path $env:NOVEL_PYTHON_EXE)) {
		return $env:NOVEL_PYTHON_EXE
	}

	if (Test-Path "D:\ProgramData\Anaconda3\python.exe") {
		return "D:\ProgramData\Anaconda3\python.exe"
	}

	$cmd = Get-Command python -ErrorAction SilentlyContinue
	if ($cmd) {
		return $cmd.Path
	}

	throw "Python not found. Install Python or set NOVEL_PYTHON_EXE."
}

$pythonExe = Resolve-PythonExe
Set-Location $ProjectRoot
& $pythonExe (Join-Path $ProjectRoot "webui.py") --host $BindHost --port "$Port"
