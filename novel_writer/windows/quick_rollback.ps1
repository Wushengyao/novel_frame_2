Param(
	[string]$ProjectPath = "",
	[int]$ToChapter = -1
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir

function Prompt-OptionalValue {
	param(
		[string]$PromptText,
		[string]$DefaultValue = ""
	)
	if ($DefaultValue -ne "") {
		$inputValue = Read-Host "$PromptText [$DefaultValue]"
		if ([string]::IsNullOrWhiteSpace($inputValue)) {
			return $DefaultValue
		}
		return $inputValue.Trim()
	}
	return (Read-Host $PromptText).Trim()
}

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

if (-not $PSBoundParameters.ContainsKey("ProjectPath")) {
	$ProjectPath = Prompt-OptionalValue -PromptText "Project directory"
}
if (-not (Test-Path $ProjectPath)) {
	throw "Project directory does not exist: $ProjectPath"
}

$projectFile = Join-Path $ProjectPath "project.json"
if (-not (Test-Path $projectFile)) {
	throw "Missing project.json in directory: $ProjectPath"
}

if (-not $PSBoundParameters.ContainsKey("ToChapter")) {
	$toChapterInput = Prompt-OptionalValue -PromptText "Keep chapters up to" -DefaultValue "0"
	try {
		$ToChapter = [int]$toChapterInput
	}
	catch {
		throw "ToChapter must be an integer."
	}
}

if ($ToChapter -lt 0) {
	throw "ToChapter must be at least 0."
}

$pythonExe = Resolve-PythonExe
& $pythonExe (Join-Path $ProjectRoot "app.py") rollback --project $ProjectPath --to-chapter "$ToChapter"
if ($LASTEXITCODE -ne 0) {
	throw "Rollback failed with exit code $LASTEXITCODE."
}
