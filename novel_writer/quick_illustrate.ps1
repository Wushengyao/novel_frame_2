Param(
	[string]$ProjectPath = "F:\novel_frame_2\novel_writer\output\novel_project_20260321T094557Z_eb96bba9",
	[string]$Chapter = "latest",
	[string]$UserRequest = "",
	[switch]$Force,
	[string]$Checkpoint = ""
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

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

function Get-ApiKeys {
	param([string]$KeysFile)
	if (-not (Test-Path $KeysFile)) {
		throw "Missing API key file: $KeysFile"
	}
	$content = Get-Content -Path $KeysFile -Raw -Encoding UTF8
	$keys = @{
		GEMINI_API_KEY = ""
		GROK_API_KEY = ""
		DEEPSEEK_API_KEY = ""
		DOUBAO_API_KEY = ""
	}
	foreach ($name in @($keys.Keys)) {
		$pattern = 'export\s+' + [regex]::Escape($name) + '="([^"]*)"'
		$match = [regex]::Match($content, $pattern)
		if ($match.Success) {
			$keys[$name] = $match.Groups[1].Value
		}
	}
	return $keys
}

if (-not (Test-Path $ProjectPath)) {
	throw "Project directory does not exist: $ProjectPath"
}

$projectFile = Join-Path $ProjectPath "project.json"
if (-not (Test-Path $projectFile)) {
	throw "Missing project.json in directory: $ProjectPath"
}

$pythonExe = Resolve-PythonExe
$apiKeys = Get-ApiKeys -KeysFile (Join-Path $ScriptDir "api_keys.sh")
$savedProject = Get-Content -Path $projectFile -Raw -Encoding UTF8 | ConvertFrom-Json
$saved = $savedProject.llm_config
if (-not $saved) { $saved = @{} }
$resolvedProvider = if ($saved.model_provider) { "$($saved.model_provider)" } else { "" }

$apiKey = $env:NOVEL_API_KEY
if (-not $apiKey) {
	switch ($resolvedProvider) {
		"gemini" { $apiKey = $apiKeys["GEMINI_API_KEY"] }
		"grok" { $apiKey = $apiKeys["GROK_API_KEY"] }
		"deepseek" { $apiKey = $apiKeys["DEEPSEEK_API_KEY"] }
		"doubao" { $apiKey = $apiKeys["DOUBAO_API_KEY"] }
	}
}

$config = [ordered]@{
	model_provider = $resolvedProvider
	model_name = if ($env:NOVEL_MODEL_NAME_OVERRIDE) { $env:NOVEL_MODEL_NAME_OVERRIDE } elseif ($saved.model_name) { "$($saved.model_name)" } elseif ($saved.model) { "$($saved.model)" } else { "" }
	api_base = if ($env:NOVEL_API_BASE_OVERRIDE) { $env:NOVEL_API_BASE_OVERRIDE } elseif ($saved.api_base) { "$($saved.api_base)" } else { "" }
	api_key = $apiKey
	temperature = if ($env:NOVEL_TEMPERATURE_OVERRIDE) { [double]$env:NOVEL_TEMPERATURE_OVERRIDE } elseif ($saved.temperature) { [double]$saved.temperature } else { 0.8 }
	max_tokens = if ($env:NOVEL_MAX_TOKENS_OVERRIDE) { [int]$env:NOVEL_MAX_TOKENS_OVERRIDE } elseif ($saved.max_tokens) { [int]$saved.max_tokens } else { 4000 }
	timeout = if ($env:NOVEL_TIMEOUT_OVERRIDE) { [int]$env:NOVEL_TIMEOUT_OVERRIDE } elseif ($saved.timeout) { [int]$saved.timeout } else { 120 }
}
if ($saved.thinking_level) {
	$config["thinking_level"] = "$($saved.thinking_level)"
}
if ($saved.thinking_budget) {
	$config["thinking_budget"] = "$($saved.thinking_budget)"
}

$tempConfig = [System.IO.Path]::Combine([System.IO.Path]::GetTempPath(), ("novel_writer_illustrate_{0}.json" -f ([guid]::NewGuid().ToString("N"))))
[System.IO.File]::WriteAllText($tempConfig, ($config | ConvertTo-Json -Depth 10), [System.Text.UTF8Encoding]::new($false))

try {
	$argsList = @(
		(Join-Path $ScriptDir "app.py"),
		"illustrate",
		"--project", $ProjectPath,
		"--chapter", $Chapter,
		"--config", $tempConfig
	)
	if ($UserRequest) {
		$argsList += @("--user-request", $UserRequest)
	}
	if ($Force) {
		$argsList += "--force"
	}
	if ($Checkpoint) {
		$argsList += @("--checkpoint", $Checkpoint)
	}

	& $pythonExe @argsList
}
finally {
	Remove-Item -Path $tempConfig -ErrorAction SilentlyContinue
}
