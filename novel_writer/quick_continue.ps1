Param(
	[string]$ProjectPath = "",
	[int]$ChapterCount = 3,
	[string]$UserRequest = "",
	[string]$ProviderOverride = "",
	[bool]$AutoIllustrate = $true
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
if ($PSVersionTable.PSVersion.Major -ge 7) {
	$PSNativeCommandUseErrorActionPreference = $false
}

function Prompt-OptionalValue {
	param(
		[string]$PromptText,
		[string]$DefaultValue = ""
	)
	if ($DefaultValue) {
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

function Normalize-Provider {
	param([string]$Name)
	if (-not $Name) { return "" }
	switch (($Name | ForEach-Object { $_.ToLowerInvariant() })) {
		"gemini" { return "gemini" }
		"grok" { return "grok" }
		"deepseek" { return "deepseek" }
		default { throw "Unsupported provider: $Name (allowed: gemini / grok / deepseek)" }
	}
}

function Test-IllustrationConnectionFailure {
	param([string]$Text)
	if (-not $Text) {
		return $false
	}
	return (
		$Text -match "Failed to connect to ComfyUI" -or
		$Text -match "actively refused" -or
		$Text -match "Connection refused" -or
		$Text -match "WinError 10061" -or
		$Text -match "No connection could be made" -or
		$Text -match "timed out"
	)
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

if (-not $PSBoundParameters.ContainsKey("ChapterCount")) {
	$chapterCountInput = Prompt-OptionalValue -PromptText "Chapter count" -DefaultValue "3"
	try {
		$ChapterCount = [int]$chapterCountInput
	}
	catch {
		throw "Chapter count must be an integer."
	}
}
if (-not $PSBoundParameters.ContainsKey("UserRequest")) {
	$UserRequest = Prompt-OptionalValue -PromptText "User request (optional)"
}
if (-not $PSBoundParameters.ContainsKey("ProviderOverride")) {
	$ProviderOverride = Prompt-OptionalValue -PromptText "Provider override (optional: gemini/grok/deepseek)"
}

$pythonExe = Resolve-PythonExe
$apiKeys = Get-ApiKeys -KeysFile (Join-Path $ScriptDir "api_keys.sh")
$savedProject = Get-Content -Path $projectFile -Raw -Encoding UTF8 | ConvertFrom-Json
$saved = $savedProject.llm_config
if (-not $saved) { $saved = @{} }

$ProviderOverride = Normalize-Provider $ProviderOverride
$resolvedProvider = if ($ProviderOverride) { $ProviderOverride } elseif ($saved.model_provider) { "$($saved.model_provider)" } else { "gemini" }

$apiKey = $env:NOVEL_API_KEY
if (-not $apiKey) {
	switch ($resolvedProvider) {
		"gemini" { $apiKey = $apiKeys["GEMINI_API_KEY"] }
		"grok" { $apiKey = $apiKeys["GROK_API_KEY"] }
		"deepseek" { $apiKey = $apiKeys["DEEPSEEK_API_KEY"] }
	}
}
if (-not $apiKey) {
	throw "provider=$resolvedProvider missing API key. Please fill $ScriptDir\api_keys.sh"
}

$modelName = if ($env:NOVEL_MODEL_NAME_OVERRIDE) { $env:NOVEL_MODEL_NAME_OVERRIDE } elseif ($saved.model_name) { "$($saved.model_name)" } elseif ($saved.model) { "$($saved.model)" } else { "" }
$apiBase = if ($env:NOVEL_API_BASE_OVERRIDE) { $env:NOVEL_API_BASE_OVERRIDE } elseif ($saved.api_base) { "$($saved.api_base)" } else { "" }
$temperature = if ($env:NOVEL_TEMPERATURE_OVERRIDE) { [double]$env:NOVEL_TEMPERATURE_OVERRIDE } elseif ($saved.temperature) { [double]$saved.temperature } else { 0.8 }
$maxTokens = if ($env:NOVEL_MAX_TOKENS_OVERRIDE) { [int]$env:NOVEL_MAX_TOKENS_OVERRIDE } elseif ($saved.max_tokens) { [int]$saved.max_tokens } else { 4000 }
$timeout = if ($env:NOVEL_TIMEOUT_OVERRIDE) { [int]$env:NOVEL_TIMEOUT_OVERRIDE } elseif ($saved.timeout) { [int]$saved.timeout } else { 120 }
$thinkingLevel = if ($env:NOVEL_THINKING_LEVEL_OVERRIDE) { $env:NOVEL_THINKING_LEVEL_OVERRIDE } elseif ($saved.thinking_level) { "$($saved.thinking_level)" } else { "" }

$config = [ordered]@{
	model_provider = $resolvedProvider
	model_name = $modelName
	api_base = $apiBase
	api_key = $apiKey
	temperature = $temperature
	max_tokens = $maxTokens
	timeout = $timeout
}
if ($thinkingLevel) {
	$config["thinking_level"] = $thinkingLevel
}
if ($saved.thinking_budget) {
	$config["thinking_budget"] = "$($saved.thinking_budget)"
}

$tempConfig = [System.IO.Path]::Combine([System.IO.Path]::GetTempPath(), ("novel_writer_config_{0}.json" -f ([guid]::NewGuid().ToString("N"))))
[System.IO.File]::WriteAllText($tempConfig, ($config | ConvertTo-Json -Depth 10), [System.Text.UTF8Encoding]::new($false))

try {
	$nextArgs = @(
		(Join-Path $ScriptDir "app.py"),
		"next",
		"--project", $ProjectPath,
		"--config", $tempConfig,
		"--count", "$ChapterCount"
	)
	if ($UserRequest) {
		$nextArgs += @("--user-request", $UserRequest)
	}

	$nextOutput = & $pythonExe @nextArgs 2>&1
	$nextExitCode = $LASTEXITCODE
	$nextOutput | ForEach-Object { Write-Output $_ }
	if ($nextExitCode -ne 0) {
		throw "Chapter generation failed with exit code $nextExitCode."
	}

	$generatedChapterPaths = @()
	foreach ($line in $nextOutput) {
		$lineText = "$line"
		if ($lineText -match '^新章节已保存:\s+(.+)$') {
			$generatedChapterPaths += $matches[1].Trim()
		}
	}

	if ($AutoIllustrate -and $generatedChapterPaths.Count -gt 0) {
		Write-Output "正在尝试自动创建插图..."
		foreach ($chapterPath in $generatedChapterPaths) {
			$illustrateArgs = @(
				(Join-Path $ScriptDir "app.py"),
				"illustrate",
				"--project", $ProjectPath,
				"--chapter", $chapterPath,
				"--config", $tempConfig
			)

			$illustrateOutput = & $pythonExe @illustrateArgs 2>&1
			$illustrateExitCode = $LASTEXITCODE
			$illustrateOutput | ForEach-Object { Write-Output $_ }

			if ($illustrateExitCode -ne 0) {
				$illustrateText = ($illustrateOutput | ForEach-Object { "$_" }) -join "`n"
				if (Test-IllustrationConnectionFailure -Text $illustrateText) {
					Write-Warning "ComfyUI 不可连接，已跳过自动插图创建。"
					break
				}
				throw "Illustration generation failed with exit code $illustrateExitCode."
			}
		}
	}
	elseif ($AutoIllustrate) {
		Write-Warning "未检测到新章节路径，已跳过自动插图创建。"
	}

	$statusOutput = & $pythonExe (Join-Path $ScriptDir "app.py") status --project $ProjectPath 2>&1
	$statusExitCode = $LASTEXITCODE
	$statusOutput | ForEach-Object { Write-Output $_ }
	if ($statusExitCode -ne 0) {
		throw "Status command failed with exit code $statusExitCode."
	}
}
finally {
	Remove-Item -Path $tempConfig -ErrorAction SilentlyContinue
}

