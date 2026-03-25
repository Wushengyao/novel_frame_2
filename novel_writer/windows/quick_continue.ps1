Param(
	[string]$ProjectPath = "",
	[int]$ChapterCount = 3,
	[string]$UserRequest = "",
	[string]$ProviderOverride = "",
	[bool]$AutoIllustrate = $true
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
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
		return @{
			GEMINI_API_KEY = ""
			GROK_API_KEY = ""
			DEEPSEEK_API_KEY = ""
			DOUBAO_API_KEY = ""
			OLLAMA_API_KEY = ""
		}
	}
	$content = Get-Content -Path $KeysFile -Raw -Encoding UTF8
	$keys = @{
		GEMINI_API_KEY = ""
		GROK_API_KEY = ""
		DEEPSEEK_API_KEY = ""
		DOUBAO_API_KEY = ""
		OLLAMA_API_KEY = ""
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
		"doubao" { return "doubao" }
		"ollama" { return "ollama" }
		default { throw "Unsupported provider: $Name (allowed: gemini / grok / deepseek / doubao / ollama)" }
	}
}

function Default-ModelForProvider {
	param([string]$Name)
	switch ($Name) {
		"gemini" { return "gemini-3.1-pro-preview" }
		"grok" { return "grok-4.20-beta-latest-reasoning" }
		"deepseek" { return "deepseek-reasoner" }
		"doubao" { return "doubao-seed-2-0-pro-260215" }
		"ollama" { return "llama3.2" }
	}
	return ""
}

function Default-ApiBaseForProvider {
	param([string]$Name)
	if ($Name -eq "doubao") { return "https://ark.cn-beijing.volces.com/api/v3" }
	if ($Name -eq "ollama") { return "http://127.0.0.1:11434/v1" }
	return ""
}

function Default-ThinkingLevel {
	param([string]$Name)
	if ($Name -eq "gemini") { return "medium" }
	return ""
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

function Invoke-NativeCommandCapture {
	param(
		[string]$Executable,
		[string[]]$Arguments
	)

	$previousErrorActionPreference = $ErrorActionPreference
	try {
		$ErrorActionPreference = "Continue"
		$output = & $Executable @Arguments 2>&1 | ForEach-Object {
			if ($_ -is [System.Management.Automation.ErrorRecord]) {
				$_.ToString()
			}
			else {
				"$_"
			}
		}
		$exitCode = $LASTEXITCODE
		return @{
			Output = @($output)
			ExitCode = $exitCode
		}
	}
	finally {
		$ErrorActionPreference = $previousErrorActionPreference
	}
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
	$ProviderOverride = Prompt-OptionalValue -PromptText "Provider override (optional: gemini/grok/deepseek/doubao/ollama)"
}

$pythonExe = Resolve-PythonExe
$apiKeys = Get-ApiKeys -KeysFile (Join-Path $ProjectRoot "api_keys.sh")
$savedProject = Get-Content -Path $projectFile -Raw -Encoding UTF8 | ConvertFrom-Json
$saved = $savedProject.llm_config
if (-not $saved) { $saved = @{} }
$savedProvider = if ($saved.model_provider) { ("$($saved.model_provider)").ToLowerInvariant() } else { "gemini" }

$ProviderOverride = Normalize-Provider $ProviderOverride
$resolvedProvider = if ($ProviderOverride) { $ProviderOverride } else { $savedProvider }

$apiKey = $env:NOVEL_API_KEY
if (-not $apiKey) {
	switch ($resolvedProvider) {
		"gemini" { $apiKey = $apiKeys["GEMINI_API_KEY"] }
		"grok" { $apiKey = $apiKeys["GROK_API_KEY"] }
		"deepseek" { $apiKey = $apiKeys["DEEPSEEK_API_KEY"] }
		"doubao" { $apiKey = $apiKeys["DOUBAO_API_KEY"] }
		"ollama" { $apiKey = $apiKeys["OLLAMA_API_KEY"] }
	}
}
if ($resolvedProvider -ne "ollama" -and -not $apiKey) {
	throw "provider=$resolvedProvider missing API key. Please fill $ProjectRoot\api_keys.sh"
}

$modelName = if ($env:NOVEL_MODEL_NAME_OVERRIDE) {
	$env:NOVEL_MODEL_NAME_OVERRIDE
} elseif ($resolvedProvider -ne $savedProvider) {
	Default-ModelForProvider $resolvedProvider
} elseif ($saved.model_name) {
	"$($saved.model_name)"
} elseif ($saved.model) {
	"$($saved.model)"
} else {
	Default-ModelForProvider $resolvedProvider
}
$apiBase = if ($env:NOVEL_API_BASE_OVERRIDE) {
	$env:NOVEL_API_BASE_OVERRIDE
} elseif ($resolvedProvider -ne $savedProvider) {
	Default-ApiBaseForProvider $resolvedProvider
} elseif ($saved.api_base) {
	"$($saved.api_base)"
} else {
	Default-ApiBaseForProvider $resolvedProvider
}
$temperature = if ($env:NOVEL_TEMPERATURE_OVERRIDE) { [double]$env:NOVEL_TEMPERATURE_OVERRIDE } elseif ($saved.temperature) { [double]$saved.temperature } else { 0.8 }
$maxTokens = if ($env:NOVEL_MAX_TOKENS_OVERRIDE) { [int]$env:NOVEL_MAX_TOKENS_OVERRIDE } elseif ($saved.max_tokens) { [int]$saved.max_tokens } else { 4000 }
$timeout = if ($env:NOVEL_TIMEOUT_OVERRIDE) { [int]$env:NOVEL_TIMEOUT_OVERRIDE } elseif ($saved.timeout) { [int]$saved.timeout } else { 120 }
$thinkingLevel = if ($env:NOVEL_THINKING_LEVEL_OVERRIDE) {
	$env:NOVEL_THINKING_LEVEL_OVERRIDE
} elseif ($resolvedProvider -eq $savedProvider -and $saved.thinking_level) {
	"$($saved.thinking_level)"
} else {
	Default-ThinkingLevel $resolvedProvider
}

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
		(Join-Path $ProjectRoot "app.py"),
		"next",
		"--project", $ProjectPath,
		"--config", $tempConfig,
		"--count", "$ChapterCount"
	)
	if ($UserRequest) {
		$nextArgs += @("--user-request", $UserRequest)
	}

	$nextResult = Invoke-NativeCommandCapture -Executable $pythonExe -Arguments $nextArgs
	$nextOutput = $nextResult.Output
	$nextExitCode = $nextResult.ExitCode
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
				(Join-Path $ProjectRoot "app.py"),
				"illustrate",
				"--project", $ProjectPath,
				"--chapter", $chapterPath,
				"--config", $tempConfig
			)

			$illustrateResult = Invoke-NativeCommandCapture -Executable $pythonExe -Arguments $illustrateArgs
			$illustrateOutput = $illustrateResult.Output
			$illustrateExitCode = $illustrateResult.ExitCode
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

	$statusResult = Invoke-NativeCommandCapture -Executable $pythonExe -Arguments @(
		(Join-Path $ProjectRoot "app.py"),
		"status",
		"--project", $ProjectPath
	)
	$statusOutput = $statusResult.Output
	$statusExitCode = $statusResult.ExitCode
	$statusOutput | ForEach-Object { Write-Output $_ }
	if ($statusExitCode -ne 0) {
		throw "Status command failed with exit code $statusExitCode."
	}
}
finally {
	Remove-Item -Path $tempConfig -ErrorAction SilentlyContinue
}
