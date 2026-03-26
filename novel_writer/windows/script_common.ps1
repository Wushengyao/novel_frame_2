Set-StrictMode -Version Latest

function Test-PythonExecutable {
	param([string]$Executable)

	try {
		$null = & $Executable "--version" 2>$null
		return ($null -eq $LASTEXITCODE -or $LASTEXITCODE -eq 0)
	}
	catch {
		return $false
	}
}

function Resolve-PythonCandidatePath {
	param([string]$PathValue)

	if ([string]::IsNullOrWhiteSpace($PathValue)) {
		return ""
	}

	if (-not (Test-Path -LiteralPath $PathValue)) {
		return ""
	}

	$item = Get-Item -LiteralPath $PathValue -ErrorAction SilentlyContinue
	if (-not $item) {
		return ""
	}

	if ($item.PSIsContainer) {
		$pythonExe = Join-Path $item.FullName "python.exe"
		if (Test-Path -LiteralPath $pythonExe) {
			return $pythonExe
		}
		return ""
	}

	return $item.FullName
}

function Prompt-OptionalValue {
	param(
		[string]$PromptText,
		[string]$DefaultValue = ""
	)

	try {
		if ([Console]::IsInputRedirected -or [Console]::IsOutputRedirected) {
			return $DefaultValue
		}
	}
	catch {
		return $DefaultValue
	}

	if ($DefaultValue) {
		$inputValue = Read-Host "$PromptText [$DefaultValue]"
		if ([string]::IsNullOrWhiteSpace($inputValue)) {
			return $DefaultValue
		}
		return $inputValue.Trim()
	}

	$inputValue = Read-Host $PromptText
	return $inputValue.Trim()
}

function Resolve-PythonExe {
	if ($env:NOVEL_PYTHON_EXE) {
		$resolvedPath = Resolve-PythonCandidatePath -PathValue $env:NOVEL_PYTHON_EXE
		if ($resolvedPath -and (Test-PythonExecutable -Executable $resolvedPath)) {
			return $resolvedPath
		}
	}

	foreach ($candidatePath in @(
		"D:\ProgramData\Anaconda3\python.exe",
		"D:\ProgramData\Anaconda3",
		"C:\ProgramData\Anaconda3\python.exe",
		"C:\ProgramData\Anaconda3"
	)) {
		$resolvedCandidatePath = Resolve-PythonCandidatePath -PathValue $candidatePath
		if ($resolvedCandidatePath -and (Test-PythonExecutable -Executable $resolvedCandidatePath)) {
			return $resolvedCandidatePath
		}
	}

	foreach ($candidate in @("python", "py", "python3")) {
		$command = Get-Command $candidate -ErrorAction SilentlyContinue
		if ($command) {
			$resolvedCommand = if ($command.Path) {
				$command.Path
			}
			elseif ($command.Source) {
				$command.Source
			}
			else {
				$candidate
			}

			if (Test-PythonExecutable -Executable $resolvedCommand) {
				return $resolvedCommand
			}
		}
	}

	throw "Python not found. Install Python or set NOVEL_PYTHON_EXE."
}

function Get-ApiKeys {
	param([string]$KeysFile)

	if (-not (Test-Path -LiteralPath $KeysFile)) {
		throw "Missing API key file: $KeysFile"
	}

	$content = Get-Content -LiteralPath $KeysFile -Raw -Encoding UTF8
	$keys = @{
		GEMINI_API_KEY = ""
		GROK_API_KEY = ""
		DEEPSEEK_API_KEY = ""
		DOUBAO_API_KEY = ""
	}

	foreach ($name in @($keys.Keys)) {
		$pattern = 'export\s+' + [regex]::Escape($name) + "=(?:""([^""]*)""|'([^']*)'|([^\r\n#]+))"
		$match = [regex]::Match($content, $pattern)
		if ($match.Success) {
			foreach ($groupIndex in 1..3) {
				$value = $match.Groups[$groupIndex].Value
				if ($value) {
					$keys[$name] = $value.Trim()
					break
				}
			}
		}
	}

	return $keys
}

function Normalize-Provider {
	param([string]$Name)

	$normalizedName = if ($null -eq $Name) { "" } else { $Name }
	switch ($normalizedName.Trim().ToLowerInvariant()) {
		"gemini" { return "gemini" }
		"grok" { return "grok" }
		"deepseek" { return "deepseek" }
		"doubao" { return "doubao" }
		default { throw "Unsupported provider: $Name (allowed: gemini / grok / deepseek / doubao)" }
	}
}

function Get-DefaultModelForProvider {
	param([string]$Provider)

	switch (Normalize-Provider $Provider) {
		"gemini" { return "gemini-3.1-pro-preview" }
		"grok" { return "grok-4.20-beta-latest-reasoning" }
		"deepseek" { return "deepseek-reasoner" }
		"doubao" { return "doubao-seed-2-0-pro-260215" }
	}
}

function Get-DefaultApiBaseForProvider {
	param([string]$Provider)

	switch (Normalize-Provider $Provider) {
		"doubao" { return "https://ark.cn-beijing.volces.com/api/v3" }
		default { return "" }
	}
}

function Get-DefaultThinkingLevelForProvider {
	param([string]$Provider)

	switch (Normalize-Provider $Provider) {
		"gemini" { return "medium" }
		default { return "" }
	}
}

function Get-ApiKeyForProvider {
	param(
		[string]$Provider,
		[hashtable]$ApiKeys
	)

	switch (Normalize-Provider $Provider) {
		"gemini" { return $ApiKeys["GEMINI_API_KEY"] }
		"grok" { return $ApiKeys["GROK_API_KEY"] }
		"deepseek" { return $ApiKeys["DEEPSEEK_API_KEY"] }
		"doubao" { return $ApiKeys["DOUBAO_API_KEY"] }
	}
}

function Ensure-ApiKeyPresent {
	param(
		[string]$Provider,
		[string]$ApiKey,
		[string]$ProjectRoot
	)

	if ([string]::IsNullOrWhiteSpace($ApiKey)) {
		throw "provider=$Provider missing API key. Please fill $ProjectRoot\api_keys.sh"
	}
}

function New-TempConfigPath {
	param([string]$Prefix = "novel_writer_config")

	return [System.IO.Path]::Combine(
		[System.IO.Path]::GetTempPath(),
		("{0}_{1}.json" -f $Prefix, [guid]::NewGuid().ToString("N"))
	)
}

function Get-LatestProjectPath {
	param([string]$OutputRoot)

	$latest = Get-ChildItem -LiteralPath $OutputRoot -Directory -ErrorAction SilentlyContinue |
		Where-Object { $_.Name -like "novel_project_*" } |
		Sort-Object LastWriteTime -Descending |
		Select-Object -First 1

	if ($latest) {
		return $latest.FullName
	}

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

		$exitCode = if ($null -eq $LASTEXITCODE) { 0 } else { $LASTEXITCODE }
		return @{
			Output = @($output)
			ExitCode = $exitCode
		}
	}
	finally {
		$ErrorActionPreference = $previousErrorActionPreference
	}
}

function Get-ProjectJson {
	param([string]$ProjectPath)

	$projectFile = Join-Path $ProjectPath "project.json"
	return Get-Content -LiteralPath $projectFile -Raw -Encoding UTF8 | ConvertFrom-Json
}

function Write-Utf8JsonFile {
	param(
		[string]$Path,
		[object]$Data
	)

	$json = $Data | ConvertTo-Json -Depth 20
	[System.IO.File]::WriteAllText($Path, $json, [System.Text.UTF8Encoding]::new($false))
}

function Write-InitConfig {
	param(
		[string]$OutputPath,
		[string]$ProjectRoot,
		[string]$ProjectName,
		[string]$ProjectDescription,
		[string]$StoryRequest,
		[string]$Provider,
		[string]$ModelName,
		[string]$ApiBase,
		[string]$ApiKey,
		[double]$Temperature,
		[int]$MaxTokens,
		[int]$Timeout,
		[string]$ThinkingLevel = ""
	)

	$outputRoot = Join-Path $ProjectRoot "output"
	if (-not (Test-Path -LiteralPath $outputRoot)) {
		New-Item -Path $outputRoot -ItemType Directory | Out-Null
	}

	$config = [ordered]@{
		project_name = $ProjectName
		project_description = $ProjectDescription
		project_path = (Join-Path $outputRoot "novel_project_{project_id}")
		init_with_llm = $true
		story_request = $StoryRequest
		model_provider = $Provider
		model_name = $ModelName
		api_base = $ApiBase
		api_key = $ApiKey
		temperature = $Temperature
		max_tokens = $MaxTokens
		timeout = $Timeout
	}

	if (-not [string]::IsNullOrWhiteSpace($ThinkingLevel)) {
		$config["thinking_level"] = $ThinkingLevel
	}

	Write-Utf8JsonFile -Path $OutputPath -Data $config
}

function Write-ContinueConfig {
	param(
		[string]$OutputPath,
		[string]$ProjectPath,
		[string]$ProviderOverride = "",
		[string]$ApiKey,
		[string]$ModelNameOverride = "",
		[string]$ApiBaseOverride = "",
		[string]$TemperatureOverride = "",
		[string]$MaxTokensOverride = "",
		[string]$TimeoutOverride = "",
		[string]$ThinkingLevelOverride = ""
	)

	$project = Get-ProjectJson -ProjectPath $ProjectPath
	$saved = $project.llm_config
	if (-not $saved) {
		$saved = [pscustomobject]@{}
	}

	$savedProvider = if ($saved.model_provider) { "$($saved.model_provider)".Trim().ToLowerInvariant() } else { "gemini" }
	$resolvedProvider = if ($ProviderOverride) { Normalize-Provider $ProviderOverride } else { $savedProvider }

	$modelName = if ($ModelNameOverride) {
		$ModelNameOverride
	}
	elseif ($resolvedProvider -ne $savedProvider) {
		Get-DefaultModelForProvider $resolvedProvider
	}
	elseif ($saved.model_name) {
		"$($saved.model_name)"
	}
	elseif ($saved.model) {
		"$($saved.model)"
	}
	else {
		Get-DefaultModelForProvider $resolvedProvider
	}

	$apiBase = if ($ApiBaseOverride) {
		$ApiBaseOverride
	}
	elseif ($resolvedProvider -ne $savedProvider) {
		Get-DefaultApiBaseForProvider $resolvedProvider
	}
	elseif ($saved.api_base) {
		"$($saved.api_base)"
	}
	else {
		Get-DefaultApiBaseForProvider $resolvedProvider
	}

	$temperature = if ($TemperatureOverride) { [double]$TemperatureOverride } elseif ($null -ne $saved.temperature) { [double]$saved.temperature } else { 0.8 }
	$maxTokens = if ($MaxTokensOverride) { [int]$MaxTokensOverride } elseif ($null -ne $saved.max_tokens) { [int]$saved.max_tokens } else { 4000 }
	$timeout = if ($TimeoutOverride) { [int]$TimeoutOverride } elseif ($null -ne $saved.timeout) { [int]$saved.timeout } else { 120 }

	$thinkingLevel = if ($ThinkingLevelOverride) {
		$ThinkingLevelOverride
	}
	elseif ($resolvedProvider -eq $savedProvider -and $saved.thinking_level) {
		"$($saved.thinking_level)"
	}
	else {
		Get-DefaultThinkingLevelForProvider $resolvedProvider
	}

	$config = [ordered]@{
		model_provider = $resolvedProvider
		model_name = $modelName
		api_base = $apiBase
		api_key = $ApiKey
		temperature = $temperature
		max_tokens = $maxTokens
		timeout = $timeout
	}

	if (-not [string]::IsNullOrWhiteSpace($thinkingLevel)) {
		$config["thinking_level"] = $thinkingLevel
	}

	if ($saved.thinking_budget) {
		$config["thinking_budget"] = "$($saved.thinking_budget)"
	}

	Write-Utf8JsonFile -Path $OutputPath -Data $config
}

function Write-IllustrateConfig {
	param(
		[string]$OutputPath,
		[string]$ProjectPath,
		[string]$ApiKey,
		[string]$ModelNameOverride = "",
		[string]$ApiBaseOverride = "",
		[string]$TemperatureOverride = "",
		[string]$MaxTokensOverride = "",
		[string]$TimeoutOverride = ""
	)

	$project = Get-ProjectJson -ProjectPath $ProjectPath
	$saved = $project.llm_config
	if (-not $saved) {
		$saved = [pscustomobject]@{}
	}

	$resolvedProvider = if ($saved.model_provider) { "$($saved.model_provider)".Trim().ToLowerInvariant() } else { "gemini" }
	$modelName = if ($ModelNameOverride) {
		$ModelNameOverride
	}
	elseif ($saved.model_name) {
		"$($saved.model_name)"
	}
	elseif ($saved.model) {
		"$($saved.model)"
	}
	else {
		""
	}

	$apiBase = if ($ApiBaseOverride) {
		$ApiBaseOverride
	}
	elseif ($saved.api_base) {
		"$($saved.api_base)"
	}
	else {
		""
	}

	$temperature = if ($TemperatureOverride) { [double]$TemperatureOverride } elseif ($null -ne $saved.temperature) { [double]$saved.temperature } else { 0.8 }
	$maxTokens = if ($MaxTokensOverride) { [int]$MaxTokensOverride } elseif ($null -ne $saved.max_tokens) { [int]$saved.max_tokens } else { 4000 }
	$timeout = if ($TimeoutOverride) { [int]$TimeoutOverride } elseif ($null -ne $saved.timeout) { [int]$saved.timeout } else { 120 }

	$config = [ordered]@{
		model_provider = $resolvedProvider
		model_name = $modelName
		api_base = $apiBase
		api_key = $ApiKey
		temperature = $temperature
		max_tokens = $maxTokens
		timeout = $timeout
	}

	if ($saved.thinking_level) {
		$config["thinking_level"] = "$($saved.thinking_level)"
	}

	if ($saved.thinking_budget) {
		$config["thinking_budget"] = "$($saved.thinking_budget)"
	}

	Write-Utf8JsonFile -Path $OutputPath -Data $config
}
