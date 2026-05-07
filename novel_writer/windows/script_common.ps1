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
		return @{
			GEMINI_API_KEY = ""
			GROK_API_KEY = ""
			DEEPSEEK_API_KEY = ""
			DOUBAO_API_KEY = ""
			OLLAMA_API_KEY = ""
			LLAMA_CPP_API_KEY = ""
		}
	}

	$content = Get-Content -LiteralPath $KeysFile -Raw -Encoding UTF8
	$keys = @{
		GEMINI_API_KEY = ""
		GROK_API_KEY = ""
		DEEPSEEK_API_KEY = ""
		DOUBAO_API_KEY = ""
		OLLAMA_API_KEY = ""
		LLAMA_CPP_API_KEY = ""
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
		"ollama" { return "ollama" }
		"llama_cpp" { return "llama_cpp" }
		"llama.cpp" { return "llama_cpp" }
		"llama-cpp" { return "llama_cpp" }
		"llamacpp" { return "llama_cpp" }
		default { throw "Unsupported provider: $Name (allowed: gemini / grok / deepseek / doubao / ollama / llama_cpp)" }
	}
}

function Get-DefaultModelForProvider {
	param([string]$Provider)

	switch (Normalize-Provider $Provider) {
		"gemini" { return "gemini-3.1-flash-lite-preview" }
		"grok" { return "grok-4.20-beta-latest-non-reasoning" }
		"deepseek" { return "deepseek-v4-pro" }
		"doubao" { return "doubao-seed-2-0-pro-260215" }
		"ollama" { return "llama3.2" }
		"llama_cpp" { return "local-model" }
	}
}

function Get-DefaultApiBaseForProvider {
	param([string]$Provider)

	switch (Normalize-Provider $Provider) {
		"doubao" { return "https://ark.cn-beijing.volces.com/api/v3" }
		"ollama" { return "http://127.0.0.1:11434/v1" }
		"llama_cpp" { return "http://127.0.0.1:8080/v1" }
		default { return "" }
	}
}

function Get-DefaultTimeoutForProvider {
	param([string]$Provider)

	switch (Normalize-Provider $Provider) {
		"ollama" { return 900 }
		"llama_cpp" { return 900 }
		default { return 120 }
	}
}

function Normalize-PlanningMode {
	param([string]$Mode)

	$normalizedMode = if ($null -eq $Mode) { "" } else { $Mode }
	switch ($normalizedMode.Trim().ToLowerInvariant()) {
		"none" { return "none" }
		"volume" { return "volume" }
		"chapter" { return "chapter" }
		"" { return "chapter" }
		default { throw "Unsupported planning mode: $Mode (allowed: none / volume / chapter)" }
	}
}

function Normalize-WorkflowMode {
	param([string]$Mode)

	$normalizedMode = if ($null -eq $Mode) { "" } else { $Mode }
	switch ($normalizedMode.Trim().ToLowerInvariant()) {
		"classic" { return "classic" }
		"agentic" { return "agentic" }
		"" { return "classic" }
		default { throw "Unsupported workflow mode: $Mode (allowed: classic / agentic)" }
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
		"ollama" { return $ApiKeys["OLLAMA_API_KEY"] }
		"llama_cpp" { return $ApiKeys["LLAMA_CPP_API_KEY"] }
	}
}

function Ensure-ApiKeyPresent {
	param(
		[string]$Provider,
		[string]$ApiKey,
		[string]$ProjectRoot
	)

	if ((Normalize-Provider $Provider) -in @("ollama", "llama_cpp")) {
		return
	}

	if ([string]::IsNullOrWhiteSpace($ApiKey)) {
		throw "provider=$Provider missing API key. Please fill $ProjectRoot\api_keys.sh"
	}
}

function Resolve-ProviderTimeout {
	param(
		[string]$Provider,
		[Nullable[int]]$Timeout
	)

	$defaultTimeout = Get-DefaultTimeoutForProvider $Provider
	if ($null -eq $Timeout) {
		return $defaultTimeout
	}

	$resolvedTimeout = [int]$Timeout
	if ((Normalize-Provider $Provider) -in @("ollama", "llama_cpp") -and $resolvedTimeout -lt $defaultTimeout) {
		return $defaultTimeout
	}

	return $resolvedTimeout
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
		[string[]]$Arguments,
		[switch]$StreamOutput
	)

	$previousErrorActionPreference = $ErrorActionPreference
	try {
		$ErrorActionPreference = "Continue"
		$output = & $Executable @Arguments 2>&1 | ForEach-Object {
			$line = if ($_ -is [System.Management.Automation.ErrorRecord]) {
				$_.ToString()
			}
			else {
				"$_"
			}
			if ($StreamOutput) {
				Write-Host $line
			}
			$line
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
		[string]$PlanningMode = "chapter",
		[string]$WorkflowMode = "classic",
		[string]$QualityProvider = "",
		[string]$QualityModelName = "",
		[string]$QualityApiBase = "",
		[string]$QualityApiKey = "",
		[string]$QualityTemperature = "",
		[string]$QualityMaxTokens = "",
		[string]$QualityTimeout = ""
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
		planning_mode = (Normalize-PlanningMode $PlanningMode)
		workflow_mode = (Normalize-WorkflowMode $WorkflowMode)
		model_provider = $Provider
		model_name = $ModelName
		api_base = $ApiBase
		api_key = $ApiKey
		temperature = $Temperature
		max_tokens = $MaxTokens
		timeout = $Timeout
	}
	$qualityModel = [ordered]@{}
	if ($QualityProvider) { $qualityModel.model_provider = Normalize-Provider $QualityProvider }
	if ($QualityModelName) {
		$qualityModel.model_name = $QualityModelName
		$qualityModel.model = $QualityModelName
	}
	if ($QualityApiBase) { $qualityModel.api_base = $QualityApiBase }
	if ($QualityApiKey) { $qualityModel.api_key = $QualityApiKey }
	if ($QualityTemperature) { $qualityModel.temperature = [double]$QualityTemperature }
	if ($QualityMaxTokens) { $qualityModel.max_tokens = [int]$QualityMaxTokens }
	if ($QualityTimeout) { $qualityModel.timeout = [int]$QualityTimeout }
	if ($qualityModel.Count -gt 0) { $config.quality_model = $qualityModel }

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
		[string]$PlanningModeOverride = "",
		[string]$WorkflowModeOverride = "",
		[string]$QualityProviderOverride = "",
		[string]$QualityModelNameOverride = "",
		[string]$QualityApiBaseOverride = "",
		[string]$QualityApiKey = "",
		[string]$QualityTemperatureOverride = "",
		[string]$QualityMaxTokensOverride = "",
		[string]$QualityTimeoutOverride = ""
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
	$timeout = if ($TimeoutOverride) {
		Resolve-ProviderTimeout -Provider $resolvedProvider -Timeout ([int]$TimeoutOverride)
	}
	elseif ($null -ne $saved.timeout) {
		Resolve-ProviderTimeout -Provider $resolvedProvider -Timeout ([int]$saved.timeout)
	}
	else {
		Resolve-ProviderTimeout -Provider $resolvedProvider -Timeout $null
	}

	$planningMode = if ($PlanningModeOverride) {
		Normalize-PlanningMode $PlanningModeOverride
	}
	elseif ($project.planning_mode) {
		Normalize-PlanningMode "$($project.planning_mode)"
	}
	else {
		"chapter"
	}
	$workflowMode = if ($WorkflowModeOverride) {
		Normalize-WorkflowMode $WorkflowModeOverride
	}
	elseif ($project.workflow_mode) {
		Normalize-WorkflowMode "$($project.workflow_mode)"
	}
	elseif ($saved.workflow_mode) {
		Normalize-WorkflowMode "$($saved.workflow_mode)"
	}
	else {
		"classic"
	}

	$config = [ordered]@{
		model_provider = $resolvedProvider
		model_name = $modelName
		api_base = $apiBase
		api_key = $ApiKey
		temperature = $temperature
		max_tokens = $maxTokens
		timeout = $timeout
		planning_mode = $planningMode
		workflow_mode = $workflowMode
	}
	$qualityModel = [ordered]@{}
	if ($saved.quality_model) {
		$savedQuality = $saved.quality_model
		if ($savedQuality.model_provider) { $qualityModel.model_provider = "$($savedQuality.model_provider)" }
		if ($savedQuality.model_name) {
			$qualityModel.model_name = "$($savedQuality.model_name)"
			$qualityModel.model = "$($savedQuality.model_name)"
		}
		elseif ($savedQuality.model) {
			$qualityModel.model_name = "$($savedQuality.model)"
			$qualityModel.model = "$($savedQuality.model)"
		}
		if ($savedQuality.api_base) { $qualityModel.api_base = "$($savedQuality.api_base)" }
		if ($null -ne $savedQuality.temperature) { $qualityModel.temperature = [double]$savedQuality.temperature }
		if ($null -ne $savedQuality.max_tokens) { $qualityModel.max_tokens = [int]$savedQuality.max_tokens }
		if ($null -ne $savedQuality.timeout) { $qualityModel.timeout = [int]$savedQuality.timeout }
	}
	if ($QualityProviderOverride) {
		$qualityModel.model_provider = Normalize-Provider $QualityProviderOverride
		if (-not $QualityModelNameOverride) {
			$qualityModel.Remove("model_name")
			$qualityModel.Remove("model")
		}
	}
	if ($QualityModelNameOverride) {
		$qualityModel.model_name = $QualityModelNameOverride
		$qualityModel.model = $QualityModelNameOverride
	}
	if ($QualityApiBaseOverride) { $qualityModel.api_base = $QualityApiBaseOverride }
	if ($QualityApiKey) { $qualityModel.api_key = $QualityApiKey }
	if ($QualityTemperatureOverride) { $qualityModel.temperature = [double]$QualityTemperatureOverride }
	if ($QualityMaxTokensOverride) { $qualityModel.max_tokens = [int]$QualityMaxTokensOverride }
	if ($QualityTimeoutOverride) { $qualityModel.timeout = [int]$QualityTimeoutOverride }
	if ($qualityModel.Count -gt 0) { $config.quality_model = $qualityModel }

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
	$timeout = if ($TimeoutOverride) {
		Resolve-ProviderTimeout -Provider $resolvedProvider -Timeout ([int]$TimeoutOverride)
	}
	elseif ($null -ne $saved.timeout) {
		Resolve-ProviderTimeout -Provider $resolvedProvider -Timeout ([int]$saved.timeout)
	}
	else {
		Resolve-ProviderTimeout -Provider $resolvedProvider -Timeout $null
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

	Write-Utf8JsonFile -Path $OutputPath -Data $config
}
