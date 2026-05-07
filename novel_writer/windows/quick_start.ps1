Param(
	[string]$Provider = "gemini",
	[string]$StoryRequest = "",
	[string]$ProjectName = "",
	[string]$ProjectDescription = "",
	[string]$PlanningMode = "chapter",
	[string]$WorkflowMode = "classic",
	[bool]$AutoCreateCoverAndPortraits = $true
)

$ErrorActionPreference = "Stop"
if ($PSVersionTable.PSVersion.Major -ge 7) {
	$PSNativeCommandUseErrorActionPreference = $false
}

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
. (Join-Path $ScriptDir "script_common.ps1")

# Editable defaults
$DefaultStoryRequest = "故事发生在一座高级太空站中，太空站收到异族入侵和占领，主角3人因为被关在在隔离区而躲过一劫。男主是团队力量担当，乐观；女主1号是倾国倾城的美丽少女，身材娇小纤细，体味清冷，团队智力担当，傲娇；女主二号同样是美丽异常的少女，善于照顾他人，温柔，体味芬芳。小说故事聚焦于他们合作生存的过程上，从初期的确保自身安全，建立安全据点，确保食物和水源，然后再逐步提升生活水平。总体风格温馨，并加入情感升温。小说应当详细描写他们协力生存的方方面面，尤其重点描写他们搭建/升级安全的避难所。故事情节方面：1、故事的开始是空间站遭到入侵，他们被困在太空站中。他们需要首先应对异族部队的搜捕和清洗，隐藏起来，建立安全据点并确保生存必要条件，等待救援。2、但是显然他们低估了敌人力量，救援似乎不会来。他们必须转战更加安全的地方、搜集并储备大量物资，并尝试资源再生与可持续利用，不断改善生活条件，由生存转向生活。3、安全地点的资源也会耗尽，因此他们决定与敌人游击作战，获取物资和装备。可靠安全地收集更多物资，进一步提高生活水平，并逐步实现可持续。4、新的希望，外部电台发来断续的信号，他们决定去看看。工作内容转向星际载具的偷取与改造。5、..."
$DefaultProjectName = "太空站生存记"
$DefaultProjectDescription = "由模型根据需求自动生成设定的长篇小说项目。"

# Optional runtime overrides
$DefaultModelName = ""
$DefaultApiBase = ""
$DefaultTemperature = "1.0"
$DefaultMaxTokens = "10240"
$DefaultTimeout = ""
$DefaultPlanningMode = "chapter"
$DefaultWorkflowMode = "classic"
$DefaultQualityProvider = ""
$DefaultQualityModelName = ""
$DefaultQualityApiBase = ""
$DefaultQualityTemperature = ""
$DefaultQualityMaxTokens = ""
$DefaultQualityTimeout = ""

if (-not $PSBoundParameters.ContainsKey("Provider")) {
	$Provider = Prompt-OptionalValue -PromptText "Provider (gemini/grok/deepseek/doubao/ollama/llama_cpp)" -DefaultValue $Provider
}
$Provider = Normalize-Provider $Provider

if (-not $PSBoundParameters.ContainsKey("StoryRequest")) {
	$StoryRequest = Prompt-OptionalValue -PromptText "Story request" -DefaultValue $DefaultStoryRequest
}
elseif ([string]::IsNullOrWhiteSpace($StoryRequest)) {
	$StoryRequest = $DefaultStoryRequest
}

if (-not $PSBoundParameters.ContainsKey("ProjectName")) {
	$ProjectName = Prompt-OptionalValue -PromptText "Project name" -DefaultValue $DefaultProjectName
}
elseif ([string]::IsNullOrWhiteSpace($ProjectName)) {
	$ProjectName = $DefaultProjectName
}

if (-not $PSBoundParameters.ContainsKey("ProjectDescription")) {
	$ProjectDescription = Prompt-OptionalValue -PromptText "Project description" -DefaultValue $DefaultProjectDescription
}
elseif ([string]::IsNullOrWhiteSpace($ProjectDescription)) {
	$ProjectDescription = $DefaultProjectDescription
}

if (-not $PSBoundParameters.ContainsKey("PlanningMode")) {
	$PlanningMode = Prompt-OptionalValue -PromptText "Planning mode (none/volume/chapter)" -DefaultValue $DefaultPlanningMode
}
$PlanningMode = Normalize-PlanningMode $PlanningMode

if (-not $PSBoundParameters.ContainsKey("WorkflowMode")) {
	$WorkflowMode = Prompt-OptionalValue -PromptText "Workflow mode (classic/agentic)" -DefaultValue $DefaultWorkflowMode
}
$WorkflowMode = Normalize-WorkflowMode $WorkflowMode

if ([string]::IsNullOrWhiteSpace($StoryRequest)) {
	throw "Usage: .\windows\quick_start.ps1 <provider> <story request> [project name] [project description] [planning mode] [workflow mode]"
}

$pythonExe = Resolve-PythonExe
$apiKeys = Get-ApiKeys -KeysFile (Join-Path $ProjectRoot "api_keys.sh")
$apiKey = if ($env:NOVEL_API_KEY) { $env:NOVEL_API_KEY } else { Get-ApiKeyForProvider -Provider $Provider -ApiKeys $apiKeys }
Ensure-ApiKeyPresent -Provider $Provider -ApiKey $apiKey -ProjectRoot $ProjectRoot

$modelName = if ($env:NOVEL_MODEL_NAME) { $env:NOVEL_MODEL_NAME } elseif ($DefaultModelName) { $DefaultModelName } else { Get-DefaultModelForProvider $Provider }
$apiBase = if ($env:NOVEL_API_BASE) { $env:NOVEL_API_BASE } elseif ($DefaultApiBase) { $DefaultApiBase } else { Get-DefaultApiBaseForProvider $Provider }
$temperature = if ($env:NOVEL_TEMPERATURE) { [double]$env:NOVEL_TEMPERATURE } else { [double]$DefaultTemperature }
$maxTokens = if ($env:NOVEL_MAX_TOKENS) { [int]$env:NOVEL_MAX_TOKENS } else { [int]$DefaultMaxTokens }
$timeout = if ($env:NOVEL_TIMEOUT) { [int]$env:NOVEL_TIMEOUT } elseif ($DefaultTimeout) { [int]$DefaultTimeout } else { Get-DefaultTimeoutForProvider $Provider }
$qualityProvider = if ($env:NOVEL_QUALITY_PROVIDER) { Normalize-Provider $env:NOVEL_QUALITY_PROVIDER } elseif ($DefaultQualityProvider) { Normalize-Provider $DefaultQualityProvider } else { "" }
$qualityModelName = if ($env:NOVEL_QUALITY_MODEL_NAME) { $env:NOVEL_QUALITY_MODEL_NAME } else { $DefaultQualityModelName }
$qualityApiBase = if ($env:NOVEL_QUALITY_API_BASE) { $env:NOVEL_QUALITY_API_BASE } else { $DefaultQualityApiBase }
$qualityTemperature = if ($env:NOVEL_QUALITY_TEMPERATURE) { $env:NOVEL_QUALITY_TEMPERATURE } else { $DefaultQualityTemperature }
$qualityMaxTokens = if ($env:NOVEL_QUALITY_MAX_TOKENS) { $env:NOVEL_QUALITY_MAX_TOKENS } else { $DefaultQualityMaxTokens }
$qualityTimeout = if ($env:NOVEL_QUALITY_TIMEOUT) { $env:NOVEL_QUALITY_TIMEOUT } else { $DefaultQualityTimeout }
$qualityApiKey = if ($qualityProvider) { Get-ApiKeyForProvider -Provider $qualityProvider -ApiKeys $apiKeys } else { "" }
if ($qualityProvider) { Ensure-ApiKeyPresent -Provider $qualityProvider -ApiKey $qualityApiKey -ProjectRoot $ProjectRoot }

$outputRoot = Join-Path $ProjectRoot "output"
$existingProjects = @()
if (Test-Path -LiteralPath $outputRoot) {
	$existingProjects = @(Get-ChildItem -LiteralPath $outputRoot -Directory -ErrorAction SilentlyContinue | ForEach-Object { $_.FullName })
}

$tempConfig = New-TempConfigPath -Prefix "novel_writer_config"

try {
	Write-InitConfig `
		-OutputPath $tempConfig `
		-ProjectRoot $ProjectRoot `
		-ProjectName $ProjectName `
		-ProjectDescription $ProjectDescription `
		-StoryRequest $StoryRequest `
		-Provider $Provider `
		-ModelName $modelName `
		-ApiBase $apiBase `
		-ApiKey $apiKey `
		-Temperature $temperature `
		-MaxTokens $maxTokens `
		-Timeout $timeout `
		-PlanningMode $PlanningMode `
		-WorkflowMode $WorkflowMode `
		-QualityProvider $qualityProvider `
		-QualityModelName $qualityModelName `
		-QualityApiBase $qualityApiBase `
		-QualityApiKey $qualityApiKey `
		-QualityTemperature $qualityTemperature `
		-QualityMaxTokens $qualityMaxTokens `
		-QualityTimeout $qualityTimeout

	$initResult = Invoke-NativeCommandCapture -Executable $pythonExe -Arguments @(
		(Join-Path $ProjectRoot "app.py"),
		"init",
		"--config", $tempConfig
	) -StreamOutput
	if ($initResult.ExitCode -ne 0) {
		throw "Project initialization failed with exit code $($initResult.ExitCode)."
	}

	$newProjects = @()
	if (Test-Path -LiteralPath $outputRoot) {
		$newProjects = @(Get-ChildItem -LiteralPath $outputRoot -Directory -ErrorAction SilentlyContinue |
			Where-Object { $existingProjects -notcontains $_.FullName } |
			Sort-Object LastWriteTime -Descending)
	}

	$projectPath = if ($newProjects.Count -gt 0) { $newProjects[0].FullName } else { Get-LatestProjectPath -OutputRoot $outputRoot }
	if (-not $projectPath) {
		throw "Unable to detect the initialized project path under the output directory."
	}

	if ($AutoCreateCoverAndPortraits) {
		Write-Output "Attempting to auto-generate cover and character portraits..."
		$assetResult = Invoke-NativeCommandCapture -Executable $pythonExe -Arguments @(
			(Join-Path $ProjectRoot "app.py"),
			"illustrate-assets",
			"--project", $projectPath
		) -StreamOutput
		$assetOutput = $assetResult.Output

		if ($assetResult.ExitCode -ne 0) {
			$assetText = ($assetOutput | ForEach-Object { "$_" }) -join "`n"
			if (Test-IllustrationConnectionFailure -Text $assetText) {
				Write-Warning "ComfyUI is not reachable. Skipping automatic cover and portrait generation."
			}
			else {
				throw "Cover or portrait generation failed with exit code $($assetResult.ExitCode)."
			}
		}
	}

	$statusResult = Invoke-NativeCommandCapture -Executable $pythonExe -Arguments @(
		(Join-Path $ProjectRoot "app.py"),
		"status",
		"--project", $projectPath
	) -StreamOutput
	if ($statusResult.ExitCode -ne 0) {
		throw "Status command failed with exit code $($statusResult.ExitCode)."
	}

	Write-Output ("Continue example: .\windows\quick_continue.bat ""{0}"" 3 ""preferred scene""" -f $projectPath)
}
finally {
	Remove-Item -LiteralPath $tempConfig -ErrorAction SilentlyContinue
}
