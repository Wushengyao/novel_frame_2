Param(
	[string]$ProjectPath = "",
	[string]$Chapter = "latest",
	[string]$UserRequest = "",
	[string]$ForceValue = "false",
	[string]$Checkpoint = ""
)

$ErrorActionPreference = "Stop"
if ($PSVersionTable.PSVersion.Major -ge 7) {
	$PSNativeCommandUseErrorActionPreference = $false
}

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
. (Join-Path $ScriptDir "script_common.ps1")

# Optional runtime overrides
$DefaultModelNameOverride = ""
$DefaultApiBaseOverride = ""
$DefaultTemperatureOverride = ""
$DefaultMaxTokensOverride = ""
$DefaultTimeoutOverride = ""

if (-not $PSBoundParameters.ContainsKey("ProjectPath")) {
	$ProjectPath = Prompt-OptionalValue -PromptText "Project directory" -DefaultValue $ProjectPath
}
if (-not $PSBoundParameters.ContainsKey("Chapter")) {
	$Chapter = Prompt-OptionalValue -PromptText "Chapter" -DefaultValue $Chapter
}
if (-not $PSBoundParameters.ContainsKey("UserRequest")) {
	$UserRequest = Prompt-OptionalValue -PromptText "User request (optional)" -DefaultValue $UserRequest
}
if (-not $PSBoundParameters.ContainsKey("ForceValue")) {
	$ForceValue = Prompt-OptionalValue -PromptText "Force regenerate? (true/false)" -DefaultValue $ForceValue
}
if (-not $PSBoundParameters.ContainsKey("Checkpoint")) {
	$Checkpoint = Prompt-OptionalValue -PromptText "Checkpoint (optional)" -DefaultValue $Checkpoint
}

if ([string]::IsNullOrWhiteSpace($ProjectPath)) {
	throw "Usage: .\windows\quick_illustrate.ps1 <project directory> [chapter] [user request] [true|false] [checkpoint]"
}
if (-not (Test-Path -LiteralPath $ProjectPath)) {
	throw "Project directory does not exist: $ProjectPath"
}

$projectFile = Join-Path $ProjectPath "project.json"
if (-not (Test-Path -LiteralPath $projectFile)) {
	throw "Missing project.json in directory: $ProjectPath"
}

$normalizedForceValue = if ($null -eq $ForceValue) { "" } else { $ForceValue }
$shouldForce = switch ($normalizedForceValue.Trim().ToLowerInvariant()) {
	"true" { $true; break }
	"false" { $false; break }
	default { throw "Force regenerate must be true or false." }
}

$pythonExe = Resolve-PythonExe
$apiKeys = Get-ApiKeys -KeysFile (Join-Path $ProjectRoot "api_keys.sh")
$savedProject = Get-Content -LiteralPath $projectFile -Raw -Encoding UTF8 | ConvertFrom-Json
$saved = $savedProject.llm_config
if (-not $saved) { $saved = @{} }
$resolvedProvider = if ($saved.model_provider) { "$($saved.model_provider)".Trim().ToLowerInvariant() } else { "gemini" }
$apiKey = if ($env:NOVEL_API_KEY) { $env:NOVEL_API_KEY } else { Get-ApiKeyForProvider -Provider $resolvedProvider -ApiKeys $apiKeys }
Ensure-ApiKeyPresent -Provider $resolvedProvider -ApiKey $apiKey -ProjectRoot $ProjectRoot

$modelNameOverride = if ($env:NOVEL_MODEL_NAME_OVERRIDE) { $env:NOVEL_MODEL_NAME_OVERRIDE } else { $DefaultModelNameOverride }
$apiBaseOverride = if ($env:NOVEL_API_BASE_OVERRIDE) { $env:NOVEL_API_BASE_OVERRIDE } else { $DefaultApiBaseOverride }
$temperatureOverride = if ($env:NOVEL_TEMPERATURE_OVERRIDE) { $env:NOVEL_TEMPERATURE_OVERRIDE } else { $DefaultTemperatureOverride }
$maxTokensOverride = if ($env:NOVEL_MAX_TOKENS_OVERRIDE) { $env:NOVEL_MAX_TOKENS_OVERRIDE } else { $DefaultMaxTokensOverride }
$timeoutOverride = if ($env:NOVEL_TIMEOUT_OVERRIDE) { $env:NOVEL_TIMEOUT_OVERRIDE } else { $DefaultTimeoutOverride }
$tempConfig = New-TempConfigPath -Prefix "novel_writer_illustrate"

try {
	Write-IllustrateConfig `
		-OutputPath $tempConfig `
		-ProjectPath $ProjectPath `
		-ApiKey $apiKey `
		-ModelNameOverride $modelNameOverride `
		-ApiBaseOverride $apiBaseOverride `
		-TemperatureOverride $temperatureOverride `
		-MaxTokensOverride $maxTokensOverride `
		-TimeoutOverride $timeoutOverride

	$argsList = @(
		(Join-Path $ProjectRoot "app.py"),
		"illustrate",
		"--project", $ProjectPath,
		"--chapter", $Chapter,
		"--config", $tempConfig
	)
	if ($UserRequest) {
		$argsList += @("--user-request", $UserRequest)
	}
	if ($shouldForce) {
		$argsList += "--force"
	}
	if ($Checkpoint) {
		$argsList += @("--checkpoint", $Checkpoint)
	}

	& $pythonExe @argsList
}
finally {
	Remove-Item -LiteralPath $tempConfig -ErrorAction SilentlyContinue
}
