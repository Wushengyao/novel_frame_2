Param(
	[string]$ProjectPath = "",
	[int]$ChapterCount = 3,
	[string]$UserRequest = "",
	[string]$ProviderOverride = "",
	[bool]$AutoIllustrate = $true
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
$DefaultThinkingLevelOverride = ""

if (-not $PSBoundParameters.ContainsKey("ProjectPath")) {
	$ProjectPath = Prompt-OptionalValue -PromptText "Project directory"
}
if ([string]::IsNullOrWhiteSpace($ProjectPath)) {
	throw "Usage: .\windows\quick_continue.ps1 <project directory> [chapter count] [user request] [provider override]"
}
if (-not (Test-Path -LiteralPath $ProjectPath)) {
	throw "Project directory does not exist: $ProjectPath"
}

$projectFile = Join-Path $ProjectPath "project.json"
if (-not (Test-Path -LiteralPath $projectFile)) {
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

$savedProject = Get-Content -LiteralPath $projectFile -Raw -Encoding UTF8 | ConvertFrom-Json
$saved = $savedProject.llm_config
if (-not $saved) { $saved = @{} }
$savedProvider = if ($saved.model_provider) { ("$($saved.model_provider)").ToLowerInvariant() } else { "gemini" }

$ProviderOverride = if ($ProviderOverride) { Normalize-Provider $ProviderOverride } else { "" }
$resolvedProvider = if ($ProviderOverride) { $ProviderOverride } else { $savedProvider }

$pythonExe = Resolve-PythonExe
$apiKeys = Get-ApiKeys -KeysFile (Join-Path $ProjectRoot "api_keys.sh")
$apiKey = if ($env:NOVEL_API_KEY) { $env:NOVEL_API_KEY } else { Get-ApiKeyForProvider -Provider $resolvedProvider -ApiKeys $apiKeys }
Ensure-ApiKeyPresent -Provider $resolvedProvider -ApiKey $apiKey -ProjectRoot $ProjectRoot

$modelNameOverride = if ($env:NOVEL_MODEL_NAME_OVERRIDE) { $env:NOVEL_MODEL_NAME_OVERRIDE } else { $DefaultModelNameOverride }
$apiBaseOverride = if ($env:NOVEL_API_BASE_OVERRIDE) { $env:NOVEL_API_BASE_OVERRIDE } else { $DefaultApiBaseOverride }
$temperatureOverride = if ($env:NOVEL_TEMPERATURE_OVERRIDE) { $env:NOVEL_TEMPERATURE_OVERRIDE } else { $DefaultTemperatureOverride }
$maxTokensOverride = if ($env:NOVEL_MAX_TOKENS_OVERRIDE) { $env:NOVEL_MAX_TOKENS_OVERRIDE } else { $DefaultMaxTokensOverride }
$timeoutOverride = if ($env:NOVEL_TIMEOUT_OVERRIDE) { $env:NOVEL_TIMEOUT_OVERRIDE } else { $DefaultTimeoutOverride }
$thinkingLevelOverride = if ($env:NOVEL_THINKING_LEVEL_OVERRIDE) { $env:NOVEL_THINKING_LEVEL_OVERRIDE } else { $DefaultThinkingLevelOverride }

$chaptersDir = Join-Path $ProjectPath "chapters"
$beforeChapterPaths = @()
if (Test-Path -LiteralPath $chaptersDir) {
	$beforeChapterPaths = @(Get-ChildItem -LiteralPath $chaptersDir -File -Filter "chapter_*.md" -ErrorAction SilentlyContinue | ForEach-Object { $_.FullName })
}

$tempConfig = New-TempConfigPath -Prefix "novel_writer_config"

try {
	Write-ContinueConfig `
		-OutputPath $tempConfig `
		-ProjectPath $ProjectPath `
		-ProviderOverride $ProviderOverride `
		-ApiKey $apiKey `
		-ModelNameOverride $modelNameOverride `
		-ApiBaseOverride $apiBaseOverride `
		-TemperatureOverride $temperatureOverride `
		-MaxTokensOverride $maxTokensOverride `
		-TimeoutOverride $timeoutOverride `
		-ThinkingLevelOverride $thinkingLevelOverride

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

	$nextResult = Invoke-NativeCommandCapture -Executable $pythonExe -Arguments $nextArgs -StreamOutput
	if ($nextResult.ExitCode -ne 0) {
		throw "Chapter generation failed with exit code $($nextResult.ExitCode)."
	}

	$newChapterPaths = @()
	if (Test-Path -LiteralPath $chaptersDir) {
		$newChapterPaths = @(Get-ChildItem -LiteralPath $chaptersDir -File -Filter "chapter_*.md" -ErrorAction SilentlyContinue |
			Where-Object { $beforeChapterPaths -notcontains $_.FullName } |
			Sort-Object Name |
			ForEach-Object { $_.FullName })
	}

	if ($AutoIllustrate -and $newChapterPaths.Count -gt 0) {
		Write-Output "Attempting to auto-generate illustrations..."
		foreach ($chapterPath in $newChapterPaths) {
			$illustrateArgs = @(
				(Join-Path $ProjectRoot "app.py"),
				"illustrate",
				"--project", $ProjectPath,
				"--chapter", $chapterPath,
				"--config", $tempConfig
			)

			$illustrateResult = Invoke-NativeCommandCapture -Executable $pythonExe -Arguments $illustrateArgs -StreamOutput
			$illustrateOutput = $illustrateResult.Output

			if ($illustrateResult.ExitCode -ne 0) {
				$illustrateText = ($illustrateOutput | ForEach-Object { "$_" }) -join "`n"
				if (Test-IllustrationConnectionFailure -Text $illustrateText) {
					Write-Warning "ComfyUI is not reachable. Skipping automatic illustration generation."
					break
				}
				throw "Illustration generation failed with exit code $($illustrateResult.ExitCode)."
			}
		}
	}
	elseif ($AutoIllustrate) {
		Write-Warning "No newly generated chapter files were detected. Skipping automatic illustration generation."
	}

	$statusResult = Invoke-NativeCommandCapture -Executable $pythonExe -Arguments @(
		(Join-Path $ProjectRoot "app.py"),
		"status",
		"--project", $ProjectPath
	) -StreamOutput
	if ($statusResult.ExitCode -ne 0) {
		throw "Status command failed with exit code $($statusResult.ExitCode)."
	}
}
finally {
	Remove-Item -LiteralPath $tempConfig -ErrorAction SilentlyContinue
}
