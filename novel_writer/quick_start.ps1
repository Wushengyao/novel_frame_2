Param(
	[string]$Provider = "gemini",
	[string]$StoryRequest = """请你以如下内容为灵感，进行小说设定：故事发生在一座高级奢华的校园中，3位主角都是学生。男主是团队力量担当，乐观；女主1号是倾国倾城的美丽少女，身材娇小纤细，团队智力担当，傲娇；女主二号同样美丽动人，善于照顾他人，温柔。小说故事聚焦于他们合作生存的过程上，从初期的保暖，到逐步确保水源和食物来源，然后再逐步提升生活水平。总体风格温馨，并加入情感升温。
故事的开始是放假期间只有主角们在校，突然极寒天气与暴风雪来临，他们被困在学校中。一开始他们认为只是短暂的极端天气很快会有救援，所以在只是团聚在女生宿舍避寒并且做了短期规划。但是显然他们低估了极寒风暴的力量，温度持续下降，救援也不会来。他们必须转战更加保暖的地方御寒（比如桑拿房）、搜集并储备大量物资，并尝试资源再生与可持续利用，不断改善生活条件，由生存转向生活。小说应当详细描写他们协力生存的方方面面，并且包括过程中的感情升温与适量的香艳情节。
请注意：
1、小说需要具备长篇潜力。
2、现在你只需要给出小说设定，而不要写正文。""",
	[string]$ProjectName = "极寒校园生存记",
	[string]$ProjectDescription = "小说"
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

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
	switch (($Name | ForEach-Object { $_.ToLowerInvariant() })) {
		"gemini" { return "gemini" }
		"grok" { return "grok" }
		"deepseek" { return "deepseek" }
		default { throw "Unsupported provider: $Name (allowed: gemini / grok / deepseek)" }
	}
}

function Default-ModelForProvider {
	param([string]$Name)
	switch ($Name) {
		"gemini" { return "gemini-3.1-pro-preview" }
		"grok" { return "grok-4.20-beta-latest-non-reasoning" }
		"deepseek" { return "deepseek-chat" }
	}
}

function Default-ThinkingLevel {
	param([string]$Name)
	if ($Name -eq "gemini") { return "medium" }
	return ""
}

function Get-LatestProjectPath {
	param([string]$OutputRoot)
	$latest = Get-ChildItem -Path $OutputRoot -Directory -ErrorAction SilentlyContinue |
		Where-Object { $_.Name -like 'novel_project_*' } |
		Sort-Object LastWriteTime -Descending |
		Select-Object -First 1
	if ($latest) {
		return $latest.FullName
	}
	return ""
}

if (-not $PSBoundParameters.ContainsKey("Provider")) {
	$Provider = Prompt-OptionalValue -PromptText "Provider (gemini/grok/deepseek)" -DefaultValue $Provider
}
$Provider = Normalize-Provider $Provider

if (-not $PSBoundParameters.ContainsKey("StoryRequest")) {
	$StoryRequest = Prompt-OptionalValue -PromptText "Story request" -DefaultValue $StoryRequest
}
if (-not $PSBoundParameters.ContainsKey("ProjectName")) {
	$ProjectName = Prompt-OptionalValue -PromptText "Project name" -DefaultValue $ProjectName
}
if (-not $PSBoundParameters.ContainsKey("ProjectDescription")) {
	$ProjectDescription = Prompt-OptionalValue -PromptText "Project description" -DefaultValue $ProjectDescription
}

$pythonExe = Resolve-PythonExe
$apiKeys = Get-ApiKeys -KeysFile (Join-Path $ScriptDir "api_keys.sh")

$apiKey = $env:NOVEL_API_KEY
if (-not $apiKey) {
	switch ($Provider) {
		"gemini" { $apiKey = $apiKeys["GEMINI_API_KEY"] }
		"grok" { $apiKey = $apiKeys["GROK_API_KEY"] }
		"deepseek" { $apiKey = $apiKeys["DEEPSEEK_API_KEY"] }
	}
}
if (-not $apiKey) {
	throw "provider=$Provider missing API key. Please fill $ScriptDir\api_keys.sh"
}

$modelName = if ($env:NOVEL_MODEL_NAME) { $env:NOVEL_MODEL_NAME } else { Default-ModelForProvider $Provider }
$apiBase = if ($env:NOVEL_API_BASE) { $env:NOVEL_API_BASE } else { "" }
$temperature = if ($env:NOVEL_TEMPERATURE) { [double]$env:NOVEL_TEMPERATURE } else { 1.0 }
$maxTokens = if ($env:NOVEL_MAX_TOKENS) { [int]$env:NOVEL_MAX_TOKENS } else { 10240 }
$timeout = if ($env:NOVEL_TIMEOUT) { [int]$env:NOVEL_TIMEOUT } else { 120 }
$thinkingLevel = if ($env:NOVEL_THINKING_LEVEL) { $env:NOVEL_THINKING_LEVEL } else { Default-ThinkingLevel $Provider }

$outputRoot = Join-Path $ScriptDir "output"
if (-not (Test-Path $outputRoot)) {
	New-Item -Path $outputRoot -ItemType Directory | Out-Null
}

$config = [ordered]@{
	project_name = $ProjectName
	project_description = $ProjectDescription
	project_path = (Join-Path $outputRoot "novel_project_{project_id}")
	init_with_llm = $true
	story_request = $StoryRequest
	model_provider = $Provider
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

$tempConfig = [System.IO.Path]::Combine([System.IO.Path]::GetTempPath(), ("novel_writer_config_{0}.json" -f ([guid]::NewGuid().ToString("N"))))
[System.IO.File]::WriteAllText($tempConfig, ($config | ConvertTo-Json -Depth 10), [System.Text.UTF8Encoding]::new($false))

try {
	$initOutput = & $pythonExe (Join-Path $ScriptDir "app.py") init --config $tempConfig
	$initOutput | ForEach-Object { Write-Output $_ }

	$projectPath = Get-LatestProjectPath -OutputRoot $outputRoot
	if (-not $projectPath) {
		throw "Unable to detect initialized project path under output directory."
	}

	& $pythonExe (Join-Path $ScriptDir "app.py") status --project $projectPath
	Write-Output ("Continue example: .\quick_continue.bat ""{0}"" 3 ""想看的情节""" -f $projectPath)
}
finally {
	Remove-Item -Path $tempConfig -ErrorAction SilentlyContinue
}

