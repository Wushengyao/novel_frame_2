# 小说自动续写框架（MVP）

这是一个基于 Python 标准库实现的小说自动续写工具，核心思路是：

- 用 `world.json`、`characters.json`、`plot_state.json` 保存结构化记忆
- 新增 `outlines.json` 保存“分卷 -> 分章”两段式大纲
- 每次生成都带上最近正文，保持文风连续
- 每章生成后自动更新剧情状态
- 每章生成后自动保存状态快照，支持按章节回滚再继续写
- 初始化阶段会先让模型根据你的故事需求具体化并创造人物和背景设定，再生成完整设定与分卷/分章规划

## 现在的推荐用法

为了减少日常使用的配置负担，当前推荐方案是：

- 不再手动维护 `config.json` 这类启动配置文件
- 只保留四个常用脚本：
  - `linux/quick_start.sh`：Linux 下只负责初始化项目
  - `linux/quick_outline.sh`：Linux 下负责重生成分卷/分章大纲
  - `linux/quick_continue.sh`：Linux 下只负责续写已有项目
  - `linux/quick_rollback.sh`：Linux 下回滚到指定章节状态
- API key 单独放在 `api_keys.sh`
- ComfyUI、VoxCPM2 这类外部服务地址和本地路径放在 `external_services.json`
- 参数优先直接写在脚本顶部的 `Editable Parameters` 区域
- 脚本内部会按这些参数临时生成运行配置

你仍然可以直接调用 `app.py` 的 `--config` 工作流，但这已经不是默认推荐方式。`config.json` 更适合一次性 LLM 运行配置；部署环境相关的外部接口建议统一放进 `external_services.json`。

## 目录结构

```text
novel_writer/
  app.py
  llm_client.py
  prompt_builder.py
  project_manager.py
  illustration_manager.py
  external_services.py
  external_services.example.json
  state_updater.py
  webui.py
  api_keys.sh
  webui_auth.env
  linux/
    quick_start.sh
    quick_outline.sh
    quick_continue.sh
    quick_rollback.sh
    quick_illustrate.sh
    quick_webui.sh
    script_common.sh
  windows/
    quick_start.bat
    quick_start.ps1
    quick_continue.bat
    quick_continue.ps1
    quick_rollback.bat
    quick_rollback.ps1
    quick_illustrate.bat
    quick_illustrate.ps1
    quick_webui.bat
    quick_webui.ps1
  README.md
```

初始化后会生成：

```text
output/
  novel_project_<project_id>/
    project.json
    story_setup.json
    world.json
    characters.json
    plot_state.json
    style.json
    outlines.json
    chapters/
    summaries/
    illustrations/
    audiobook/
    snapshots/
```

## 环境要求

- Python 3.10+
- 仅使用标准库

## 支持的模型后端

- `gemini`
- `grok`
- `deepseek`
- `doubao`
- `ollama`
- `llama_cpp`
- `openai_compatible`

当前脚本默认主要面向：

- `gemini`
- `grok`
- `deepseek`
- `doubao`
- `ollama`
- `llama_cpp`

## 1. 配置 API key

编辑 [api_keys.sh](/home/wsy/novel_frame_2/novel_writer/api_keys.sh)：

```bash
export GEMINI_API_KEY="你的 Gemini Key"
export GROK_API_KEY="你的 xAI Key"
export DEEPSEEK_API_KEY="你的 DeepSeek Key"
export DOUBAO_API_KEY="你的豆包 / 火山方舟 Key"
export LLAMA_CPP_API_KEY=""
```

不用的 provider 可以留空。

如果你使用本地 `ollama`，可以不填写任何 API key；脚本和 Web UI 默认会连接：

```text
http://127.0.0.1:11434/v1
```

如果你使用本地 `llama.cpp` server，也可以不填写 API key；选择 `llama_cpp` 后默认会连接：

```text
http://127.0.0.1:8080/v1
```

同时，针对本地 Ollama / llama.cpp，初始化、续写和 Web UI 现在默认会使用更长的请求超时（`900` 秒），避免长章节在生成过程中被过早判定超时。

建议先确认本地服务和模型都已准备好，例如：

```bash
ollama serve
ollama pull llama3.2
```

llama.cpp 可用类似方式启动 OpenAI-compatible server：

```bash
llama-server -m /path/to/model.gguf --host 127.0.0.1 --port 8080
```

## 2. 外部服务配置（部署迁移）

外部服务统一由 [external_services.py](/home/wsy/novel_frame_2/novel_writer/external_services.py) 读取配置。首次部署时复制示例文件：

```bash
cd /home/wsy/novel_frame_2/novel_writer
cp external_services.example.json external_services.json
```

`external_services.json` 已被 `.gitignore` 忽略，适合保存每台机器不同的 ComfyUI 地址、workflow 路径、checkpoint、VoxCPM2 环境路径等部署参数。如果你想把配置文件放到别处：

```bash
export NOVEL_EXTERNAL_SERVICES_CONFIG=/your/path/external_services.json
```

覆盖优先级：

```text
命令行参数 / Web 表单临时参数 > NOVEL_* 环境变量 > external_services.json > 项目中保存的历史配置 > 内置默认值和自动探测
```

常用配置片段：

```json
{
  "comfyui": {
    "api_base": "http://127.0.0.1:8188",
    "root": "/home/wsy/ComfyUI_cu128_50XX/ComfyUI",
    "workflow_template": "/home/wsy/ComfyUI_cu128_50XX/workflow/image_z_image_turbo (2).json",
    "checkpoint": "illusious/illustrij_v21.safetensors",
    "width": 1280,
    "height": 1280,
    "steps": 8,
    "cfg": 1.0
  },
  "voxcpm2": {
    "root": "/home/wsy/VoxCPM2",
    "python": "/home/wsy/VoxCPM2/.venv/bin/python",
    "model_id": "openbmb/VoxCPM2",
    "device": "auto"
  }
}
```

说明：

- `comfyui.api_base` 是 ComfyUI 的 HTTP API 地址
- `comfyui.root` 用于自动寻找 `models/checkpoints/`
- `comfyui.workflow_template` 可直接指定 workflow JSON
- `voxcpm2.python` 是 VoxCPM2 虚拟环境里的 Python
- `voxcpm2.root` 会自动加入 worker 的 `PYTHONPATH`
- API key 仍然放在 `api_keys.sh`，不要写进 `external_services.json`

## 3. 初始化新项目

`linux/quick_start.sh` 现在只做一件事：初始化。

最推荐的方式是先编辑脚本顶部的 `Editable Parameters`：

```bash
DEFAULT_PROVIDER="gemini"
DEFAULT_STORY_REQUEST="现代奢华校园中，男女主在寒假被暴风雪困住，从保暖求生开始逐步建立长期生活。"
DEFAULT_PROJECT_NAME="雪封穹顶"
DEFAULT_PROJECT_DESCRIPTION="由模型根据需求自动生成设定的长篇小说项目。"
DEFAULT_OUTLINE_REQUEST=""
```

然后直接运行：

```bash
./linux/quick_start.sh
```

用法：

```bash
./linux/quick_start.sh <provider> "<故事需求>" [项目名] [项目简介] [大纲额外要求]
```

命令行参数仍然可用，但现在更适合作为临时覆盖。

示例：

```bash
./linux/quick_start.sh gemini "现代奢华校园中，男女主在寒假被暴风雪困住，从保暖求生开始逐步建立长期生活。"
```

或者：

```bash
./linux/quick_start.sh deepseek "三人小队在封闭校园里进行长期生存建设，要求注重水源、食物和保温细节。" "雪封穹顶"
```

或：

```bash
./linux/quick_start.sh doubao "极寒校园中的长期生存故事，要求兼顾生活建设、人物互动与细节描写。" "雪封穹顶"
```

或直接使用本地 Ollama：

```bash
./linux/quick_start.sh ollama "极寒校园中的长期生存故事，要求兼顾生活建设、人物互动与细节描写。" "雪封穹顶"
```

或使用 8080 端口上的本地 llama.cpp：

```bash
./linux/quick_start.sh llama_cpp "极寒校园中的长期生存故事，要求兼顾生活建设、人物互动与细节描写。" "雪封穹顶"
```

初始化时脚本会：

1. 按 provider 读取 `api_keys.sh` 中对应的 API key（`ollama` / `llama_cpp` 可为空）
2. 根据 provider 自动选择默认模型
3. 临时生成运行配置
4. 调用 `app.py init`
5. 先生成分卷大纲，再基于分卷大纲生成每卷分章大纲
6. 输出新项目路径
7. 显示项目状态

通过脚本初始化时，新项目默认会创建在 [output](/home/wsy/novel_frame_2/novel_writer/output) 目录下。

## 4. 续写已有项目

`linux/quick_continue.sh` 只负责续写。

续写时，系统现在会自动读取“所属卷大纲 + 当前章纲 + plot_state + 最近正文”。
其中只要当前章纲存在，正文生成就以“当前章纲”为唯一章节任务来源，`plot_state.next_chapter_goal` 只作为无章纲时的兜底字段，不再重复给模型第二份章节目标。
同时，系统会在续写前先为“当前最后一章”的状态补一个快照，续写成功后再为新章节状态写一个新快照。

同样推荐先编辑脚本顶部的 `Editable Parameters`：

```bash
DEFAULT_PROJECT_PATH="./output/novel_project_xxx"
DEFAULT_CHAPTER_COUNT="3"
DEFAULT_USER_REQUEST="想先推进食堂据点建设，并增加一点轻松互怼的互动。"
DEFAULT_PROVIDER_OVERRIDE=""
```

然后直接运行：

```bash
./linux/quick_continue.sh
```

用法：

```bash
./linux/quick_continue.sh <项目目录> [续写章节数] [用户额外要求] [provider覆盖]
```

命令行参数仍然可用，但现在更适合作为临时覆盖。

示例：

```bash
./linux/quick_continue.sh ./output/novel_project_20260318T022023Z_a3f280b2
```

默认会：

- 续写 3 章
- 不额外指定情节要求
- 使用项目已有的 provider 配置

带额外要求的示例：

```bash
./linux/quick_continue.sh ./output/novel_project_20260318T022023Z_a3f280b2 2 "想先推进食堂据点建设，并增加一点轻松互怼的互动。"
```

如果你想临时换模型后端，也可以加第四个参数：

```bash
./linux/quick_continue.sh ./output/novel_project_20260318T022023Z_a3f280b2 2 "这几章想更注重生存细节" deepseek
```

也可以临时切到豆包：

```bash
./linux/quick_continue.sh ./output/novel_project_20260318T022023Z_a3f280b2 2 "想让人物互动更细腻" doubao
```

说明：

- 如果不传第四个参数，脚本会读取项目里保存的 `model_provider`
- 如果传了新的 provider，脚本会自动用 `api_keys.sh` 中对应的 key
- 如果切到 `ollama` 或 `llama_cpp`，则默认不要求 API key

## 5. 回滚到指定章节

现在支持把项目回滚到“第 N 章写完后的状态”，回滚后可以直接继续 `continue`。

最推荐的方式是先编辑 `linux/quick_rollback.sh` 顶部的参数：

```bash
DEFAULT_PROJECT_PATH="./output/novel_project_xxx"
DEFAULT_TARGET_CHAPTER="4"
```

然后直接运行：

```bash
./linux/quick_rollback.sh
```

用法：

```bash
./linux/quick_rollback.sh <项目目录> <保留到第几章>
```

示例：

```bash
./linux/quick_rollback.sh ./output/novel_project_xxx 4
```

如果你觉得“第 5 章开始写歪了”，那就回滚到第 4 章：

```bash
python3 app.py rollback --project ./output/novel_project_xxx --to-chapter 4
```

回滚时会：

- 恢复 `world.json`、`characters.json`、`plot_state.json`、`style.json`、`outlines.json` 的对应状态
- 把 `project.json` 里的 `chapter_count` 改回目标章节数
- 删除目标章节之后的 `chapters/chapter_xxxx.md`
- 删除目标章节之后的 `summaries/summary_xxxx.json`
- 删除目标章节之后的 `illustrations/chapter_xxxx/`
- 删除目标章节之后的 `snapshots/chapter_xxxx/`
- 最后重新同步 `outlines.json` 进度和 `plot_state.next_chapter_goal`

兼容说明：

- 新项目会从初始化开始自动保存章节快照，回滚是精确的
- 旧项目如果之前还没有快照，系统会优先尝试用 `snapshots/` 恢复；缺失时会退化为用 `summary_xxxx.json` 做 best-effort 恢复
- 对旧项目来说，只要升级后再继续写一次，系统就会先补当前状态快照，之后再回滚会更稳

## 6. 重生成分卷 / 分章大纲

`linux/quick_outline.sh` 用来单独重生成大纲，支持写入你想看的剧情、节奏或限制条件。

最推荐的方式是先编辑脚本顶部的 `Editable Parameters`：

```bash
DEFAULT_PROJECT_PATH="./output/novel_project_xxx"
DEFAULT_STAGE="all"
DEFAULT_USER_REQUEST="第一卷更注重据点建设，第二卷强化人物关系和外部探索。"
DEFAULT_VOLUME_NUMBER=""
DEFAULT_PROVIDER_OVERRIDE=""
```

然后直接运行：

```bash
./linux/quick_outline.sh
```

用法：

```bash
./linux/quick_outline.sh <项目目录> [volumes|chapters|all] [大纲额外要求] [卷号] [provider覆盖]
```

示例：

```bash
./linux/quick_outline.sh ./output/novel_project_xxx all "想让第二卷更早引出外部威胁"
```

只重生成分章大纲：

```bash
./linux/quick_outline.sh ./output/novel_project_xxx chapters "想补强暧昧互动和据点经营细节"
```

只重生成第 2 卷的分章大纲：

```bash
./linux/quick_outline.sh ./output/novel_project_xxx chapters "第二卷希望推进探索线" 2
```

说明：

- 如果你先重生成了分卷大纲，再直接续写，系统会提示先同步分章大纲
- 这样可以避免“卷纲已经换了，但章纲还是旧的”造成正文跑偏

## 7. Web UI

现在项目已经带了一个基础 Web UI，支持：

- 浏览 `output/` 里的全部小说项目
- 在线阅读章节
- 查看当前 `plot_state`
- 直接在网页里续写
- 在网页里新建项目
- 用 ComfyUI 为章节生成插图并在网页中浏览
- 用 VoxCPM2 为章节生成有声版并在网页中播放
- 生成“下一章推进选项”，再按选项续写
- 在页面里直接重启 Web UI，或拉取代码更新后自动重启

启动方式：

```bash
cd /home/wsy/novel_frame_2/novel_writer
./linux/quick_webui.sh
```

默认会监听：

```text
http://0.0.0.0:8008
```

如果你只想本机访问，也可以：

```bash
python3 ./webui.py --host 127.0.0.1 --port 8008
```

如果想从局域网或公网访问，请保持：

```bash
python3 ./webui.py --host 0.0.0.0 --port 8008
```

然后确保服务器防火墙或安全组放行对应端口。

### Web UI 登录鉴权

如果你需要远程访问，推荐开启 Web UI 自带的简单登录鉴权。

默认配置文件现在放在项目内，并且已经被 `.gitignore` 忽略：

- [webui_auth.env](/home/wsy/novel_frame_2/novel_writer/webui_auth.env)

常用配置示例：

```bash
NOVEL_WRITER_AUTH_ENABLED=1
NOVEL_WRITER_AUTH_USERNAME=admin
NOVEL_WRITER_AUTH_PASSWORD='请改成你自己的强密码'
NOVEL_WRITER_AUTH_SECRET_KEY='请改成你自己的随机长字符串'
NOVEL_WRITER_AUTH_COOKIE_NAME=novel_writer_webui_session
NOVEL_WRITER_AUTH_COOKIE_SECURE=0
NOVEL_WRITER_AUTH_SESSION_MAX_AGE_SECONDS=604800
NOVEL_WRITER_AUTH_LOGIN_MAX_ATTEMPTS=5
NOVEL_WRITER_AUTH_LOGIN_WINDOW_SECONDS=300
NOVEL_WRITER_AUTH_LOGIN_LOCKOUT_SECONDS=900
```

说明：

- `NOVEL_WRITER_AUTH_ENABLED=1` 表示启用登录鉴权
- `USERNAME / PASSWORD / SECRET_KEY` 建议你部署时务必改成自己的值
- 当前如果你还是用 `http://` 访问，`NOVEL_WRITER_AUTH_COOKIE_SECURE` 应保持为 `0`
- 如果后面接了 HTTPS，再把 `NOVEL_WRITER_AUTH_COOKIE_SECURE` 改成 `1`
- 如果你想把鉴权文件放到别处，也可以设置环境变量 `NOVEL_WRITER_AUTH_CONFIG=/your/path/webui_auth.env`

启用后，远程打开 Web UI 会先进入登录页；登录成功后，才能：

- 浏览项目和章节
- 直接续写与回滚
- 使用“重启 Web UI”
- 使用“拉取更新并重启”

如果你只是本机临时使用，也可以把：

```bash
NOVEL_WRITER_AUTH_ENABLED=0
```

然后重启 Web UI。

## 8. ComfyUI 插图能力

现在项目支持把章节正文送入 ComfyUI 生成插图：

- `app.py illustrate`：为指定章节或全部章节生成插图
- `app.py next --illustrate`：续写完后立即为本批新章节配图
- `windows/quick_illustrate.ps1` / `windows/quick_illustrate.bat`：Windows 下快速给章节配图
- Web UI 项目页与章节页都可以直接触发插图生成

推荐先在 `external_services.json` 中配置 ComfyUI。未配置时，系统会继续尝试自动寻找同级目录中的 ComfyUI 安装，例如：

```text
../ComfyUI_cu128_50XX/ComfyUI
```

默认会尝试从 `ComfyUI/models/checkpoints/` 中挑选合适的 checkpoint；如果自动识别失败，可在 `external_services.json` 中填写 `comfyui.checkpoint`，也可通过环境变量覆盖：

- `NOVEL_COMFYUI_ROOT`
- `NOVEL_COMFYUI_API_BASE`
- `NOVEL_COMFYUI_CHECKPOINT`
- `NOVEL_COMFYUI_WIDTH`
- `NOVEL_COMFYUI_HEIGHT`
- `NOVEL_COMFYUI_STEPS`
- `NOVEL_COMFYUI_CFG`
- `NOVEL_COMFYUI_TIMEOUT`

示例：

```bash
python3 app.py illustrate --project ./output/novel_project_xxx --chapter chapter_0001 --checkpoint illusious/illustrij_v21.safetensors
```

或在续写后自动配图：

```bash
python3 app.py next --project ./output/novel_project_xxx --config ./runtime_config.json --count 1 --illustrate
```

插图会保存到项目目录下的 `illustrations/chapter_xxxx/` 中，并生成对应的 `metadata.json` 记录提示词和生成参数。

## 9. VoxCPM2 有声小说

项目支持把章节拆分成旁白、对话和内心独白片段，再通过 VoxCPM2 worker 或 Audio Frame 合成为 WAV。

有声小说有两种生成模式：

- `advanced`：默认模式。旁白使用用户选择的旁白预设；人物根据 `characters.json` 中的人物设定生成各自音色。
- `simple`：整章统一音色。所有旁白、对话和内心独白都使用同一个旁白预设音色。

为了避免同一角色在不同段落里音色漂移，系统会先为每种实际使用的音色生成项目级参考 WAV，后续片段都克隆这段参考音频。用户上传的旁白或角色参考 WAV 会优先使用，不会被自动参考音频覆盖。

推荐先在 `external_services.json` 中配置：

```json
{
  "voxcpm2": {
    "root": "/home/wsy/VoxCPM2",
    "python": "/home/wsy/VoxCPM2/.venv/bin/python",
    "model_id": "openbmb/VoxCPM2",
    "device": "auto",
    "silence_ms": 260
  }
}
```

命令行示例：

```bash
python3 app.py audiobook --project ./output/novel_project_xxx --chapter latest
```

简单模式示例：

```bash
python3 app.py audiobook --project ./output/novel_project_xxx --chapter latest --audiobook-mode simple
```

临时覆盖示例：

```bash
python3 app.py audiobook --project ./output/novel_project_xxx --chapter chapter_0001 --voxcpm-root /srv/VoxCPM2 --voxcpm-python /srv/VoxCPM2/.venv/bin/python --voxcpm-model-id /models/VoxCPM2
```

也可以用环境变量覆盖：

- `NOVEL_VOXCPM2_ROOT`
- `NOVEL_VOXCPM2_PYTHON`
- `NOVEL_VOXCPM2_MODEL_ID`
- `NOVEL_VOXCPM2_DEVICE`
- `NOVEL_VOXCPM2_TIMEOUT_SECONDS`

Web UI 的章节页和项目页也可以直接生成有声章节，并支持上传旁白或角色参考 WAV。生成结果会保存到项目目录下的 `audiobook/chapter_xxxx/`，并写入 `manifest.json`。

## 10. 默认模型与可覆盖项

脚本会为不同 provider 自动选择默认模型：

- `gemini` -> `gemini-3.1-flash-lite-preview`
- `grok` -> `grok-4.20-beta-latest-non-reasoning`
- `deepseek` -> `deepseek-v4-flash`
- `doubao` -> `doubao-seed-1-8-251228`
- `ollama` -> `llama3.2`
- `llama_cpp` -> `local-model`

其中豆包默认会使用火山方舟 Ark Chat API：

- `api_base` -> `https://ark.cn-beijing.volces.com/api/v3`

其中本地 Ollama 默认会使用 OpenAI 兼容入口：

- `api_base` -> `http://127.0.0.1:11434/v1`

其中本地 llama.cpp 默认会使用 8080 端口的 OpenAI 兼容入口：

- `api_base` -> `http://127.0.0.1:8080/v1`

说明：

- 豆包的 `model_name` 建议填写模型 ID 或你的 Endpoint ID
- 如果留空，脚本和 Web UI 会默认使用 `doubao-seed-1-8-251228`

你也可以通过脚本顶部参数块长期保存这些设置，或者通过环境变量临时覆盖：

初始化时可用：

- `NOVEL_MODEL_NAME`
- `NOVEL_API_BASE`
- `NOVEL_TEMPERATURE`
- `NOVEL_MAX_TOKENS`
- `NOVEL_TIMEOUT`
- `NOVEL_OUTLINE_REQUEST`

续写时可用：

- `NOVEL_MODEL_NAME_OVERRIDE`
- `NOVEL_API_BASE_OVERRIDE`
- `NOVEL_TEMPERATURE_OVERRIDE`
- `NOVEL_MAX_TOKENS_OVERRIDE`
- `NOVEL_TIMEOUT_OVERRIDE`
- `NOVEL_API_KEY`

## 11. 用户想看的内容

现在有两种入口可以临时加入你的要求：

- 续写正文时，通过 `quick_continue.sh` / `app.py next --user-request`
- 重生成大纲时，通过 `quick_outline.sh` / `app.py outline --user-request`

正文阶段支持临时加入用户偏好，比如：

- 想看的互动
- 想推进的剧情方向
- 想增加的场景元素

如果不传，模型就按当前设定和剧情状态自由发挥。

## 12. 统计信息

项目会在 `project.json` 中累计记录：

- 请求次数
- 成功次数 / 失败次数
- `prompt_tokens`
- `completion_tokens`
- `total_tokens`
- 如果后端提供，还会记录：
  - `cached_tokens`
  - `reasoning_tokens`
  - `thought_tokens`

执行：

```bash
python3 app.py status --project <项目目录>
```

即可看到这些统计信息和当前快照覆盖到第几章。

## 13. 设计说明

当前方案相较之前更轻：

- 你不需要维护多份 `config.*.json`
- API key 不再混在项目配置里
- ComfyUI、VoxCPM2 等外部接口有统一配置文件和封装模块，换机器部署时只改 `external_services.json`
- `linux/quick_start.sh`、`linux/quick_outline.sh`、`linux/quick_continue.sh` 和 `linux/quick_rollback.sh` 职责清晰
- 初始化先做分卷，再做分章，正文写作时会显式参考章纲，整体流程更稳
- 状态快照和章节文件分离，方便回滚后从保留章节继续写新分支
- 仍保留 `app.py --config` 这条底层能力，方便高级场景或后续自动化

## 参考文档

- Gemini `generateContent`: https://ai.google.dev/api/generate-content
- Gemini Thinking: https://ai.google.dev/gemini-api/docs/thinking
- xAI Chat Completions: https://docs.x.ai/developers/model-capabilities/legacy/chat-completions
- DeepSeek Chat Completions: https://api-docs.deepseek.com/api/create-chat-completion
- 火山方舟 / 豆包 Chat API: https://www.volcengine.com/docs/82379/1494384
