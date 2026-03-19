# 小说自动续写框架（MVP）

这是一个基于 Python 标准库实现的小说自动续写工具，核心思路是：

- 用 `world.json`、`characters.json`、`plot_state.json` 保存结构化记忆
- 每次生成都带上最近正文，保持文风连续
- 每章生成后自动更新剧情状态
- 初始化阶段可直接让模型根据你的需求生成设定

## 现在的推荐用法

为了减少日常使用的配置负担，当前推荐方案是：

- 不再手动维护 `config.json` 这类启动配置文件
- 只保留两个常用脚本：
  - `quick_start.sh`：只负责初始化项目
  - `quick_continue.sh`：只负责续写已有项目
- API key 单独放在 `api_keys.sh`
- 参数优先直接写在脚本顶部的 `Editable Parameters` 区域
- 脚本内部会按这些参数临时生成运行配置

你仍然可以直接调用 `app.py` 的 `--config` 工作流，但这已经不是默认推荐方式。

## 目录结构

```text
novel_writer/
  app.py
  llm_client.py
  prompt_builder.py
  project_manager.py
  state_updater.py
  webui.py
  quick_start.sh
  quick_continue.sh
  quick_webui.sh
  api_keys.sh
  script_common.sh
  README.md
```

初始化后会生成：

```text
output/
  novel_project_<project_id>/
    project.json
    world.json
    characters.json
    plot_state.json
    style.json
    chapters/
    summaries/
```

## 环境要求

- Python 3.10+
- 仅使用标准库

## 支持的模型后端

- `gemini`
- `grok`
- `deepseek`
- `openai_compatible`

当前脚本默认主要面向：

- `gemini`
- `grok`
- `deepseek`

## 1. 配置 API key

编辑 [api_keys.sh](/home/wsy/novel_frame_2/novel_writer/api_keys.sh)：

```bash
export GEMINI_API_KEY="你的 Gemini Key"
export GROK_API_KEY="你的 xAI Key"
export DEEPSEEK_API_KEY="你的 DeepSeek Key"
```

不用的 provider 可以留空。

## 2. 初始化新项目

`quick_start.sh` 现在只做一件事：初始化。

最推荐的方式是先编辑脚本顶部的 `Editable Parameters`：

```bash
DEFAULT_PROVIDER="gemini"
DEFAULT_STORY_REQUEST="现代奢华校园中，男女主在寒假被暴风雪困住，从保暖求生开始逐步建立长期生活。"
DEFAULT_PROJECT_NAME="雪封穹顶"
DEFAULT_PROJECT_DESCRIPTION="由模型根据需求自动生成设定的长篇小说项目。"
```

然后直接运行：

```bash
./quick_start.sh
```

用法：

```bash
./quick_start.sh <provider> "<故事需求>" [项目名] [项目简介]
```

命令行参数仍然可用，但现在更适合作为临时覆盖。

示例：

```bash
./quick_start.sh gemini "现代奢华校园中，男女主在寒假被暴风雪困住，从保暖求生开始逐步建立长期生活。"
```

或者：

```bash
./quick_start.sh deepseek "三人小队在封闭校园里进行长期生存建设，要求注重水源、食物和保温细节。" "雪封穹顶"
```

初始化时脚本会：

1. 从 `api_keys.sh` 读取对应 provider 的 API key
2. 根据 provider 自动选择默认模型
3. 临时生成运行配置
4. 调用 `app.py init`
5. 输出新项目路径
6. 显示项目状态

通过脚本初始化时，新项目默认会创建在 [output](/home/wsy/novel_frame_2/novel_writer/output) 目录下。

## 3. 续写已有项目

`quick_continue.sh` 只负责续写。

同样推荐先编辑脚本顶部的 `Editable Parameters`：

```bash
DEFAULT_PROJECT_PATH="./output/novel_project_xxx"
DEFAULT_CHAPTER_COUNT="3"
DEFAULT_USER_REQUEST="想先推进食堂据点建设，并增加一点轻松互怼的互动。"
DEFAULT_PROVIDER_OVERRIDE=""
```

然后直接运行：

```bash
./quick_continue.sh
```

用法：

```bash
./quick_continue.sh <项目目录> [续写章节数] [用户额外要求] [provider覆盖]
```

命令行参数仍然可用，但现在更适合作为临时覆盖。

示例：

```bash
./quick_continue.sh ./output/novel_project_20260318T022023Z_a3f280b2
```

默认会：

- 续写 3 章
- 不额外指定情节要求
- 使用项目已有的 provider 配置

带额外要求的示例：

```bash
./quick_continue.sh ./output/novel_project_20260318T022023Z_a3f280b2 2 "想先推进食堂据点建设，并增加一点轻松互怼的互动。"
```

如果你想临时换模型后端，也可以加第四个参数：

```bash
./quick_continue.sh ./output/novel_project_20260318T022023Z_a3f280b2 2 "这几章想更注重生存细节" deepseek
```

说明：

- 如果不传第四个参数，脚本会读取项目里保存的 `model_provider`
- 如果传了新的 provider，脚本会自动用 `api_keys.sh` 中对应的 key

## 4. Web UI

现在项目已经带了一个基础 Web UI，支持：

- 浏览 `output/` 里的全部小说项目
- 在线阅读章节
- 查看当前 `plot_state`
- 直接在网页里续写
- 在网页里新建项目

启动方式：

```bash
cd /home/wsy/novel_frame_2/novel_writer
./quick_webui.sh
```

默认会监听：

```text
http://0.0.0.0:8008
```

如果你只想本机访问，也可以：

```bash
python3 webui.py --host 127.0.0.1 --port 8008
```

如果想从局域网或公网访问，请保持：

```bash
python3 webui.py --host 0.0.0.0 --port 8008
```

然后确保服务器防火墙或安全组放行对应端口。

## 5. 默认模型与可覆盖项

脚本会为不同 provider 自动选择默认模型：

- `gemini` -> `gemini-3.1-flash-lite-preview`
- `grok` -> `grok-4.20-beta-latest-non-reasoning`
- `deepseek` -> `deepseek-chat`

你也可以通过脚本顶部参数块长期保存这些设置，或者通过环境变量临时覆盖：

初始化时可用：

- `NOVEL_MODEL_NAME`
- `NOVEL_API_BASE`
- `NOVEL_TEMPERATURE`
- `NOVEL_MAX_TOKENS`
- `NOVEL_TIMEOUT`
- `NOVEL_THINKING_LEVEL`

续写时可用：

- `NOVEL_MODEL_NAME_OVERRIDE`
- `NOVEL_API_BASE_OVERRIDE`
- `NOVEL_TEMPERATURE_OVERRIDE`
- `NOVEL_MAX_TOKENS_OVERRIDE`
- `NOVEL_TIMEOUT_OVERRIDE`
- `NOVEL_THINKING_LEVEL_OVERRIDE`
- `NOVEL_API_KEY`

例如：

```bash
NOVEL_THINKING_LEVEL=high ./quick_start.sh gemini "现代校园极寒生存故事"
```

## 6. 用户想看的内容

续写时支持临时加入用户偏好，比如：

- 想看的互动
- 想推进的剧情方向
- 想增加的场景元素

如果不传，模型就按当前设定和剧情状态自由发挥。

## 7. 统计信息

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

即可看到这些统计信息。

## 8. 设计说明

当前方案相较之前更轻：

- 你不需要维护多份 `config.*.json`
- API key 不再混在项目配置里
- `quick_start.sh` 和 `quick_continue.sh` 职责清晰
- 仍保留 `app.py --config` 这条底层能力，方便高级场景或后续自动化

## 参考文档

- Gemini `generateContent`: https://ai.google.dev/api/generate-content
- Gemini Thinking: https://ai.google.dev/gemini-api/docs/thinking
- xAI Chat Completions: https://docs.x.ai/developers/model-capabilities/legacy/chat-completions
- DeepSeek Chat Completions: https://api-docs.deepseek.com/api/create-chat-completion
