# ZiMu — 音频语音转写

基于 **SenseVoice**（FunASR）的语音识别工具，支持 CLI 与 HTTP 服务两种使用方式，可将音频转写为带时间戳的字幕片段。

## 前置条件

- Python 3.10+（推荐 3.10–3.12；3.13 下 torch/funasr 兼容性需自行验证）
- 本地 SenseVoice 与 VAD 模型（需手动下载，程序不会自动在线下载）

## 安装

```powershell
# 激活虚拟环境
.\Scripts\Activate.ps1

pip install -r requirements.txt
```

**GPU 用户**：若需 CUDA 加速，请按 [pytorch.org](https://pytorch.org/get-started/locally/) 选择对应 CUDA 版本的 `torch` / `torchaudio` wheel 后再安装其余依赖。

**CPU-only 示例**：

```powershell
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### 依赖说明

| 包 | 用途 |
| --- | --- |
| `torch` / `torchaudio` | SenseVoice 推理引擎 |
| `funasr` | SenseVoice + VAD 封装 |
| `fastapi` / `uvicorn` / `python-multipart` | HTTP 转写服务 |

## 配置

所有运行参数通过项目根目录的 `config.json` 加载。首次使用请复制模板：

```powershell
Copy-Item config.example.json config.json
```

编辑 `config.json` 中的模型路径与推理参数：

```json
{
  "device": "cpu",
  "max_subtitle_chars": 14,
  "max_subtitle_duration_sec": 3.5,
  "sensevoice_model_path": "D:/models/SenseVoiceSmall",
  "sensevoice_vad_path": "D:/models/speech_fsmn_vad_zh-cn-16k",
  "server": {
    "host": "0.0.0.0",
    "port": 8000
  }
}
```

| 配置项 | 说明 |
| --- | --- |
| `device` | 推理设备：`cpu` / `cuda` / `auto` |
| `max_subtitle_chars` | 单条字幕最大字数（断句） |
| `max_subtitle_duration_sec` | 单条字幕最大时长（秒） |
| `sensevoice_model_path` | SenseVoice 本地模型目录 |
| `sensevoice_vad_path` | FSMN-VAD 本地模型目录 |
| `server.host` / `server.port` | HTTP 服务监听地址与端口 |

`config.json` 已加入 `.gitignore`，每台机器自行维护；仓库内提供 `config.example.json` 作为模板。

## 准备本地模型

需同时准备主模型与 VAD 模型：

| 模型 | 示例目录 | 来源 |
| --- | --- | --- |
| SenseVoiceSmall | `D:/models/SenseVoiceSmall` | [ModelScope](https://www.modelscope.cn/models/iic/SenseVoiceSmall) 或 [HuggingFace](https://huggingface.co/FunAudioLLM/SenseVoiceSmall) |
| fsmn-vad | `D:/models/speech_fsmn_vad_zh-cn-16k` | [ModelScope fsmn-vad](https://modelscope.cn/models/damo/speech_fsmn_vad_zh-cn-16k-common-pytorch) |

每个目录应至少包含 `config.yaml` 与 `model.pt`。

## CLI 使用

支持的音频格式：`.wav`、`.mp3`、`.flac`、`.ogg`、`.m4a`、`.aac`、`.wma`。

```powershell
# 基本用法：转写文本输出到标准输出
python main.py path\to\audio.wav

# 指定语言
python main.py path\to\audio.wav --language zh

# 指定配置文件
python main.py path\to\audio.wav --config path\to\config.json

# 调试日志
python main.py path\to\audio.wav -v
```

### CLI 参数

| 参数 | 说明 |
| --- | --- |
| `input` | 输入音频路径 |
| `--config` | 配置文件路径（默认 `config.json`） |
| `--language` | 指定语言（如 `zh`、`en`），不填则自动检测 |
| `-v` / `--verbose` | 输出调试日志 |

推理设备、字幕断句参数、模型路径均在 `config.json` 中配置，不再通过命令行传入。

## HTTP 服务

启动服务（监听地址与端口默认读取 `config.json` 中的 `server` 段）：

```powershell
python server.py

# 临时覆盖监听地址
python server.py --host 127.0.0.1 --port 8080

# 指定配置文件
python server.py --config path\to\config.json
```

### 转写接口

**`POST /v1/audio/transcriptions`**

请求：`multipart/form-data`

| 字段 | 必填 | 说明 |
| --- | --- | --- |
| `file` | 是 | 音频文件 |
| `language` | 否 | 语言代码，如 `zh`、`en` |

成功响应示例（`200 application/json`）：

```json
{
  "transcription_text": "你好世界",
  "detected_language": "zh",
  "language_probability": 0.95,
  "segments": [
    {"index": 1, "start": 0.0, "end": 1.5, "text": "你好世界"}
  ],
  "config": {
    "input_filename": "demo.wav",
    "language": "zh",
    "device": "cpu",
    "max_subtitle_chars": 14,
    "max_subtitle_duration_sec": 3.5,
    "sensevoice_model_path": "D:/models/SenseVoiceSmall",
    "sensevoice_vad_path": "D:/models/speech_fsmn_vad_zh-cn-16k"
  }
}
```

调用示例：

```powershell
curl -X POST http://localhost:8000/v1/audio/transcriptions `
  -F "file=@demo.wav" `
  -F "language=zh"
```

## 错误日志

程序运行期间会在内存中缓冲全链路日志。若运行成功，不写任何日志文件；若发生异常，自动将完整日志写入 `log/{时间戳}_error.log`。

## 项目结构

```
zimu/
  models.py          # 领域模型与 PipelineConfig
  settings.py        # config.json 配置加载
  logging_setup.py   # 错误日志持久化
  serialization.py   # PipelineResult JSON 序列化
  transcribe.py      # SenseVoice 转写
  segment.py         # 字幕断句后处理
  pipeline.py        # 流水线编排
  api.py             # FastAPI 路由
  ffmpeg.py          # ffmpeg 音频提取工具（可选）
  srt.py             # SRT 时间戳格式化工具
main.py              # CLI 入口
server.py            # HTTP 服务入口
config.example.json  # 配置模板
```

## 处理流程

1. 校验输入音频格式与文件存在性
2. SenseVoice + VAD 转写音频，得到带时间戳的字幕片段
3. 按 `max_subtitle_chars` / `max_subtitle_duration_sec` 断句后处理
4. CLI 输出拼接后的转写文本；HTTP 返回完整 `PipelineResult` JSON

## Docker 部署（预留）

将 `config.json` 与模型目录通过 volume 挂载进容器即可：

```dockerfile
COPY config.example.json /app/config.example.json
VOLUME ["/app/config.json", "/models"]
CMD ["python", "server.py", "--config", "/app/config.json"]
```

容器内 `config.json` 示例：

```json
{
  "device": "cuda",
  "max_subtitle_chars": 14,
  "max_subtitle_duration_sec": 3.5,
  "sensevoice_model_path": "/models/SenseVoiceSmall",
  "sensevoice_vad_path": "/models/speech_fsmn_vad_zh-cn-16k",
  "server": { "host": "0.0.0.0", "port": 8000 }
}
```
