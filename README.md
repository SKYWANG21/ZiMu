# src — 视频字幕生成 Demo

基于 **faster-whisper** / **SenseVoice** 语音识别与 **ffmpeg** 硬字幕烧录，将 `.mp4` 视频自动转为带字幕的视频。

## 前置条件

- Python 3.10+（推荐 3.10–3.12；3.13 下 torch/funasr 兼容性需自行验证）
- [ffmpeg](https://ffmpeg.org/) 已安装（默认使用 `D:\ffmpeg\bin\ffmpeg.exe`）
- 本地模型（按所选后端准备，需手动下载，程序不会自动在线下载）

```powershell
D:\ffmpeg\bin\ffmpeg.exe -version
```

## 安装

```powershell
# 激活虚拟环境
.\Scripts\Activate.ps1

# 安装全部依赖（含 Whisper 与 SenseVoice）
pip install -r requirements.txt
```

**GPU 用户**：若需 CUDA 加速，请按 [pytorch.org](https://pytorch.org/get-started/locally/) 选择对应 CUDA 版本的 `torch` / `torchaudio` wheel 后再安装其余依赖。

**CPU-only 示例**：

```powershell
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install faster-whisper funasr
```

### 依赖说明

| 包                     | 用途                                                                    |
| ---------------------- | ----------------------------------------------------------------------- |
| `faster-whisper`       | Whisper 转写后端（默认）                                                |
| `torch` / `torchaudio` | SenseVoice 推理引擎                                                     |
| `funasr`               | SenseVoice + VAD 封装（会拉取 scipy、librosa、transformers 等传递依赖） |

## 准备本地模型

### Whisper 后端（默认）

将 [Systran/faster-whisper-large-v3](https://huggingface.co/Systran/faster-whisper-large-v3) 下载到 `D:\models\faster-whisper-large-v3`，目录内应包含 `config.json`、`model.bin`、`tokenizer.json` 等文件。

```powershell
git clone https://hf-mirror.com/Systran/faster-whisper-large-v3 D:\models\faster-whisper-large-v3
```

### SenseVoice 后端

需同时准备主模型与 VAD 模型：

| 模型            | 默认目录                              | 来源                                                                                                                                             |
| --------------- | ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| SenseVoiceSmall | `D:\models\SenseVoiceSmall`           | [ModelScope](https://www.modelscope.cn/models/iic/SenseVoiceSmall) 或 [HuggingFace](https://huggingface.co/FunAudioLLM/SenseVoiceSmall) 手动下载 |
| fsmn-vad        | `D:\models\speech_fsmn_vad_zh-cn-16k` | [ModelScope fsmn-vad](https://modelscope.cn/models/damo/speech_fsmn_vad_zh-cn-16k-common-pytorch) 手动下载                                       |

每个目录应至少包含 `config.yaml` 与 `model.pt`。

## 使用

```powershell
# 默认 Whisper 后端
python main.py path\to\video.mp4

# SenseVoice 后端（使用默认本地模型路径）
python main.py path\to\video.mp4 --backend sensevoice --language zh
```

### 常用参数

| 参数                            | 说明                                                                              |
| ------------------------------- | --------------------------------------------------------------------------------- |
| `-o DIR`                        | 指定输出目录（默认与输入同目录）                                                  |
| `--backend whisper\|sensevoice` | 转写后端（默认 `whisper`）                                                        |
| `--model-path DIR`              | Whisper 本地模型目录（默认 `D:\models\faster-whisper-large-v3`，仅 whisper 后端） |
| `--sensevoice-model-path DIR`   | SenseVoice 本地模型目录（默认 `D:\models\SenseVoiceSmall`）                       |
| `--sensevoice-vad-path DIR`     | VAD 本地模型目录（默认 `D:\models\speech_fsmn_vad_zh-cn-16k`）                    |
| `--language zh`                 | 指定语言，不填则自动检测                                                          |
| `--device cuda`                 | 推理设备：`auto` / `cuda` / `cpu`                                                 |
| `--font "Microsoft YaHei"`      | 烧录字幕字体                                                                      |
| `--font-size 12`                | 字幕字号                                                                          |
| `--max-chars 14`                | 单条字幕最大字数（断句）                                                          |
| `--max-duration 3.5`            | 单条字幕最大时长（秒）                                                            |
| `-v`                            | 调试日志                                                                          |

### 输出文件

以 `demo.mp4` 为例：

- `demo.srt` — 字幕文件
- `demo_subtitled.mp4` — 硬字幕视频（字幕烧录在画面上）

## 项目结构

```
src/
  models.py      # 配置与领域模型
  ffmpeg.py      # ffmpeg 音频提取 / 字幕烧录
  transcribe.py  # Whisper / SenseVoice 转写
  srt.py         # SRT 生成
  segment.py     # 字幕断句后处理
  pipeline.py    # 流水线编排
main.py          # CLI 入口
```

## 处理流程

1. ffmpeg 从 MP4 提取 16 kHz 单声道 WAV
2. 按 `--backend` 选择 Whisper 或 SenseVoice 转写，得到带时间戳的字幕
3. 写入 SRT 文件
4. ffmpeg 将 SRT 硬烧录到视频画面
