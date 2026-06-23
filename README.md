# ZiMu — 视频字幕生成 Demo

基于 **faster-whisper** 语音识别与 **ffmpeg** 硬字幕烧录，将 `.mp4` 视频自动转为带字幕的视频。

## 前置条件

- Python 3.10+（当前环境为 3.13）
- [ffmpeg](https://ffmpeg.org/) 已安装（默认使用 `D:\ffmpeg\bin\ffmpeg.exe`）
- 本地 Whisper 模型（默认目录 `D:\models\faster-whisper-medium`）

```powershell
D:\ffmpeg\bin\ffmpeg.exe -version
```

### 准备本地模型

将 [Systran/faster-whisper-medium](https://huggingface.co/Systran/faster-whisper-medium) 下载到 `D:\models\faster-whisper-medium`，目录内应包含 `config.json`、`model.bin`、`tokenizer.json` 等文件。

```powershell
git clone https://hf-mirror.com/Systran/faster-whisper-medium D:\models\faster-whisper-medium
```

## 安装

```powershell
# 激活虚拟环境
.\Scripts\Activate.ps1

# 依赖已预装时可跳过
pip install -r requirements.txt
```

## 使用

```powershell
python main.py path\to\video.mp4
```

### 常用参数

| 参数                       | 说明                                                            |
| -------------------------- | --------------------------------------------------------------- |
| `-o DIR`                   | 指定输出目录（默认与输入同目录）                                |
| `--model-path DIR`         | 本地 Whisper 模型目录（默认 `D:\models\faster-whisper-medium`） |
| `--language zh`            | 指定语言，不填则自动检测                                        |
| `--device cuda`            | 推理设备：`auto` / `cuda` / `cpu`                               |
| `--font "Microsoft YaHei"` | 烧录字幕字体                                                    |
| `--font-size 24`           | 字幕字号                                                        |
| `-v`                       | 调试日志                                                        |

### 输出文件

以 `demo.mp4` 为例：

- `demo.srt` — 字幕文件
- `demo_subtitled.mp4` — 硬字幕视频（字幕烧录在画面上）

## 项目结构

```
zimu/
  models.py      # 配置与领域模型
  ffmpeg.py      # ffmpeg 音频提取 / 字幕烧录
  transcribe.py  # faster-whisper 转写
  srt.py         # SRT 生成
  pipeline.py    # 流水线编排
main.py          # CLI 入口
```

## 处理流程

1. ffmpeg 从 MP4 提取 16 kHz 单声道 WAV
2. faster-whisper 转写音频，得到带时间戳的字幕
3. 写入 SRT 文件
4. ffmpeg 将 SRT 硬烧录到视频画面
