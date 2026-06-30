"""FastAPI HTTP 接口。"""

from __future__ import annotations

import logging
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from zimu.logging_setup import error_log_session
from zimu.models import SUPPORTED_AUDIO_EXTENSIONS
from zimu.pipeline import SubtitlePipeline
from zimu.serialization import pipeline_result_to_dict
from zimu.settings import AppSettings, to_pipeline_config
from zimu.transcribe import SenseVoiceTranscriptionService

logger = logging.getLogger(__name__)


class _AppState:
    """应用运行时状态。"""

    settings: AppSettings
    pipeline: SubtitlePipeline


def create_app(settings: AppSettings) -> FastAPI:
    """根据已加载的配置创建 FastAPI 应用。"""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        transcription = SenseVoiceTranscriptionService(
            model_path=settings.sensevoice_model_path,
            vad_path=settings.sensevoice_vad_path,
            device=settings.device,
        )
        app.state.zimu = _AppState()
        app.state.zimu.settings = settings
        app.state.zimu.pipeline = SubtitlePipeline(transcription=transcription)
        logger.info(
            "转写服务已就绪 (device=%s, model=%s)",
            settings.device,
            settings.sensevoice_model_path,
        )
        yield

    app = FastAPI(title="ZiMu", lifespan=lifespan)

    @app.post("/v1/audio/transcriptions")
    async def transcribe_audio(
        file: UploadFile = File(...),
        language: Optional[str] = Form(default=None),
    ) -> JSONResponse:
        """上传音频并返回完整 PipelineResult JSON。"""
        filename = file.filename or "audio.wav"
        suffix = Path(filename).suffix.lower()
        if suffix not in SUPPORTED_AUDIO_EXTENSIONS:
            supported = ", ".join(sorted(SUPPORTED_AUDIO_EXTENSIONS))
            raise HTTPException(
                status_code=400,
                detail=f"不支持的音频格式: {suffix}，支持: {supported}",
            )

        zimu_state: _AppState = app.state.zimu
        temp_path: Optional[Path] = None

        try:
            with error_log_session():
                with tempfile.NamedTemporaryFile(
                    suffix=suffix,
                    delete=False,
                ) as tmp:
                    content = await file.read()
                    tmp.write(content)
                    temp_path = Path(tmp.name)

                config = to_pipeline_config(
                    zimu_state.settings,
                    input_audio=temp_path,
                    language=language,
                )
                result = zimu_state.pipeline.run(config)
                body = pipeline_result_to_dict(result, input_filename=filename)
                return JSONResponse(content=body)
        except HTTPException:
            raise
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("转写失败: %s", exc)
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        finally:
            if temp_path is not None and temp_path.is_file():
                temp_path.unlink(missing_ok=True)

    return app
