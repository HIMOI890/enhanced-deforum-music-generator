from __future__ import annotations

import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, UploadFile

from ..core.audio_analyzer import AudioAnalyzer

router = APIRouter()


@router.post("/analyze-audio")
async def analyze_audio(file: UploadFile = File(...), enable_lyrics: bool = True):
    """Analyze an uploaded audio file (tempo, beats, lyrics if enabled)."""
    analyzer = AudioAnalyzer(max_duration=600)
    contents = await file.read()
    suffix = Path(file.filename or "").suffix

    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        result = analyzer.analyze(tmp_path, enable_lyrics=enable_lyrics)
    finally:
        if tmp_path:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    return {
        "tempo_bpm": result.tempo_bpm,
        "duration": result.duration,
        "beats": getattr(result, "beats", []),
        "lyrics": getattr(result, "raw_text", None),
    }
