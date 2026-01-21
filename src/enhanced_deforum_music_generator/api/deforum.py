from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter

from ..public_api import AudioAnalysis, DeforumMusicGenerator

router = APIRouter()


@router.post("/generate-deforum")
async def generate_deforum(payload: Dict[str, Any]):
    """Generate Deforum-ready settings JSON based on audio analysis + user input.

    Backward compatible:
    - If payload contains {"analysis": {...}, "settings": {...}} it uses both.
    - Otherwise, payload itself is treated as "settings" with an empty analysis.
    """
    generator = DeforumMusicGenerator()

    analysis_data = (payload or {}).get("analysis")
    settings = (payload or {}).get("settings")

    if isinstance(analysis_data, dict) and isinstance(settings, dict):
        analysis = AudioAnalysis(**analysis_data)
        return generator.build_deforum_settings(analysis, settings)

    analysis = AudioAnalysis(duration=0.0, beats=[], energy=[])
    return generator.build_deforum_settings(analysis, payload or {})
