from fastapi import APIRouter
from src.enhanced_deforum_music_generator import DeforumMusicGenerator, AudioAnalysis

router = APIRouter()

@router.post("/generate-deforum")
async def generate_deforum(settings: dict):
    """
    Generate Deforum-ready settings JSON based on audio analysis + user input.
    """
    generator = DeforumMusicGenerator()
    analysis = AudioAnalysis()
    result = generator.build_deforum_settings(analysis, settings)
    return result
