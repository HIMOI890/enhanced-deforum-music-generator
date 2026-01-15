from fastapi import APIRouter, UploadFile, File
from src.enhanced_deforum_music_generator import AudioAnalyzer

router = APIRouter()

@router.post("/analyze-audio")
async def analyze_audio(file: UploadFile = File(...), enable_lyrics: bool = True):
    """
    Analyze an uploaded audio file (tempo, beats, lyrics if enabled).
    """
    analyzer = AudioAnalyzer(max_duration=600)
    contents = await file.read()
    path = f"/tmp/{file.filename}"
    with open(path, "wb") as f:
        f.write(contents)

    result = analyzer.analyze(path, enable_lyrics=enable_lyrics)
    return {
        "tempo_bpm": result.tempo_bpm,
        "duration": result.duration,
        "beats": getattr(result, "beats", []),
        "lyrics": getattr(result, "raw_text", None),
    }
