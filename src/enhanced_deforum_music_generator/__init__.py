"""Enhanced Deforum Music Generator package.

This package exports a **fast-importing** public API from `public_api.py`.
The larger standalone implementation remains available at
`enhanced_deforum_music_generator/enhanced_deforum_music_generator.py`.
"""

from __future__ import annotations

from typing import Any
from importlib import import_module

__all__ = ["DeforumMusicGenerator", "AudioAnalysis"]

_public = import_module(".public_api", __name__)

DeforumMusicGenerator = _public.DeforumMusicGenerator
AudioAnalysis = _public.AudioAnalysis

def __getattr__(name: str) -> Any:
    # Keep a backdoor to the full standalone module without importing it eagerly.
    if name in {"StandaloneDeforumMusicGenerator", "StandaloneAudioAnalysis"}:
        mod = import_module(".enhanced_deforum_music_generator", __name__)
        return getattr(mod, name.replace("Standalone", ""))
    raise AttributeError(name)
