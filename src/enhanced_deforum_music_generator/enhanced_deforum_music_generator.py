"""
Enhanced Deforum Music Generator - Complete Implementation
Put this file in your Automatic1111 'extensions/<your-extension>/scripts' folder,
or run standalone as `python enhanced_deforum_music_generator.py ui`.
"""

# ------------------------------------------------------------
# Imports and dependency detection
# ------------------------------------------------------------
import os
import sys
import json
import time
import math
import random
import tempfile
import zipfile
import traceback
from typing import Dict, Any, Optional, List, Tuple, Union

# Optional heavy libraries (use if installed)
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except Exception:
    np = None
    NUMPY_AVAILABLE = False

try:
    import librosa
    LIBROSA_AVAILABLE = True
except Exception:
    librosa = None
    LIBROSA_AVAILABLE = False

# Whisper optional for lyrics transcription
try:
    import whisper
    WHISPER_AVAILABLE = True
except Exception:
    whisper = None
    WHISPER_AVAILABLE = False

# SciPy used for peak detection
try:
    from scipy.signal import find_peaks
    SCIPY_AVAILABLE = True
except Exception:
    find_peaks = None
    SCIPY_AVAILABLE = False

# Gradio for UI
try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except Exception:
    gr = None
    GRADIO_AVAILABLE = False

# Automatic1111 environment detection
try:
    import modules.shared as shared  # type: ignore
    import modules.scripts as scripts  # type: ignore
    A1111_AVAILABLE = True
except Exception:
    shared = None
    scripts = None
    A1111_AVAILABLE = False

# script_callbacks for on_ui_tabs
try:
    import modules.script_callbacks as script_callbacks  # type: ignore
    SCRIPT_CALLBACKS_AVAILABLE = True
except Exception:
    script_callbacks = None
    SCRIPT_CALLBACKS_AVAILABLE = False

try:
    try:
        from .enhanced_nlp_ai_module import (
            EnhancedNLPAnalyzer,
            AIPromptGenerator,
            LyricsAnalysis,
            integrate_enhanced_nlp,
        )
    except Exception:
        from enhanced_nlp_ai_module import (
            EnhancedNLPAnalyzer,
            AIPromptGenerator,
            LyricsAnalysis,
            integrate_enhanced_nlp,
        )
    ENHANCED_NLP_AVAILABLE = True
except Exception:
    print("Enhanced NLP module not found. Using basic analysis.")
    ENHANCED_NLP_AVAILABLE = False

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
# Where to put generated packages by default (if running CLI)
DEFAULT_OUTPUT_DIR = None  # None -> tempfile
# Max audio time allowed by default (seconds). You told "10 min max" so default 600
MAX_AUDIO_DURATION = 600

# Default sampler options
DEFAULT_SAMPLERS = ["Euler a", "DPM++ 2M Karras", "DDIM", "LMS"]

# Set random seed for deterministic behaviors during dev if desired
DEVELOPMENT_SEED = None  # set to int to lock behavior for testing

# ------------------------------------------------------------
# Utility helpers
# ------------------------------------------------------------
def safe_makedirs(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def dump_json_file(path: str, data: Any) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def now_timestring() -> str:
    return time.strftime("%Y%m%d_%H%M%S")

# ------------------------------------------------------------
# Audio analysis data structure
# ------------------------------------------------------------
class AudioAnalysis:
    def __init__(self):
        # metadata
        self.filepath: str = ""
        self.duration: float = 0.0
        # tempo / rhythm
        self.tempo_bpm: float = 0.0
        self.beat_frames: List[float] = []
        self.rhythm_pattern: str = ""
        self.tempo_confidence: float = 0.0
        # energy / dynamics
        self.dynamic_range: float = 0.0
        self.energy_segments: List[float] = []
        self.audio_reactive_points: List[Dict[str, Any]] = []
        # spectral features
        self.spectral_features: Dict[str, float] = {}
        # lyrics & semantic analysis
        self.raw_text: str = ""
        self.lyric_emotions: List[str] = []
        self.visual_elements: List[str] = []
        self.enhanced_lyrics: Optional['LyricsAnalysis'] = None
        # miscellaneous
        self.sampling_rate: int = 44100
        self.frames_per_second_for_ui: int = 24

    def to_dict(self) -> Dict[str, Any]:
        return {
            "filepath": self.filepath,
            "duration": self.duration,
            "tempo_bpm": self.tempo_bpm,
            "beat_frames": self.beat_frames,
            "rhythm_pattern": self.rhythm_pattern,
            "tempo_confidence": self.tempo_confidence,
            "dynamic_range": self.dynamic_range,
            "energy_segments": self.energy_segments,
            "audio_reactive_points": self.audio_reactive_points,
            "spectral_features": self.spectral_features,
            "raw_text": self.raw_text,
            "lyric_emotions": self.lyric_emotions,
            "visual_elements": self.visual_elements,
            "sampling_rate": self.sampling_rate,
            "fps_for_ui": self.frames_per_second_for_ui
        }

# ------------------------------------------------------------
# AudioAnalyzer: does beat/energy/spectral/lyrics analysis
# ------------------------------------------------------------
class AudioAnalyzer:
    def __init__(self, max_duration: int = MAX_AUDIO_DURATION):
        self.max_duration = max_duration
        # fallback tempo detection parameters
        self._fallback_tempo_min = 60
        self._fallback_tempo_max = 140

    def analyze(self, audio_path: str, enable_lyrics: bool = False) -> AudioAnalysis:
        analysis = AudioAnalysis()
        analysis.filepath = audio_path

        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        if LIBROSA_AVAILABLE:
            try:
                y, sr = librosa.load(audio_path, sr=None, duration=self.max_duration)
                analysis.sampling_rate = int(sr)
                analysis.duration = float(len(y) / sr)

                # Beat / tempo
                tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, trim=False)
                analysis.tempo_bpm = float(tempo) if tempo is not None else 0.0
                analysis.beat_frames = librosa.frames_to_time(beat_frames, sr=sr).tolist()

                # RMS energy
                hop_length = 512
                frame_length = 2048
                rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
                # Normalize/clip small arrays
                if len(rms) == 0:
                    # fallback
                    rms = np.array([0.0])

                # spectral features
                spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
                spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
                analysis.spectral_features = {
                    "brightness": float(np.mean(spectral_centroids) / (sr / 2)) if NUMPY_AVAILABLE else float(np.mean(spectral_centroids) / (sr / 2)),
                    "warmth": float(np.mean(spectral_rolloff) / (sr / 2))
                }

                # dynamic range
                analysis.dynamic_range = float(np.max(rms) - np.min(rms)) if NUMPY_AVAILABLE else float(np.max(rms) - np.min(rms))

                # segmentation into 8 energy segments (or fewer if very short)
                segments = max(1, min(16, int(analysis.duration // 5)))  # more segments for long songs
                rmsp = np.array(rms)
                segment_length = max(1, int(len(rmsp) / segments))
                analysis.energy_segments = []
                for i in range(segments):
                    s = i * segment_length
                    e = min((i + 1) * segment_length, len(rmsp))
                    seg_mean = float(np.mean(rmsp[s:e])) if e > s else float(rmsp[s])
                    analysis.energy_segments.append(seg_mean)

                # find peaks in RMS for audio reactive points
                if SCIPY_AVAILABLE:
                    peaks, props = find_peaks(rmsp, height=np.mean(rmsp) + np.std(rmsp), distance=4)
                else:
                    # simple local maxima fallback
                    peaks = []
                    for i in range(1, len(rmsp)-1):
                        if rmsp[i] > rmsp[i-1] and rmsp[i] >= rmsp[i+1] and rmsp[i] > np.mean(rmsp):
                            peaks.append(i)
                    peaks = np.array(peaks)

                analysis.audio_reactive_points = []
                max_points = 40
                for idx in peaks[:max_points]:
                    t = librosa.frames_to_time(idx, sr=sr, hop_length=hop_length)
                    frame = int(t * 24)
                    intensity = float(rmsp[idx])
                    analysis.audio_reactive_points.append({"time": float(t), "frame": int(frame), "intensity": float(intensity)})

                # tempo confidence heuristic: more beats -> more confidence
                num_beats = len(beat_frames)
                analysis.tempo_confidence = min(0.98, max(0.2, num_beats / max(1, analysis.duration / 0.5)))

                # basic rhythm pattern guess
                if analysis.tempo_bpm < 70:
                    analysis.rhythm_pattern = "slow"
                elif analysis.tempo_bpm < 120:
                    analysis.rhythm_pattern = "moderate"
                else:
                    analysis.rhythm_pattern = "fast"

                # lyrics analysis with Whisper (optional)
                if enable_lyrics and WHISPER_AVAILABLE:
                    try:
                        # load the small model for speed; change to "base" or "small" if installed
                        model = whisper.load_model("small")
                        # transcribe (Whisper will internally handle sample rate conversion)
                        result = model.transcribe(audio_path, verbose=False)
                        analysis.raw_text = result.get("text", "")
                        emotions, visuals = self._analyze_lyrics(analysis.raw_text)
                        analysis.lyric_emotions = emotions
                        analysis.visual_elements = visuals
                    except Exception as e:
                        print(f"[AudioAnalyzer] Whisper transcription failed: {e}")
                return analysis

            except Exception as e:
                print(f"[AudioAnalyzer] Librosa-based analysis failed: {e}\nFalling back to basic analysis.")
                traceback.print_exc()
                # Fall through to basic analysis

        # Basic fallback analysis (no librosa)
        return self._basic_analysis(audio_path, analysis)

    def _basic_analysis(self, audio_path: str, analysis: AudioAnalysis) -> AudioAnalysis:
        # Very conservative fallback: estimate duration, generate pseudo features
        try:
            import wave
            with wave.open(audio_path, 'rb') as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                analysis.duration = frames / float(rate)
                analysis.sampling_rate = rate
        except Exception:
            analysis.duration = min(self.max_duration, 180.0)  # default 3 minutes

        # random tempo heuristics
        analysis.tempo_bpm = float(random.uniform(self._fallback_tempo_min, self._fallback_tempo_max))
        analysis.beat_frames = [i * (60.0 / analysis.tempo_bpm) for i in range(int(analysis.duration / (60.0 / analysis.tempo_bpm)))]
        analysis.tempo_confidence = 0.5
        analysis.dynamic_range = 0.5
        # energy segments random
        num_segments = 8
        analysis.energy_segments = [random.uniform(0.2, 0.9) for _ in range(num_segments)]
        # audio reactive points
        analysis.audio_reactive_points = []
        for i in range(min(12, int(analysis.duration // 10 + 1))):
            t = random.uniform(0, analysis.duration)
            analysis.audio_reactive_points.append({"time": t, "frame": int(t * 24), "intensity": random.uniform(0.4, 1.0)})
        # dummy spectral
        analysis.spectral_features = {"brightness": random.uniform(0.3, 0.7), "warmth": random.uniform(0.3, 0.7)}
        return analysis

    def _analyze_lyrics(self, text: str) -> Tuple[List[str], List[str]]:
        text_lower = (text or "").lower()
        emotions = []
        visuals = []
        emotion_map = {
            "joy": ["happy", "joy", "celebrate", "bright", "smile"],
            "energy": ["power", "strong", "energy", "fire", "intense", "electric"],
            "peace": ["peace", "calm", "quiet", "gentle", "serene"],
            "melancholy": ["sad", "sadness", "cry", "tears", "lonely", "blue"],
            "love": ["love", "heart", "kiss", "romance"],
            "mystery": ["mystery", "secret", "unknown", "dark", "shadow"]
        }
        for k, keywords in emotion_map.items():
            for kw in keywords:
                if kw in text_lower:
                    emotions.append(k)
                    break
        visual_keywords = ["sky", "sun", "moon", "ocean", "mountain", "city", "river", "forest", "rain", "fire", "light", "shadow", "flower"]
        for kw in visual_keywords:
            if kw in text_lower:
                visuals.append(kw)
        return list(dict.fromkeys(emotions))[:5], list(dict.fromkeys(visuals))[:10]

# ------------------------------------------------------------
# ScheduleGenerator (turns keyframes into Deforum schedule strings)
# ------------------------------------------------------------
class ScheduleGenerator:
    @staticmethod
    def ensure_frame0(keyframes: List[Tuple[int, float]]) -> List[Tuple[int, float]]:
        if not keyframes:
            return [(0, 1.0)]
        keyframes = sorted(keyframes, key=lambda kv: kv[0])
        if keyframes[0][0] != 0:
            # insert a copy at frame 0
            keyframes.insert(0, (0, keyframes[0][1]))
        return keyframes

    @staticmethod
    def keyframes_to_schedule_string(keyframes: List[Tuple[int, float]], total_frames: int, fmt: str = "{frame}:({value:.4f})") -> str:
        parts = []
        for frame, value in keyframes:
            if frame < 0:
                continue
            if frame >= total_frames:
                # clamp to last frame-1
                continue
            parts.append(fmt.format(frame=int(frame), value=value))
        return ", ".join(parts) if parts else "0:(0.0)"

    @staticmethod
    def linear_interpolated_keyframes(raw_keyframes: List[Tuple[int, float]], total_frames: int, downsample: int = 1) -> str:
        # Generate an interpolated schedule by simply listing the keyframes
        kf = ScheduleGenerator.ensure_frame0(raw_keyframes)
        return ScheduleGenerator.keyframes_to_schedule_string(kf, total_frames)

# ------------------------------------------------------------
# DeforumMusicGenerator: produces schedules & prompts
# ------------------------------------------------------------
class DeforumMusicGenerator:
    def __init__(self):
        self.schedule_gen = ScheduleGenerator()

    # Top-level builder
    def build_deforum_settings(self, analysis: AudioAnalysis, user_settings: Dict[str, Any]) -> Dict[str, Any]:
        # Merge user settings with defaults
        fps = int(user_settings.get("fps", 24))
        w = int(user_settings.get("width", 1024))
        h = int(user_settings.get("height", 576))
        steps = int(user_settings.get("steps", 25))
        cfg_scale = float(user_settings.get("cfg_scale", 7.0))
        seed = int(user_settings.get("seed", -1))
        sampler = user_settings.get("sampler", "Euler a")
        negative_prompt = user_settings.get("negative_prompt", "low quality, blurry, distorted, watermark, text")
        base_prompt = user_settings.get("base_prompt", "cinematic masterpiece, highly detailed")
        style_prompt = user_settings.get("style_prompt", "film grain, dynamic lighting")

        # compute schedules & prompts
        schedules = self.generate_schedules(analysis, {"fps": fps})
        prompts = self.generate_prompts(analysis, {"base_prompt": base_prompt, "style_prompt": style_prompt})

        total_frames = int(max(1, analysis.duration * fps))

        settings = {
            "W": w,
            "H": h,
            "seed": seed,
            "sampler": sampler,
            "steps": steps,
            "scale": cfg_scale,

            # animation settings
            "animation_mode": "3D",
            "max_frames": total_frames,
            "fps": fps,
            "border": "replicate",

            # schedules (spread into top-level keys)
            **schedules,

            # prompts
            "prompts": prompts,
            "negative_prompts": {"0": negative_prompt},

            # additional Deforum/SD settings
            "color_coherence": "Match Frame 0 LAB",
            "diffusion_cadence": 1,
            "use_depth_warping": True,
            "midas_weight": 0.3,
            "fov": 70,
            "padding_mode": "border",
            "sampling_mode": "bicubic",

            # meta + saving
            "batch_name": user_settings.get("batch_name", f"DefMusic_{now_timestring()}"),
            "filename_format": "{timestring}_{index:05}_{prompt:.120}",
            "seed_behavior": "iter",
            "make_grid": False,
            "save_settings": True,
            "save_samples": True,

            # soundtrack
            "add_soundtrack": "File",
            "soundtrack_path": user_settings.get("soundtrack_path", ""),

            # metadata for debugging
            "_enhanced_metadata": {
                "audio_analysis": analysis.to_dict(),
                "generated_at": time.time(),
                "generator_version": "2.0_complete",
                "audio_reactive_features": list(schedules.keys())
            }
        }
        return settings

    # Produce multiple schedules based on analysis
    def generate_schedules(self, analysis: AudioAnalysis, user_settings: Dict[str, Any]) -> Dict[str, str]:
        fps = int(user_settings.get("fps", 24))
        total_frames = max(1, int(analysis.duration * fps))
        # build schedules for many properties
        schedules = {
            "translation_x": self._generate_movement_schedule(analysis, total_frames, axis="x"),
            "translation_y": self._generate_movement_schedule(analysis, total_frames, axis="y"),
            "translation_z": self._generate_movement_schedule(analysis, total_frames, axis="z"),
            "rotation_3d_x": self._generate_rotation_schedule(analysis, total_frames, axis="x"),
            "rotation_3d_y": self._generate_rotation_schedule(analysis, total_frames, axis="y"),
            "rotation_3d_z": self._generate_rotation_schedule(analysis, total_frames, axis="z"),
            "zoom": self._generate_zoom_schedule(analysis, total_frames),
            "fov_schedule": self._generate_fov_schedule(analysis, total_frames),
            "strength_schedule": self._generate_strength_schedule(analysis, total_frames),
            "cfg_scale_schedule": self._generate_cfg_schedule(analysis, total_frames),
            "noise_schedule": self._generate_noise_schedule(analysis, total_frames),
            "contrast_schedule": self._generate_contrast_schedule(analysis, total_frames),
            "diffusion_cadence_schedule": self._generate_diffusion_cadence_schedule(analysis, total_frames),
            "color_coherence_schedule": self._generate_color_coherence_schedule(analysis, total_frames)
        }
        return schedules

    # Movement schedule example
    def _generate_movement_schedule(self, analysis: AudioAnalysis, total_frames: int, axis: str) -> str:
        keyframes: List[Tuple[int, float]] = [(0, 0.0)]
        # base movement from energy segments
        nseg = len(analysis.energy_segments) or 8
        for i, energy in enumerate(analysis.energy_segments):
            frame = int(total_frames * i / max(1, nseg - 1)) if nseg > 1 else 0
            # map energy to movement magnitude differently per axis
            if axis == "x":
                val = (energy - 0.5) * 30.0  # -15 .. +15 px-ish
            elif axis == "y":
                val = (energy - 0.5) * 18.0
            else:  # z -> deep movement = scale (negative moves camera backward)
                val = -energy * 50.0
            keyframes.append((frame, float(val)))

        # add audio-reactive bursts at detected peaks
        for p in analysis.audio_reactive_points[:40]:
            f = int(p["frame"])
            if f < total_frames:
                intensity = float(p["intensity"])
                jitter = (random.random() - 0.5) * intensity * 4.0
                if axis == "x":
                    val = jitter * 10.0
                elif axis == "y":
                    val = jitter * 6.0
                else:
                    val = -intensity * 12.0
                keyframes.append((f, float(val)))
        # convert to schedule string
        keyframes = sorted({k: v for k, v in keyframes}.items(), key=lambda x: x[0])
        return self.schedule_gen.keyframes_to_schedule_string(keyframes, total_frames)

    # Rotation schedule example
    def _generate_rotation_schedule(self, analysis: AudioAnalysis, total_frames: int, axis: str) -> str:
        kf: List[Tuple[int, float]] = [(0, 0.0)]
        tempo_factor = max(0.5, analysis.tempo_bpm / 120.0)
        for i, t in enumerate(analysis.beat_frames[:200]):
            frame = int(t * 24)
            if frame >= total_frames:
                continue
            if axis == "x":
                val = math.sin(t * 0.2 + i) * 5.0 * tempo_factor
            elif axis == "y":
                val = math.cos(t * 0.15 + i) * 6.0 * tempo_factor
            else:
                val = math.sin(t * 0.05 + i) * 3.0 * tempo_factor
            kf.append((frame, float(val)))
        return self.schedule_gen.keyframes_to_schedule_string(kf, total_frames)

    # zoom schedule
    def _generate_zoom_schedule(self, analysis: AudioAnalysis, total_frames: int) -> str:
        kf = [(0, 1.0)]
        nseg = len(analysis.energy_segments) or 8
        for i, energy in enumerate(analysis.energy_segments):
            frame = int(total_frames * i / max(1, nseg - 1)) if nseg > 1 else 0
            zoom = 1.0 + (energy - 0.4) * 0.25  # gentle zoom variations
            kf.append((frame, float(zoom)))
        # bursts on reactive points
        for p in analysis.audio_reactive_points[:30]:
            f = int(p["frame"])
            if f < total_frames and p["intensity"] > 0.6:
                kf.append((f, 1.0 + p["intensity"] * 0.25))
        return self.schedule_gen.keyframes_to_schedule_string(kf, total_frames)

    # field-of-view schedule
    def _generate_fov_schedule(self, analysis: AudioAnalysis, total_frames: int) -> str:
        kf = [(0, 70.0)]
        for i, energy in enumerate(analysis.energy_segments):
            frame = int(total_frames * i / max(1, len(analysis.energy_segments) - 1))
            fov = 70.0 + (energy - 0.5) * 25.0
            kf.append((frame, float(fov)))
        return self.schedule_gen.keyframes_to_schedule_string(kf, total_frames)

    # strength schedule (for "strength" parameter used by some samplers / latents)
    def _generate_strength_schedule(self, analysis: AudioAnalysis, total_frames: int) -> str:
        base = 0.7
        kf = [(0, base)]
        for i, energy in enumerate(analysis.energy_segments):
            frame = int(total_frames * i / max(1, len(analysis.energy_segments) - 1))
            val = base + (energy - 0.5) * 0.25
            kf.append((frame, float(val)))
        return self.schedule_gen.keyframes_to_schedule_string(kf, total_frames)

    # CFG schedule (control scale)
    def _generate_cfg_schedule(self, analysis: AudioAnalysis, total_frames: int) -> str:
        base = 7.0
        kf = [(0, base)]
        for i, energy in enumerate(analysis.energy_segments):
            frame = int(total_frames * i / max(1, len(analysis.energy_segments) - 1))
            val = base + (energy - 0.5) * 3.0
            kf.append((frame, float(val)))
        return self.schedule_gen.keyframes_to_schedule_string(kf, total_frames)

    # noise schedule (guiding noise)
    def _generate_noise_schedule(self, analysis: AudioAnalysis, total_frames: int) -> str:
        base = 0.04
        kf = [(0, base)]
        for i, energy in enumerate(analysis.energy_segments):
            frame = int(total_frames * i / max(1, len(analysis.energy_segments) - 1))
            val = base + energy * 0.02
            kf.append((frame, float(val)))
        return self.schedule_gen.keyframes_to_schedule_string(kf, total_frames)

    # contrast schedule
    def _generate_contrast_schedule(self, analysis: AudioAnalysis, total_frames: int) -> str:
        base = 1.0
        kf = [(0, base)]
        for i, energy in enumerate(analysis.energy_segments):
            frame = int(total_frames * i / max(1, len(analysis.energy_segments) - 1))
            val = base + (energy - 0.5) * 0.25
            kf.append((frame, float(val)))
        return self.schedule_gen.keyframes_to_schedule_string(kf, total_frames)

    def _generate_diffusion_cadence_schedule(self, analysis: AudioAnalysis, total_frames: int) -> str:
        # cadence 1 or 2 depending on beats
        kf = [(0, 1)]
        for i, t in enumerate(analysis.beat_frames[:200]):
            frame = int(t * 24)
            if frame >= total_frames:
                continue
            cadence = 2 if (i % 8 == 0) else 1
            kf.append((frame, int(cadence)))
        parts = []
        for frame, val in kf:
            parts.append(f"{frame}:({int(val)})")
        return ", ".join(parts)

    def _generate_color_coherence_schedule(self, analysis: AudioAnalysis, total_frames: int) -> str:
        kf = [(0, 0.3)]
        for i, energy in enumerate(analysis.energy_segments):
            frame = int(total_frames * i / max(1, len(analysis.energy_segments) - 1))
            coherence = max(0.05, 0.5 - energy * 0.3)
            kf.append((frame, float(coherence)))
        return self.schedule_gen.keyframes_to_schedule_string(kf, total_frames)

    # prompt generation
    def generate_prompts(self, analysis: AudioAnalysis, user_settings: Dict[str, Any]) -> Dict[str, str]:
        base_prompt = user_settings.get("base_prompt", "cinematic masterpiece, highly detailed")
        style_prompt = user_settings.get("style_prompt", "film grain, dynamic lighting")

        prompts: Dict[str, str] = {}
        nseg = max(1, len(analysis.energy_segments))
        for i in range(nseg):
            frame_idx = int(analysis.duration * 24 * i / max(1, nseg - 1)) if nseg > 1 else 0
            energy = analysis.energy_segments[i] if i < len(analysis.energy_segments) else 0.5
            parts = [base_prompt]

            # energy-driven modifiers
            if energy >= 0.8:
                parts.append("epic, cinematic, high energy, dynamic movement, dramatic lighting")
            elif energy >= 0.6:
                parts.append("bold composition, vibrant motion, cinematic lighting")
            elif energy >= 0.4:
                parts.append("gentle movement, soft lighting, cinematic composition")
            else:
                parts.append("calm, tranquil, dreamlike, atmospheric")

            # spectral based
            brightness = analysis.spectral_features.get("brightness", 0.5)
            warmth = analysis.spectral_features.get("warmth", 0.5)
            if brightness > 0.6:
                parts.append("bright lighting, high key, vivid colors")
            elif brightness < 0.35:
                parts.append("low key, moody, cinematic shadows")
            if warmth > 0.6:
                parts.append("warm tones, golden hour ambience")
            elif warmth < 0.35:
                parts.append("cool tones, ethereal blues")

            # lyric emotions
            for emotion in analysis.lyric_emotions[:3]:
                if emotion == "joy":
                    parts.append("uplifting, celebratory, radiant")
                elif emotion == "energy":
                    parts.append("electric, intense, visceral")
                elif emotion == "peace":
                    parts.append("serene, meditative, tranquil")
                elif emotion == "melancholy":
                    parts.append("wistful, melancholic, cinematic sadness")
                elif emotion == "love":
                    parts.append("romantic, tender, intimate")
                elif emotion == "mystery":
                    parts.append("mysterious, enigmatic, nocturnal")

            # visual keywords from lyrics
            if analysis.visual_elements:
                parts.append("featuring " + ", ".join(analysis.visual_elements[:3]))

            # style
            parts.append(style_prompt)

            prompts[str(frame_idx)] = ", ".join(parts)

        # ensure a prompt at frame 0
        if "0" not in prompts:
            prompts["0"] = f"{base_prompt}, {style_prompt}"

        return prompts

# ------------------------------------------------------------
# Gradio UI: create_gradio_interface (accepts analyzer/generator optionally)
# ------------------------------------------------------------

def create_gradio_interface(analyzer: Optional[AudioAnalyzer] = None, generator: Optional[DeforumMusicGenerator] = None):
    """Create the Gradio UI.

    Default experience: **Deforum JSON Expert mode** (full template editable).
    """
    if not GRADIO_AVAILABLE:
        raise ImportError("Gradio not available. Install with: pip install gradio")

    # create defaults if not provided
    analyzer = analyzer if analyzer is not None else AudioAnalyzer(max_duration=MAX_AUDIO_DURATION)
    generator = generator if generator is not None else DeforumMusicGenerator()

    # Import here to keep the module usable both as a package and as a standalone copied script.
    try:
        from .deforum_defaults import deep_merge_dicts, make_deforum_settings_template
    except Exception:  # pragma: no cover
        from enhanced_deforum_music_generator.deforum_defaults import deep_merge_dicts, make_deforum_settings_template  # type: ignore

    import gradio as gr  # local reference

    default_template_dict = make_deforum_settings_template()
    default_template_json = json.dumps(default_template_dict, indent=2, ensure_ascii=False)

    css = r"""
    .edmg-header { text-align: center; margin-bottom: 12px; }
    .edmg-header h1 { margin: 0; }
    .edmg-sub { opacity: 0.85; }
    .edmg-small { opacity: 0.7; font-size: 0.9em; }
    .edmg-box { border: 1px solid rgba(127,127,127,0.25); border-radius: 10px; padding: 12px; }
    """

    with gr.Blocks(
        title="Enhanced Deforum Music Generator",
        theme=getattr(getattr(gr, "themes", None), "Soft", None)() if hasattr(gr, "themes") else None,
        css=css,
    ) as interface:
        gr.HTML(
            """
            <div class="edmg-header">
              <h1>🎵 Enhanced Deforum Music Generator</h1>
              <div class="edmg-sub">Audio-reactive Deforum schedules + prompts + full Deforum JSON template</div>
              <div class="edmg-small">Default mode: <b>Deforum JSON Expert</b> (edit the full template, then generate)</div>
            </div>
            """
        )

        # ---------- Helpers ----------
        def _validate_json(s: str) -> Tuple[str, str]:
            try:
                obj = json.loads(s) if s else {}
                if not isinstance(obj, dict):
                    return "❌ Template JSON must be a JSON object (dictionary).", s
                pretty = json.dumps(obj, indent=2, ensure_ascii=False)
                return "✅ Template JSON looks valid.", pretty
            except Exception as e:
                return f"❌ Invalid JSON: {e}", s

        def _reset_template() -> str:
            return default_template_json

        def _apply_quick_overrides_to_template(
            template_json: str,
            width_val: int,
            height_val: int,
            fps_val: int,
            steps_val: int,
            cfg_val: float,
            seed_val: int,
            sampler_val: str,
        ) -> str:
            try:
                obj = json.loads(template_json) if template_json else {}
                if not isinstance(obj, dict):
                    obj = {}
            except Exception:
                obj = {}

            obj["W"] = int(width_val)
            obj["H"] = int(height_val)
            obj["fps"] = int(fps_val)
            obj["steps"] = int(steps_val)
            obj["scale"] = float(cfg_val)
            obj["seed"] = int(seed_val)
            obj["sampler"] = str(sampler_val)

            return json.dumps(obj, indent=2, ensure_ascii=False)

        def _analyze(audio_path: str, enable_lyrics_flag: bool) -> Tuple[Dict[str, Any], str]:
            if not audio_path:
                return {}, "No audio file provided."
            try:
                analysis = analyzer.analyze(audio_path, enable_lyrics=bool(enable_lyrics_flag))
                summary = (
                    f"**Duration:** {analysis.duration:.1f}s  \n"
                    f"**Tempo:** {analysis.tempo_bpm:.1f} BPM  \n"
                    f"**Beats:** {len(getattr(analysis, 'beat_frames', []) or [])}  \n"
                    f"**Lyrics:** {'enabled' if bool(enable_lyrics_flag) else 'disabled'}"
                )
                return analysis.to_dict(), summary
            except Exception as e:
                tb = traceback.format_exc()
                return {}, f"Analysis failed: {e}\n\n```text\n{tb}\n```"

        def _generate(
            audio_path: str,
            enable_lyrics_flag: bool,
            base_prompt_text: str,
            style_prompt_text: str,
            neg_prompt_text: str,
            width_val: int,
            height_val: int,
            fps_val: int,
            steps_val: int,
            cfg_val: float,
            seed_val: int,
            sampler_val: str,
            batch_name_val: str,
            template_json: str,
        ) -> Tuple[str, str, str]:
            log_lines: List[str] = []
            if not audio_path:
                return "", "", "No audio file provided."

            # validate template JSON first
            try:
                template_obj = json.loads(template_json) if template_json else {}
                if not isinstance(template_obj, dict):
                    return "", "", "Template JSON must be a JSON object."
            except Exception as e:
                return "", "", f"Invalid template JSON: {e}"

            try:
                log_lines.append("Analyzing audio...")
                analysis = analyzer.analyze(audio_path, enable_lyrics=bool(enable_lyrics_flag))
                log_lines.append(f"Audio OK: duration {analysis.duration:.1f}s | tempo {analysis.tempo_bpm:.1f} BPM")

                seed_final = int(seed_val) if int(seed_val) != -1 else random.randint(1, 2**31 - 1)

                user_settings = {
                    "base_prompt": base_prompt_text,
                    "style_prompt": style_prompt_text,
                    "negative_prompt": neg_prompt_text,
                    "width": int(width_val),
                    "height": int(height_val),
                    "fps": int(fps_val),
                    "steps": int(steps_val),
                    "cfg_scale": float(cfg_val),
                    "seed": seed_final,
                    "sampler": sampler_val,
                    "batch_name": batch_name_val,
                    "soundtrack_path": audio_path,
                }

                log_lines.append("Generating Deforum schedules & prompts...")
                generated = generator.build_deforum_settings(analysis, user_settings)

                # Expert mode rule: user template overrides the generated output
                final_settings = deep_merge_dicts(generated, template_obj)

                final_json = json.dumps(final_settings, indent=2, ensure_ascii=False)

                log_lines.append("Creating package files...")
                temp_dir = tempfile.mkdtemp(prefix="deforum_music_")
                settings_path = os.path.join(temp_dir, "deforum_settings.json")
                dump_json_file(settings_path, final_settings)

                analysis_report_path = os.path.join(temp_dir, "analysis_report.json")
                dump_json_file(analysis_report_path, analysis.to_dict())

                readme_path = os.path.join(temp_dir, "README.md")
                readme_contents = (
                    "# Deforum Music Generation Package\n\n"
                    "This package was generated by Enhanced Deforum Music Generator (EDMG).\n\n"
                    "## Files\n"
                    "- `deforum_settings.json` : full Deforum settings (all keys)\n"
                    "- `analysis_report.json`  : extracted audio/lyrics analysis\n\n"
                    "## Use\n"
                    "Import `deforum_settings.json` into Deforum (A1111 Deforum extension).\n"
                )
                with open(readme_path, "w", encoding="utf-8") as f:
                    f.write(readme_contents)

                package_name = f"{batch_name_val}_{now_timestring()}.zip"
                package_path = os.path.join(temp_dir, package_name)
                with zipfile.ZipFile(package_path, "w", zipfile.ZIP_DEFLATED) as zf:
                    zf.write(settings_path, arcname="deforum_settings.json")
                    zf.write(analysis_report_path, arcname="analysis_report.json")
                    zf.write(readme_path, arcname="README.md")

                log_lines.append(f"Package ready: {package_path}")

                return final_json, package_path, "\n".join(log_lines)
            except Exception as e:
                tb = traceback.format_exc()
                log_lines.append(f"Generation failed: {e}")
                log_lines.append(tb)
                return "", "", "\n".join(log_lines)

        # ---------- UI ----------
        with gr.Tabs():
            # Default tab = first tab => expert mode
            with gr.Tab("Deforum JSON Expert"):
                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Group():
                            gr.Markdown("### 🎵 Audio Input")
                            audio_input = gr.Audio(label="Music File", type="filepath")
                            enable_lyrics = gr.Checkbox(label="Enable Lyrics Analysis (Whisper)", value=False)

                        with gr.Group():
                            gr.Markdown("### ✍️ Prompts")
                            base_prompt = gr.Textbox(label="Base Prompt", value="cinematic masterpiece, highly detailed")
                            style_prompt = gr.Textbox(label="Style Prompt", value="film grain, dynamic lighting")
                            negative_prompt = gr.Textbox(
                                label="Negative Prompt",
                                value="low quality, blurry, distorted, watermark, text",
                            )

                        with gr.Group():
                            gr.Markdown("### ⚙️ Quick Overrides (optional)")
                            with gr.Row():
                                width = gr.Number(label="Width (W)", value=1024, precision=0)
                                height = gr.Number(label="Height (H)", value=576, precision=0)
                            with gr.Row():
                                fps = gr.Number(label="FPS", value=24, precision=0)
                                steps = gr.Number(label="Steps", value=25, precision=0)
                            with gr.Row():
                                cfg_scale = gr.Number(label="CFG Scale", value=7.0)
                                seed = gr.Number(label="Seed (-1 = random)", value=-1, precision=0)
                            sampler = gr.Dropdown(choices=DEFAULT_SAMPLERS, value=DEFAULT_SAMPLERS[0], label="Sampler")
                            batch_name = gr.Textbox(label="Batch Name", value=f"DefMusic_{now_timestring()}")

                        with gr.Row():
                            analyze_btn = gr.Button("Analyze Audio", variant="secondary")
                            generate_btn = gr.Button("Generate Package (Expert)", variant="primary")

                        generation_log = gr.Textbox(label="Log", lines=12, interactive=False)

                    with gr.Column(scale=1):
                        with gr.Group():
                            gr.Markdown("### 🧠 Full Deforum Template (Editable JSON)")
                            template_editor = gr.Code(
                                label="Deforum Settings Template (JSON)",
                                language="json",
                                value=default_template_json,
                                lines=26,
                                interactive=True,
                            )
                            with gr.Row():
                                reset_template_btn = gr.Button("Reset Template")
                                validate_template_btn = gr.Button("Validate + Pretty Print")
                                apply_quick_btn = gr.Button("Apply Quick Overrides → Template")

                            template_status = gr.Markdown(value="")

                        with gr.Group():
                            gr.Markdown("### ✅ Final Output")
                            final_settings_out = gr.Code(
                                label="Final Deforum Settings (JSON)",
                                language="json",
                                lines=26,
                                interactive=False,
                            )
                            package_file = gr.File(label="Download Package")

                analysis_json = gr.JSON(label="Analysis Results", visible=False)
                analysis_summary = gr.Markdown(value="")

                # events
                analyze_btn.click(
                    fn=_analyze,
                    inputs=[audio_input, enable_lyrics],
                    outputs=[analysis_json, analysis_summary],
                )
                validate_template_btn.click(
                    fn=_validate_json,
                    inputs=[template_editor],
                    outputs=[template_status, template_editor],
                )
                reset_template_btn.click(
                    fn=_reset_template,
                    inputs=[],
                    outputs=[template_editor],
                )
                apply_quick_btn.click(
                    fn=_apply_quick_overrides_to_template,
                    inputs=[template_editor, width, height, fps, steps, cfg_scale, seed, sampler],
                    outputs=[template_editor],
                )
                generate_btn.click(
                    fn=_generate,
                    inputs=[
                        audio_input,
                        enable_lyrics,
                        base_prompt,
                        style_prompt,
                        negative_prompt,
                        width,
                        height,
                        fps,
                        steps,
                        cfg_scale,
                        seed,
                        sampler,
                        batch_name,
                        template_editor,
                    ],
                    outputs=[final_settings_out, package_file, generation_log],
                )

            with gr.Tab("Audio Analysis"):
                gr.Markdown("### Audio / Lyrics Analysis")
                aa_audio = gr.Audio(label="Music File", type="filepath")
                aa_enable_lyrics = gr.Checkbox(label="Enable Lyrics Analysis (Whisper)", value=False)
                aa_btn = gr.Button("Analyze", variant="primary")
                aa_json = gr.JSON(label="Analysis Results")
                aa_summary = gr.Markdown(value="")
                aa_btn.click(fn=_analyze, inputs=[aa_audio, aa_enable_lyrics], outputs=[aa_json, aa_summary])

            with gr.Tab("Guided Generate"):
                gr.Markdown(
                    "### Guided generation\n"
                    "This uses the internal full Deforum template and returns a complete settings JSON.\n"
                    "If you want full control, use the **Deforum JSON Expert** tab."
                )
                gg_audio = gr.Audio(label="Music File", type="filepath")
                gg_enable_lyrics = gr.Checkbox(label="Enable Lyrics Analysis (Whisper)", value=False)
                gg_base = gr.Textbox(label="Base Prompt", value="cinematic masterpiece, highly detailed")
                gg_style = gr.Textbox(label="Style Prompt", value="film grain, dynamic lighting")
                gg_neg = gr.Textbox(label="Negative Prompt", value="low quality, blurry, distorted, watermark, text")
                with gr.Row():
                    gg_w = gr.Number(label="Width (W)", value=1024, precision=0)
                    gg_h = gr.Number(label="Height (H)", value=576, precision=0)
                with gr.Row():
                    gg_fps = gr.Number(label="FPS", value=24, precision=0)
                    gg_steps = gr.Number(label="Steps", value=25, precision=0)
                with gr.Row():
                    gg_cfg = gr.Number(label="CFG Scale", value=7.0)
                    gg_seed = gr.Number(label="Seed (-1 = random)", value=-1, precision=0)
                gg_sampler = gr.Dropdown(choices=DEFAULT_SAMPLERS, value=DEFAULT_SAMPLERS[0], label="Sampler")
                gg_batch = gr.Textbox(label="Batch Name", value=f"DefMusic_{now_timestring()}")
                gg_btn = gr.Button("Generate Package (Guided)", variant="primary")
                gg_log = gr.Textbox(label="Log", lines=12, interactive=False)
                gg_out = gr.Code(label="Final Deforum Settings (JSON)", language="json", lines=22, interactive=False)
                gg_file = gr.File(label="Download Package")

                # guided uses default template (expert JSON not required)
                def _generate_guided(*args):
                    # args align with _generate but uses default template json
                    return _generate(*args, default_template_json)

                gg_btn.click(
                    fn=_generate_guided,
                    inputs=[
                        gg_audio,
                        gg_enable_lyrics,
                        gg_base,
                        gg_style,
                        gg_neg,
                        gg_w,
                        gg_h,
                        gg_fps,
                        gg_steps,
                        gg_cfg,
                        gg_seed,
                        gg_sampler,
                        gg_batch,
                    ],
                    outputs=[gg_out, gg_file, gg_log],
                )

            with gr.Tab("Info"):
                deps = [
                    ("NumPy", NUMPY_AVAILABLE),
                    ("Librosa", LIBROSA_AVAILABLE),
                    ("Whisper", WHISPER_AVAILABLE),
                    ("SciPy (peak detection)", SCIPY_AVAILABLE),
                    ("Automatic1111 integration", A1111_AVAILABLE and SCRIPT_CALLBACKS_AVAILABLE),
                ]
                deps_md_lines = ["### System Status", "**Dependencies:**"]
                for name, ok in deps:
                    deps_md_lines.append(f"- {name}: {'Available' if ok else 'Missing'}")
                gr.Markdown("\n".join(deps_md_lines))
                gr.Markdown(
                    "### Quick start\n"
                    "- Run: `python -m enhanced_deforum_music_generator ui --port 7860`\n"
                    "- Install: use `installer_gui.py` or `install.ps1` / `install.sh`\n"
                    "- A1111: install Deforum + the EDMG extension via the installer GUI.\n"
                )

        return interface
# ------------------------------------------------------------
# Standalone package creation function (CLI-friendly)
# ------------------------------------------------------------
def create_package(audio_file: str, output_dir: Optional[str] = None, **kwargs) -> str:
    if not output_dir:
        output_dir = tempfile.mkdtemp(prefix="deforum_music_")
    safe_makedirs(output_dir)

    analyzer = AudioAnalyzer(max_duration=MAX_AUDIO_DURATION)
    generator = DeforumMusicGenerator()

    analysis = analyzer.analyze(audio_file, kwargs.get("enable_lyrics", False))
    user_settings = {
        "base_prompt": kwargs.get("base_prompt", "cinematic masterpiece, highly detailed"),
        "style_prompt": kwargs.get("style_prompt", "film grain, dynamic lighting"),
        "negative_prompt": kwargs.get("negative_prompt", "low quality, blurry"),
        "width": kwargs.get("width", 1024),
        "height": kwargs.get("height", 576),
        "fps": kwargs.get("fps", 24),
        "steps": kwargs.get("steps", 25),
        "cfg_scale": kwargs.get("cfg_scale", 7.0),
        "seed": kwargs.get("seed", -1),
        "sampler": kwargs.get("sampler", "Euler a"),
        "batch_name": kwargs.get("batch_name", f"DefMusic_{now_timestring()}"),
        "soundtrack_path": audio_file
    }
    settings = generator.build_deforum_settings(analysis, user_settings)

    settings_path = os.path.join(output_dir, "deforum_settings.json")
    dump_json_file(settings_path, settings)

    analysis_path = os.path.join(output_dir, "analysis_report.json")
    dump_json_file(analysis_path, analysis.to_dict())

    # write README
    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(f"Generated on {time.asctime()}\nSoundtrack: {audio_file}\nFrames: {settings.get('max_frames')}\n")

    # create zip
    package_zip = os.path.join(output_dir, f"{user_settings['batch_name']}.zip")
    with zipfile.ZipFile(package_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(settings_path, "deforum_settings.json")
        zf.write(analysis_path, "analysis_report.json")
        zf.write(readme_path, "README.md")

    return package_zip

# ------------------------------------------------------------
# CLI entrypoint
# ------------------------------------------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Enhanced Deforum Music Generator")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    ui_parser = subparsers.add_parser("ui", help="Launch local Gradio UI")
    ui_parser.add_argument("--port", type=int, default=7860)
    ui_parser.add_argument("--share", action="store_true", default=False)

    gen_parser = subparsers.add_parser("generate", help="Generate package from audio (CLI)")
    gen_parser.add_argument("audio_file", help="Path to audio file")
    gen_parser.add_argument("--output", "-o", help="Output directory")
    gen_parser.add_argument("--base-prompt", default="cinematic masterpiece, highly detailed")
    gen_parser.add_argument("--style-prompt", default="film grain, dynamic lighting")
    gen_parser.add_argument("--width", type=int, default=1024)
    gen_parser.add_argument("--height", type=int, default=576)
    gen_parser.add_argument("--fps", type=int, default=24)
    gen_parser.add_argument("--steps", type=int, default=25)
    gen_parser.add_argument("--cfg-scale", type=float, default=7.0)
    gen_parser.add_argument("--seed", type=int, default=-1)
    gen_parser.add_argument("--sampler", default=DEFAULT_SAMPLERS[0])
    gen_parser.add_argument("--lyrics", action="store_true", default=False)

    analyze_parser = subparsers.add_parser("analyze", help="Analyze audio only")
    analyze_parser.add_argument("audio_file", help="Path to audio file")
    analyze_parser.add_argument("--lyrics", action="store_true", default=False)

    args = parser.parse_args()

    if args.command == "ui":
        if not GRADIO_AVAILABLE:
            print("Gradio not installed. pip install gradio")
            return 1
        interface = create_gradio_interface()
        interface.launch(server_port=args.port, share=args.share, inbrowser=True)
        return 0

    elif args.command == "generate":
        try:
            out = create_package(
                args.audio_file,
                args.output,
                base_prompt=args.base_prompt,
                style_prompt=args.style_prompt,
                width=args.width,
                height=args.height,
                fps=args.fps,
                steps=args.steps,
                cfg_scale=args.cfg_scale,
                seed=args.seed,
                sampler=args.sampler,
                enable_lyrics=args.lyrics
            )
            print("Package created at:", out)
            return 0
        except Exception as e:
            traceback.print_exc()
            print("Error generating package:", e)
            return 1

    elif args.command == "analyze":
        try:
            analyzer = AudioAnalyzer()
            analysis = analyzer.analyze(args.audio_file, enable_lyrics=args.lyrics)
            print(json.dumps(analysis.to_dict(), indent=2))
            return 0
        except Exception as e:
            traceback.print_exc()
            print("Error analyzing:", e)
            return 1

    else:
        parser.print_help()
        return 1

# ------------------------------------------------------------
# Automatic1111 integration
# ------------------------------------------------------------
# We register a script tab via script_callbacks.on_ui_tabs, which is the preferred method
def register_a1111_tab():
    # if A1111 isn't present, skip
    if not (A1111_AVAILABLE and SCRIPT_CALLBACKS_AVAILABLE and GRADIO_AVAILABLE):
        return

    # define a callback that returns (gr.Blocks, title, id)
    def on_ui_tabs():
        try:
            analyzer = AudioAnalyzer(max_duration=MAX_AUDIO_DURATION)
            generator = DeforumMusicGenerator()
            interface = create_gradio_interface(analyzer=analyzer, generator=generator)
            # return (interface, "Tab Title", "internal_id")
            return (interface, "Enhanced Deforum Music Generator", "deforum_music_generator_tab")
        except Exception as e:
            print("[register_a1111_tab] Failed to create UI tab:", e)
            traceback.print_exc()
            return None

    # register it
    try:
        script_callbacks.on_ui_tabs(on_ui_tabs)
    except Exception as e:
        print("[register_a1111_tab] script_callbacks.on_ui_tabs failed:", e)
        traceback.print_exc()

# run registration now (if in A1111)
if A1111_AVAILABLE and SCRIPT_CALLBACKS_AVAILABLE:
    register_a1111_tab()

# ------------------------------------------------------------
# Entrypoint guard: when run as a script
# ------------------------------------------------------------
if __name__ == "__main__":
    if DEVELOPMENT_SEED is not None:
        random.seed(DEVELOPMENT_SEED)
        if NUMPY_AVAILABLE:
            np.random.seed(DEVELOPMENT_SEED)
    sys.exit(main())    # Top-level builder
    def build_deforum_settings(self, analysis: AudioAnalysis, user_settings: Dict[str, Any]) -> Dict[str, Any]:
        """Build a *full* Deforum settings dict (all known keys), then overlay EDMG schedules/prompts."""
        # Import here to keep the file runnable in multiple contexts (package vs standalone copy)
        try:
            from .deforum_defaults import deep_merge_dicts, make_deforum_settings_template
        except Exception:  # pragma: no cover
            from enhanced_deforum_music_generator.deforum_defaults import deep_merge_dicts, make_deforum_settings_template  # type: ignore

        # Merge user settings with defaults (support common synonyms)
        fps = int(user_settings.get("fps", 24))
        w = int(user_settings.get("W", user_settings.get("width", 1024)))
        h = int(user_settings.get("H", user_settings.get("height", 576)))
        steps = int(user_settings.get("steps", 25))
        cfg_scale = float(user_settings.get("scale", user_settings.get("cfg_scale", 7.0)))
        seed = int(user_settings.get("seed", -1))
        sampler = user_settings.get("sampler", "Euler a")
        negative_prompt = user_settings.get("negative_prompt", "low quality, blurry, distorted, watermark, text")
        base_prompt = user_settings.get("base_prompt", "cinematic masterpiece, highly detailed")
        style_prompt = user_settings.get("style_prompt", "film grain, dynamic lighting")

        # compute schedules & prompts
        schedules = self.generate_schedules(analysis, {"fps": fps})
        prompts = self.generate_prompts(analysis, {"base_prompt": base_prompt, "style_prompt": style_prompt})

        total_frames = int(max(1, analysis.duration * fps))

        # Start from a full Deforum template (101+ keys), then overlay EDMG output.
        template = make_deforum_settings_template(
            {
                "W": w,
                "H": h,
                "fps": fps,
                "steps": steps,
                "cfg_scale": cfg_scale,
                "seed": seed,
                "sampler": sampler,
                "max_frames": total_frames,
            }
        )

        overlay = {
            # schedules (spread into top-level keys)
            **schedules,

            # prompts
            "prompts": prompts,
            "negative_prompts": {"0": str(negative_prompt)},

            # core animation
            "animation_mode": user_settings.get("animation_mode", "3D"),
            "border": user_settings.get("border", "replicate"),

            # soundtrack (Deforum extension convention)
            "add_soundtrack": user_settings.get("add_soundtrack", "File"),
            "soundtrack_path": user_settings.get("soundtrack_path", ""),

            # metadata for debugging
            "_enhanced_metadata": {
                "audio_analysis": analysis.to_dict(),
                "generated_at": time.time(),
                "generator_version": "2.0_complete",
                "audio_reactive_features": list(schedules.keys()),
            },
        }

        return deep_merge_dicts(template, overlay)


    # Produce multiple schedules based on analysis
    def generate_schedules(self, analysis: AudioAnalysis, user_settings: Dict[str, Any]) -> Dict[str, str]:
        fps = int(user_settings.get("fps", 24))
        total_frames = max(1, int(analysis.duration * fps))
        # build schedules for many properties
        schedules = {
            "translation_x": self._generate_movement_schedule(analysis, total_frames, axis="x"),
            "translation_y": self._generate_movement_schedule(analysis, total_frames, axis="y"),
            "translation_z": self._generate_movement_schedule(analysis, total_frames, axis="z"),
            "rotation_3d_x": self._generate_rotation_schedule(analysis, total_frames, axis="x"),
            "rotation_3d_y": self._generate_rotation_schedule(analysis, total_frames, axis="y"),
            "rotation_3d_z": self._generate_rotation_schedule(analysis, total_frames, axis="z"),
            "zoom": self._generate_zoom_schedule(analysis, total_frames),
            "fov_schedule": self._generate_fov_schedule(analysis, total_frames),
            "strength_schedule": self._generate_strength_schedule(analysis, total_frames),
            "cfg_scale_schedule": self._generate_cfg_schedule(analysis, total_frames),
            "noise_schedule": self._generate_noise_schedule(analysis, total_frames),
            "contrast_schedule": self._generate_contrast_schedule(analysis, total_frames),
            "diffusion_cadence_schedule": self._generate_diffusion_cadence_schedule(analysis, total_frames),
            "color_coherence_schedule": self._generate_color_coherence_schedule(analysis, total_frames)
        }
        return schedules

    # Movement schedule example
    def _generate_movement_schedule(self, analysis: AudioAnalysis, total_frames: int, axis: str) -> str:
        keyframes: List[Tuple[int, float]] = [(0, 0.0)]
        # base movement from energy segments
        nseg = len(analysis.energy_segments) or 8
        for i, energy in enumerate(analysis.energy_segments):
            frame = int(total_frames * i / max(1, nseg - 1)) if nseg > 1 else 0
            # map energy to movement magnitude differently per axis
            if axis == "x":
                val = (energy - 0.5) * 30.0  # -15 .. +15 px-ish
            elif axis == "y":
                val = (energy - 0.5) * 18.0
            else:  # z -> deep movement = scale (negative moves camera backward)
                val = -energy * 50.0
            keyframes.append((frame, float(val)))

        # add audio-reactive bursts at detected peaks
        for p in analysis.audio_reactive_points[:40]:
            f = int(p["frame"])
            if f < total_frames:
                intensity = float(p["intensity"])
                jitter = (random.random() - 0.5) * intensity * 4.0
                if axis == "x":
                    val = jitter * 10.0
                elif axis == "y":
                    val = jitter * 6.0
                else:
                    val = -intensity * 12.0
                keyframes.append((f, float(val)))
        # convert to schedule string
        keyframes = sorted({k: v for k, v in keyframes}.items(), key=lambda x: x[0])
        return self.schedule_gen.keyframes_to_schedule_string(keyframes, total_frames)

    # Rotation schedule example
    def _generate_rotation_schedule(self, analysis: AudioAnalysis, total_frames: int, axis: str) -> str:
        kf: List[Tuple[int, float]] = [(0, 0.0)]
        tempo_factor = max(0.5, analysis.tempo_bpm / 120.0)
        for i, t in enumerate(analysis.beat_frames[:200]):
            frame = int(t * 24)
            if frame >= total_frames:
                continue
            if axis == "x":
                val = math.sin(t * 0.2 + i) * 5.0 * tempo_factor
            elif axis == "y":
                val = math.cos(t * 0.15 + i) * 6.0 * tempo_factor
            else:
                val = math.sin(t * 0.05 + i) * 3.0 * tempo_factor
            kf.append((frame, float(val)))
        return self.schedule_gen.keyframes_to_schedule_string(kf, total_frames)

    # zoom schedule
    def _generate_zoom_schedule(self, analysis: AudioAnalysis, total_frames: int) -> str:
        kf = [(0, 1.0)]
        nseg = len(analysis.energy_segments) or 8
        for i, energy in enumerate(analysis.energy_segments):
            frame = int(total_frames * i / max(1, nseg - 1)) if nseg > 1 else 0
            zoom = 1.0 + (energy - 0.4) * 0.25  # gentle zoom variations
            kf.append((frame, float(zoom)))
        # bursts on reactive points
        for p in analysis.audio_reactive_points[:30]:
            f = int(p["frame"])
            if f < total_frames and p["intensity"] > 0.6:
                kf.append((f, 1.0 + p["intensity"] * 0.25))
        return self.schedule_gen.keyframes_to_schedule_string(kf, total_frames)

    # field-of-view schedule
    def _generate_fov_schedule(self, analysis: AudioAnalysis, total_frames: int) -> str:
        kf = [(0, 70.0)]
        for i, energy in enumerate(analysis.energy_segments):
            frame = int(total_frames * i / max(1, len(analysis.energy_segments) - 1))
            fov = 70.0 + (energy - 0.5) * 25.0
            kf.append((frame, float(fov)))
        return self.schedule_gen.keyframes_to_schedule_string(kf, total_frames)

    # strength schedule (for "strength" parameter used by some samplers / latents)
    def _generate_strength_schedule(self, analysis: AudioAnalysis, total_frames: int) -> str:
        base = 0.7
        kf = [(0, base)]
        for i, energy in enumerate(analysis.energy_segments):
            frame = int(total_frames * i / max(1, len(analysis.energy_segments) - 1))
            val = base + (energy - 0.5) * 0.25
            kf.append((frame, float(val)))
        return self.schedule_gen.keyframes_to_schedule_string(kf, total_frames)

    # CFG schedule (control scale)
    def _generate_cfg_schedule(self, analysis: AudioAnalysis, total_frames: int) -> str:
        base = 7.0
        kf = [(0, base)]
        for i, energy in enumerate(analysis.energy_segments):
            frame = int(total_frames * i / max(1, len(analysis.energy_segments) - 1))
            val = base + (energy - 0.5) * 3.0
            kf.append((frame, float(val)))
        return self.schedule_gen.keyframes_to_schedule_string(kf, total_frames)

    # noise schedule (guiding noise)
    def _generate_noise_schedule(self, analysis: AudioAnalysis, total_frames: int) -> str:
        base = 0.04
        kf = [(0, base)]
        for i, energy in enumerate(analysis.energy_segments):
            frame = int(total_frames * i / max(1, len(analysis.energy_segments) - 1))
            val = base + energy * 0.02
            kf.append((frame, float(val)))
        return self.schedule_gen.keyframes_to_schedule_string(kf, total_frames)

    # contrast schedule
    def _generate_contrast_schedule(self, analysis: AudioAnalysis, total_frames: int) -> str:
        base = 1.0
        kf = [(0, base)]
        for i, energy in enumerate(analysis.energy_segments):
            frame = int(total_frames * i / max(1, len(analysis.energy_segments) - 1))
            val = base + (energy - 0.5) * 0.25
            kf.append((frame, float(val)))
        return self.schedule_gen.keyframes_to_schedule_string(kf, total_frames)

    def _generate_diffusion_cadence_schedule(self, analysis: AudioAnalysis, total_frames: int) -> str:
        # cadence 1 or 2 depending on beats
        kf = [(0, 1)]
        for i, t in enumerate(analysis.beat_frames[:200]):
            frame = int(t * 24)
            if frame >= total_frames:
                continue
            cadence = 2 if (i % 8 == 0) else 1
            kf.append((frame, int(cadence)))
        parts = []
        for frame, val in kf:
            parts.append(f"{frame}:({int(val)})")
        return ", ".join(parts)

    def _generate_color_coherence_schedule(self, analysis: AudioAnalysis, total_frames: int) -> str:
        kf = [(0, 0.3)]
        for i, energy in enumerate(analysis.energy_segments):
            frame = int(total_frames * i / max(1, len(analysis.energy_segments) - 1))
            coherence = max(0.05, 0.5 - energy * 0.3)
            kf.append((frame, float(coherence)))
        return self.schedule_gen.keyframes_to_schedule_string(kf, total_frames)

    # prompt generation
    def generate_prompts(self, analysis: AudioAnalysis, user_settings: Dict[str, Any]) -> Dict[str, str]:
        base_prompt = user_settings.get("base_prompt", "cinematic masterpiece, highly detailed")
        style_prompt = user_settings.get("style_prompt", "film grain, dynamic lighting")

        prompts: Dict[str, str] = {}
        nseg = max(1, len(analysis.energy_segments))
        for i in range(nseg):
            frame_idx = int(analysis.duration * 24 * i / max(1, nseg - 1)) if nseg > 1 else 0
            energy = analysis.energy_segments[i] if i < len(analysis.energy_segments) else 0.5
            parts = [base_prompt]

            # energy-driven modifiers
            if energy >= 0.8:
                parts.append("epic, cinematic, high energy, dynamic movement, dramatic lighting")
            elif energy >= 0.6:
                parts.append("bold composition, vibrant motion, cinematic lighting")
            elif energy >= 0.4:
                parts.append("gentle movement, soft lighting, cinematic composition")
            else:
                parts.append("calm, tranquil, dreamlike, atmospheric")

            # spectral based
            brightness = analysis.spectral_features.get("brightness", 0.5)
            warmth = analysis.spectral_features.get("warmth", 0.5)
            if brightness > 0.6:
                parts.append("bright lighting, high key, vivid colors")
            elif brightness < 0.35:
                parts.append("low key, moody, cinematic shadows")
            if warmth > 0.6:
                parts.append("warm tones, golden hour ambience")
            elif warmth < 0.35:
                parts.append("cool tones, ethereal blues")

            # lyric emotions
            for emotion in analysis.lyric_emotions[:3]:
                if emotion == "joy":
                    parts.append("uplifting, celebratory, radiant")
                elif emotion == "energy":
                    parts.append("electric, intense, visceral")
                elif emotion == "peace":
                    parts.append("serene, meditative, tranquil")
                elif emotion == "melancholy":
                    parts.append("wistful, melancholic, cinematic sadness")
                elif emotion == "love":
                    parts.append("romantic, tender, intimate")
                elif emotion == "mystery":
                    parts.append("mysterious, enigmatic, nocturnal")

            # visual keywords from lyrics
            if analysis.visual_elements:
                parts.append("featuring " + ", ".join(analysis.visual_elements[:3]))

            # style
            parts.append(style_prompt)

            prompts[str(frame_idx)] = ", ".join(parts)

        # ensure a prompt at frame 0
        if "0" not in prompts:
            prompts["0"] = f"{base_prompt}, {style_prompt}"

        return prompts

# ------------------------------------------------------------
# Gradio UI: create_gradio_interface (accepts analyzer/generator optionally)
# ------------------------------------------------------------
def create_gradio_interface(analyzer: Optional[AudioAnalyzer] = None, generator: Optional[DeforumMusicGenerator] = None):
    if not GRADIO_AVAILABLE:
        raise ImportError("Gradio not available. Install with: pip install gradio")
    # create defaults if not provided
    analyzer = analyzer if analyzer is not None else AudioAnalyzer(max_duration=MAX_AUDIO_DURATION)
    generator = generator if generator is not None else DeforumMusicGenerator()

    import gradio as gr  # local reference

    with gr.Blocks(title="Enhanced Deforum Music Generator", theme=getattr(gr.themes, "Soft", None)() if hasattr(gr, "themes") else None) as interface:
        gr.Markdown("""
# Enhanced Deforum Music Generator
Turn any soundtrack into audio-reactive Deforum settings.
- Fully automatic schedule & prompt generation
- Optional lyrics transcription (Whisper)
- Export a Deforum settings package (.json + analysis)
""")

        with gr.Tab("Audio Analysis"):
            with gr.Row():
                with gr.Column(scale=1):
                    audio_input = gr.Audio(label="Upload Music File", type="filepath")
                    with gr.Row():
                        max_duration = gr.Slider(minimum=10, maximum=MAX_AUDIO_DURATION, value=MAX_AUDIO_DURATION, step=10, label="Max Duration (seconds)")
                        enable_lyrics = gr.Checkbox(label="Enable Lyrics Analysis (Whisper)", value=False)
                    analyze_btn = gr.Button("Analyze Audio", variant="primary")
                with gr.Column(scale=1):
                    analysis_display = gr.JSON(label="Analysis Results")
                    analysis_summary = gr.Markdown(value="Analysis summary will appear here.")

        with gr.Tab("Generation Settings"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Prompts")
                    base_prompt = gr.Textbox(label="Base Prompt", value="cinematic masterpiece, highly detailed", lines=3)
                    style_prompt = gr.Textbox(label="Style Prompt", value="film grain, dynamic lighting", lines=2)
                    negative_prompt = gr.Textbox(label="Negative Prompt", value="low quality, blurry, distorted, watermark, text", lines=2)
                with gr.Column(scale=1):
                    gr.Markdown("### Video Settings")
                    width = gr.Slider(minimum=256, maximum=2048, value=1024, step=64, label="Width")
                    height = gr.Slider(minimum=256, maximum=2048, value=576, step=64, label="Height")
                    fps = gr.Slider(minimum=8, maximum=60, value=24, step=1, label="FPS")
                with gr.Column(scale=1):
                    gr.Markdown("### Generation Settings")
                    steps = gr.Slider(minimum=5, maximum=100, value=25, step=1, label="Steps")
                    cfg_scale = gr.Slider(minimum=1.0, maximum=30.0, value=7.0, step=0.5, label="CFG Scale")
                    seed = gr.Number(label="Seed (-1 for random)", value=-1)
                    sampler = gr.Dropdown(choices=DEFAULT_SAMPLERS, value=DEFAULT_SAMPLERS[0], label="Sampler")

        with gr.Tab("Generate"):
            with gr.Row():
                with gr.Column(scale=1):
                    batch_name = gr.Textbox(label="Batch Name", value=f"DefMusic_{now_timestring()}")
                    generate_btn = gr.Button("Generate Deforum Settings Package", variant="primary")
                    generation_log = gr.Textbox(label="Generation Log", lines=10, interactive=False)
                with gr.Column(scale=1):
                    settings_output = gr.JSON(label="Generated Settings", visible=False)
                    package_file = gr.File(label="Download Package", visible=False)

        with gr.Tab("Info"):
            deps = [
                ("NumPy", NUMPY_AVAILABLE),
                ("Librosa", LIBROSA_AVAILABLE),
                ("Whisper", WHISPER_AVAILABLE),
                ("SciPy (peak detection)", SCIPY_AVAILABLE),
                ("Automatic1111 integration", A1111_AVAILABLE and SCRIPT_CALLBACKS_AVAILABLE)
            ]
            deps_md_lines = ["### System Status", "**Dependencies:**"]
            for name, ok in deps:
                deps_md_lines.append(f"- {name}: {'Available' if ok else 'Missing'}")
            deps_md = "\n".join(deps_md_lines)
            gr.Markdown(deps_md)
            gr.Markdown("""
**How it works**
1. Upload audio and analyze to extract tempo, energy, spectral features and (optionally) lyrics.
2. Generator converts that analysis into Deforum schedules and dynamic prompts.
3. Export the package and import into Automatic1111's Deforum extension.

Supported formats: MP3, WAV, M4A, FLAC, OGG
""")

        # ----- Event handlers -----
        def run_analysis_fn(audio_path: str, max_dur_val: float, lyrics_enabled: bool):
            if not audio_path:
                return {}, "No audio file provided."

            try:
                analyzer.max_duration = int(max_dur_val)
                analysis = analyzer.analyze(audio_path, enable_lyrics=lyrics_enabled)
                # build summary markdown
                summary = f"**Analysis Complete**\n\n- File: `{os.path.basename(audio_path)}`\n- Duration: {analysis.duration:.1f}s\n- Tempo: {analysis.tempo_bpm:.1f} BPM ({analysis.rhythm_pattern})\n- Tempo confidence: {analysis.tempo_confidence:.2f}\n- Energy segments: {len(analysis.energy_segments)}\n- Reactive points: {len(analysis.audio_reactive_points)}\n"
                if analysis.raw_text:
                    summary += f"- Lyrics length: {len(analysis.raw_text)} characters\n"
                if analysis.lyric_emotions:
                    summary += f"- Lyric emotions: {', '.join(analysis.lyric_emotions)}\n"
                if analysis.visual_elements:
                    summary += f"- Visual keywords: {', '.join(analysis.visual_elements[:6])}\n"
                return analysis.to_dict(), summary
            except Exception as e:
                tb = traceback.format_exc()
                print("[run_analysis_fn] Exception:", e, tb)
                return {"error": str(e)}, f"Analysis failed: {e}"

        def generate_settings_fn(audio_path: str, base_prompt_text: str, style_prompt_text: str, neg_prompt_text: str,
                                 width_val: int, height_val: int, fps_val: int, steps_val: int, cfg_val: float,
                                 seed_val: int, sampler_val: str, batch_name_val: str):
            log_lines: List[str] = []
            if not audio_path:
                return gr.update(visible=False), gr.update(visible=False), "No audio file provided."

            try:
                log_lines.append("Analyzing audio...")
                analysis = analyzer.analyze(audio_path, enable_lyrics=True)
                log_lines.append(f"Audio - duration {analysis.duration:.1f}s, tempo {analysis.tempo_bpm:.1f} BPM")

                # prepare user settings
                seed_final = int(seed_val) if int(seed_val) != -1 else random.randint(1, 2**31 - 1)
                user_settings = {
                    "base_prompt": base_prompt_text,
                    "style_prompt": style_prompt_text,
                    "negative_prompt": neg_prompt_text,
                    "width": int(width_val),
                    "height": int(height_val),
                    "fps": int(fps_val),
                    "steps": int(steps_val),
                    "cfg_scale": float(cfg_val),
                    "seed": seed_final,
                    "sampler": sampler_val,
                    "batch_name": batch_name_val,
                    "soundtrack_path": audio_path
                }

                log_lines.append("Generating Deforum schedules & prompts...")
                settings = generator.build_deforum_settings(analysis, user_settings)

                # Save to temporary folder and zip
                log_lines.append("Creating package files...")
                temp_dir = tempfile.mkdtemp(prefix="deforum_music_")
                settings_path = os.path.join(temp_dir, "deforum_settings.json")
                dump_json_file(settings_path, settings)

                analysis_report_path = os.path.join(temp_dir, "analysis_report.json")
                dump_json_file(analysis_report_path, analysis.to_dict())

                readme_path = os.path.join(temp_dir, "README.md")
                readme_contents = f"# Deforum Music Generation Results\n\nGenerated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\nAudio file: {os.path.basename(audio_path)}\nDuration: {analysis.duration:.1f} s\nTempo: {analysis.tempo_bpm:.1f} BPM\nPrompts: {len(settings.get('prompts', {}))} segments\nFrames: {settings.get('max_frames')}\n\nFiles:\n- deforum_settings.json\n- analysis_report.json\n\nImport deforum_settings.json into Deforum.\n"
                with open(readme_path, 'w', encoding='utf-8') as f:
                    f.write(readme_contents)

                package_name = f"{batch_name_val}_{now_timestring()}.zip"
                package_path = os.path.join(temp_dir, package_name)
                with zipfile.ZipFile(package_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                    zf.write(settings_path, arcname="deforum_settings.json")
                    zf.write(analysis_report_path, arcname="analysis_report.json")
                    zf.write(readme_path, arcname="README.md")

                log_lines.append("Package ready: " + package_path)
                log_msg = "\n".join(log_lines)
                # Make settings visible in UI and package file available for download
                return gr.update(value=settings, visible=True), gr.update(value=package_path, visible=True), log_msg

            except Exception as e:
                tb = traceback.format_exc()
                print("[generate_settings_fn] Exception:", e, tb)
                return gr.update(visible=False), gr.update(visible=False), f"Generation failed: {e}\n\n{tb}"

        # wire up events
        analyze_btn.click(fn=run_analysis_fn, inputs=[audio_input, max_duration, enable_lyrics], outputs=[analysis_display, analysis_summary])
        generate_btn.click(fn=generate_settings_fn,
                           inputs=[audio_input, base_prompt, style_prompt, negative_prompt, width, height, fps, steps, cfg_scale, seed, sampler, batch_name],
                           outputs=[settings_output, package_file, generation_log])

    # return the Blocks object
    return interface

# ------------------------------------------------------------
# Standalone package creation function (CLI-friendly)
# ------------------------------------------------------------
def create_package(audio_file: str, output_dir: Optional[str] = None, **kwargs) -> str:
    if not output_dir:
        output_dir = tempfile.mkdtemp(prefix="deforum_music_")
    safe_makedirs(output_dir)

    analyzer = AudioAnalyzer(max_duration=MAX_AUDIO_DURATION)
    generator = DeforumMusicGenerator()

    analysis = analyzer.analyze(audio_file, kwargs.get("enable_lyrics", False))
    user_settings = {
        "base_prompt": kwargs.get("base_prompt", "cinematic masterpiece, highly detailed"),
        "style_prompt": kwargs.get("style_prompt", "film grain, dynamic lighting"),
        "negative_prompt": kwargs.get("negative_prompt", "low quality, blurry"),
        "width": kwargs.get("width", 1024),
        "height": kwargs.get("height", 576),
        "fps": kwargs.get("fps", 24),
        "steps": kwargs.get("steps", 25),
        "cfg_scale": kwargs.get("cfg_scale", 7.0),
        "seed": kwargs.get("seed", -1),
        "sampler": kwargs.get("sampler", "Euler a"),
        "batch_name": kwargs.get("batch_name", f"DefMusic_{now_timestring()}"),
        "soundtrack_path": audio_file
    }
    settings = generator.build_deforum_settings(analysis, user_settings)

    settings_path = os.path.join(output_dir, "deforum_settings.json")
    dump_json_file(settings_path, settings)

    analysis_path = os.path.join(output_dir, "analysis_report.json")
    dump_json_file(analysis_path, analysis.to_dict())

    # write README
    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(f"Generated on {time.asctime()}\nSoundtrack: {audio_file}\nFrames: {settings.get('max_frames')}\n")

    # create zip
    package_zip = os.path.join(output_dir, f"{user_settings['batch_name']}.zip")
    with zipfile.ZipFile(package_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(settings_path, "deforum_settings.json")
        zf.write(analysis_path, "analysis_report.json")
        zf.write(readme_path, "README.md")

    return package_zip

# ------------------------------------------------------------
# CLI entrypoint
# ------------------------------------------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Enhanced Deforum Music Generator")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    ui_parser = subparsers.add_parser("ui", help="Launch local Gradio UI")
    ui_parser.add_argument("--port", type=int, default=7860)
    ui_parser.add_argument("--share", action="store_true", default=False)

    gen_parser = subparsers.add_parser("generate", help="Generate package from audio (CLI)")
    gen_parser.add_argument("audio_file", help="Path to audio file")
    gen_parser.add_argument("--output", "-o", help="Output directory")
    gen_parser.add_argument("--base-prompt", default="cinematic masterpiece, highly detailed")
    gen_parser.add_argument("--style-prompt", default="film grain, dynamic lighting")
    gen_parser.add_argument("--width", type=int, default=1024)
    gen_parser.add_argument("--height", type=int, default=576)
    gen_parser.add_argument("--fps", type=int, default=24)
    gen_parser.add_argument("--steps", type=int, default=25)
    gen_parser.add_argument("--cfg-scale", type=float, default=7.0)
    gen_parser.add_argument("--seed", type=int, default=-1)
    gen_parser.add_argument("--sampler", default=DEFAULT_SAMPLERS[0])
    gen_parser.add_argument("--lyrics", action="store_true", default=False)

    analyze_parser = subparsers.add_parser("analyze", help="Analyze audio only")
    analyze_parser.add_argument("audio_file", help="Path to audio file")
    analyze_parser.add_argument("--lyrics", action="store_true", default=False)

    args = parser.parse_args()

    if args.command == "ui":
        if not GRADIO_AVAILABLE:
            print("Gradio not installed. pip install gradio")
            return 1
        interface = create_gradio_interface()
        interface.launch(server_port=args.port, share=args.share, inbrowser=True)
        return 0

    elif args.command == "generate":
        try:
            out = create_package(
                args.audio_file,
                args.output,
                base_prompt=args.base_prompt,
                style_prompt=args.style_prompt,
                width=args.width,
                height=args.height,
                fps=args.fps,
                steps=args.steps,
                cfg_scale=args.cfg_scale,
                seed=args.seed,
                sampler=args.sampler,
                enable_lyrics=args.lyrics
            )
            print("Package created at:", out)
            return 0
        except Exception as e:
            traceback.print_exc()
            print("Error generating package:", e)
            return 1

    elif args.command == "analyze":
        try:
            analyzer = AudioAnalyzer()
            analysis = analyzer.analyze(args.audio_file, enable_lyrics=args.lyrics)
            print(json.dumps(analysis.to_dict(), indent=2))
            return 0
        except Exception as e:
            traceback.print_exc()
            print("Error analyzing:", e)
            return 1

    else:
        parser.print_help()
        return 1

# ------------------------------------------------------------
# Automatic1111 integration
# ------------------------------------------------------------
# We register a script tab via script_callbacks.on_ui_tabs, which is the preferred method
def register_a1111_tab():
    # if A1111 isn't present, skip
    if not (A1111_AVAILABLE and SCRIPT_CALLBACKS_AVAILABLE and GRADIO_AVAILABLE):
        return

    # define a callback that returns (gr.Blocks, title, id)
    def on_ui_tabs():
        try:
            analyzer = AudioAnalyzer(max_duration=MAX_AUDIO_DURATION)
            generator = DeforumMusicGenerator()
            interface = create_gradio_interface(analyzer=analyzer, generator=generator)
            # return (interface, "Tab Title", "internal_id")
            return (interface, "Enhanced Deforum Music Generator", "deforum_music_generator_tab")
        except Exception as e:
            print("[register_a1111_tab] Failed to create UI tab:", e)
            traceback.print_exc()
            return None

    # register it
    try:
        script_callbacks.on_ui_tabs(on_ui_tabs)
    except Exception as e:
        print("[register_a1111_tab] script_callbacks.on_ui_tabs failed:", e)
        traceback.print_exc()

# run registration now (if in A1111)
if A1111_AVAILABLE and SCRIPT_CALLBACKS_AVAILABLE:
    register_a1111_tab()

# ------------------------------------------------------------
# Entrypoint guard: when run as a script
# ------------------------------------------------------------
if __name__ == "__main__":
    if DEVELOPMENT_SEED is not None:
        random.seed(DEVELOPMENT_SEED)
        if NUMPY_AVAILABLE:
            np.random.seed(DEVELOPMENT_SEED)
    sys.exit(main())
