"""
Gradio Interface Module
Clean, organized interface for the enhanced Deforum music generator
"""

import gradio as gr
from typing import Tuple, Optional
import traceback

def create_interface(generator):
    """Create the main Gradio interface"""
    
    def process_music_wrapper(*args):
        """Wrapper function for processing with proper error handling"""
        try:
            # Unpack arguments
            (audio_file, base_prompt, style_prompt, width, height, fps, steps, 
             cfg_scale, sampler, seed, model_name, max_duration, enable_lyrics, 
             chunk_processing) = args
            
            if not audio_file:
                return "❌ No audio file provided", "{}", None
            
            # Build settings dict
            settings = {
                'fps': fps,
                'width': width,
                'height': height,
                'steps': steps,
                'cfg_scale': cfg_scale,
                'sampler': sampler,
                'seed': seed if seed != -1 else None,
                'model_name': model_name,
                'max_duration': max_duration,
                'enable_lyrics': enable_lyrics,
                'chunk_processing': chunk_processing,
                'base_prompt': base_prompt,
                'style_prompt': style_prompt
            }
            
            # Process the file
            result = generator.process_music_file(audio_file, settings)
            
            if result['success']:
                # Format preview settings
                preview_settings = _format_settings_preview(result['settings'])
                
                return (
                    result['status_report'],
                    preview_settings,
                    result['output_package']
                )
            else:
                error_msg = f"❌ Processing failed: {result['error']}"
                if generator.config.interface.show_error_details:
                    error_msg += f"\n\nDetails:\n{result.get('traceback', 'No details available')}"
                
                return error_msg, str(result['error']), None
                
        except Exception as e:
            error_msg = f"❌ Interface error: {str(e)}"
            if generator.config.interface.show_error_details:
                error_msg += f"\n\nTraceback:\n{traceback.format_exc()}"
            return error_msg, str(e), None
    
    # Get A1111 integration data
    try:
        from integration.a1111_connector import A1111Connector
        a1111 = A1111Connector(generator.config.a1111)
        available_models = a1111.get_models()
        available_samplers = a1111.get_samplers()
    except Exception:
        available_models = ["Default"]
        available_samplers = ["Euler a", "DPM++ 2M Karras", "DDIM"]
    
    # Create the interface
    with gr.Blocks(
        title="Enhanced A1111 Deforum Music Generator",
        theme=gr.themes.Soft(),
        css="""
        .main-header { 
            text-align: center; 
            padding: 20px;
            background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
            color: white;
            margin-bottom: 20px;
            border-radius: 10px;
        }
        .preset-buttons {
            display: flex;
            gap: 10px;
            margin: 10px 0;
        }
        .status-output {
            font-family: monospace;
            font-size: 12px;
        }
        """
    ) as app:
        
        gr.HTML("""
        <div class="main-header">
            <h1>🎵 Enhanced A1111 Deforum Music Generator</h1>
            <p>Advanced audio-reactive animation with improved performance and quality</p>
            <p><small>Refactored architecture • Memory-efficient processing • Intelligent analysis</small></p>
        </div>
        """)
        
        with gr.Row():
            # Left Column - Input Settings
            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown("### 🎵 Audio Input")
                    
                    audio_input = gr.Audio(
                        label="Music File",
                        type="filepath",
                        info="MP3, WAV, M4A supported"
                    )
                    
                    with gr.Row():
                        max_duration = gr.Slider(
                            minimum=30,
                            maximum=600,
                            value=generator.config.audio.max_duration,
                            step=30,
                            label="Max Duration (seconds)",
                            info="Longer = more processing time"
                        )
                        
                        enable_lyrics = gr.Checkbox(
                            label="Lyric Analysis",
                            value=generator.config.lyrics.enable_transcription,
                            info="Uses Whisper AI (slower but better)"
                        )
                    
                    chunk_processing = gr.Checkbox(
                        label="Memory-Efficient Processing",
                        value=True,
                        info="Recommended for tracks > 2 minutes"
                    )
                
                with gr.Group():
                    gr.Markdown("### 🎨 Visual Prompts")
                    
                    base_prompt = gr.Textbox(
                        label="Base Visual Prompt",
                        value="cinematic masterpiece, professional photography, dramatic composition, highly detailed",
                        lines=3,
                        info="Core visual description"
                    )
                    
                    style_prompt = gr.Textbox(
                        label="Style & Quality Modifiers",
                        value="film grain, depth of field, dynamic lighting, vibrant colors, high contrast, sharp focus",
                        lines=2,
                        info="Technical quality and style terms"
                    )
            
            # Right Column - Generation Settings
            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown("### ⚙️ Generation Settings")
                    
                    with gr.Row():
                        width = gr.Slider(
                            256, 1536, generator.config.interface.default_width,
                            step=64, label="Width"
                        )
                        height = gr.Slider(
                            256, 1536, generator.config.interface.default_height,
                            step=64, label="Height"
                        )
                    
                    with gr.Row():
                        fps = gr.Slider(
                            12, 30, generator.config.interface.default_fps,
                            step=1, label="FPS"
                        )
                        steps = gr.Slider(
                            15, 50, generator.config.interface.default_steps,
                            step=1, label="Steps"
                        )
                    
                    with gr.Row():
                        cfg_scale = gr.Slider(
                            3.0, 15.0, generator.config.interface.default_cfg_scale,
                            step=0.5, label="CFG Scale"
                        )
                        seed = gr.Number(
                            -1, precision=0, label="Seed (-1 = random)"
                        )
                
                with gr.Group():
                    gr.Markdown("### 🤖 Model Settings")
                    
                    model_dropdown = gr.Dropdown(
                        choices=available_models,
                        value=available_models[0],
                        label="Model",
                        info="A1111 checkpoint model"
                    )
                    
                    sampler_dropdown = gr.Dropdown(
                        choices=available_samplers,
                        value=available_samplers[0],
                        label="Sampler",
                        info="Sampling algorithm"
                    )
        
        # Generation Button
        with gr.Row():
            generate_btn = gr.Button(
                "Generate Enhanced Settings",
                variant="primary",
                size="lg",
                scale=2
            )
        
        # Quick Presets
        with gr.Row():
            gr.Markdown("**Quick Presets:**")
        
        with gr.Row():
            preset_music_video = gr.Button("Music Video", size="sm")
            preset_cinematic = gr.Button("Cinematic", size="sm")
            preset_abstract = gr.Button("Abstract Art", size="sm")
            preset_performance = gr.Button("Fast Processing", size="sm")
        
        # Output Section
        gr.Markdown("---")
        
        with gr.Row():
            with gr.Column(scale=1):
                status_output = gr.Textbox(
                    label="Processing Status & Analysis",
                    lines=18,
                    max_lines=25,
                    elem_classes="status-output",
                    info="Detailed analysis results and processing information"
                )
            
            with gr.Column(scale=1):
                preview_output = gr.Code(
                    label="Settings Preview",
                    language="json",
                    lines=18,
                    info="Preview of generated Deforum settings"
                )
        
        download_output = gr.File(
            label="Download Complete Package",
            info="ZIP file with settings, analysis, and documentation"
        )
        
        # Event Handlers
        generate_btn.click(
            fn=process_music_wrapper,
            inputs=[
                audio_input, base_prompt, style_prompt, width, height,
                fps, steps, cfg_scale, sampler_dropdown, seed, model_dropdown,
                max_duration, enable_lyrics, chunk_processing
            ],
            outputs=[status_output, preview_output, download_output],
            show_progress=True
        )
        
        # Preset Functions
        def apply_music_video_preset():
            return (
                "dynamic music video, concert performance, stage lighting, energetic atmosphere, live performance",
                "concert photography, stage lights, dynamic composition, high energy, motion blur, vibrant colors",
                512, 896, 30, 20, 7.0
            )
        
        def apply_cinematic_preset():
            return (
                "cinematic scene, movie-like storytelling, professional cinematography, dramatic narrative",
                "anamorphic lens, film grain, color grading, cinematic lighting, depth of field, 35mm film",
                1024, 576, 24, 30, 7.5
            )
        
        def apply_abstract_preset():
            return (
                "abstract art visualization, flowing forms, geometric patterns, artistic interpretation, fluid motion",
                "abstract expressionism, vibrant gradients, creative visuals, artistic composition, flowing colors",
                1024, 1024, 24, 25, 8.0
            )
        
        def apply_performance_preset():
            return (
                "clean composition, optimized for speed, efficient processing, simple forms",
                "clean lines, high contrast, minimal complexity, optimized rendering",
                512, 512, 20, 20, 6.0
            )
        
        # Wire up preset buttons
        preset_music_video.click(
            lambda: apply_music_video_preset(),
            outputs=[base_prompt, style_prompt, width, height, fps, steps, cfg_scale]
        )
        preset_cinematic.click(
            lambda: apply_cinematic_preset(),
            outputs=[base_prompt, style_prompt, width, height, fps, steps, cfg_scale]
        )
        preset_abstract.click(
            lambda: apply_abstract_preset(),
            outputs=[base_prompt, style_prompt, width, height, fps, steps, cfg_scale]
        )
        preset_performance.click(
            lambda: apply_performance_preset(),
            outputs=[base_prompt, style_prompt, width, height, fps, steps, cfg_scale]
        )
        
        # Documentation Section
        with gr.Accordion("Documentation & Features", open=False):
            gr.Markdown(_create_documentation_content(generator.config))
    
    return app

def _format_settings_preview(settings: dict) -> str:
    """Format settings for JSON preview"""
    preview = {
        "generation": {
            "resolution": f"{settings.get('W', 1024)}x{settings.get('H', 576)}",
            "fps": settings.get('fps', 24),
            "total_frames": settings.get('max_frames', 0),
            "steps": settings.get('steps', 25),
            "cfg_scale": settings.get('scale', 7.0),
            "sampler": settings.get('sampler', 'Euler a')
        },
        "animation": {
            "mode": settings.get('animation_mode', '3D'),
            "zoom_keyframes": _count_keyframes(settings.get('zoom', '')),
            "strength_keyframes": _count_keyframes(settings.get('strength_schedule', '')),
            "has_camera_movement": any(
                settings.get(key, '0:(0)') != '0:(0)' 
                for key in ['translation_x', 'translation_y', 'translation_z']
            ),
            "has_rotation": any(
                settings.get(key, '0:(0)') != '0:(0)' 
                for key in ['rotation_3d_x', 'rotation_3d_y', 'rotation_3d_z']
            )
        },
        "audio_reactive": {
            "beat_synchronized": True,
            "energy_responsive": True,
            "spectral_guided": True
        },
        "sample_prompts": {
            "frame_0": list(settings.get('prompts', {}).values())[0] if settings.get('prompts') else "No prompts generated"
        }
    }
    
    import json
    return json.dumps(preview, indent=2)

def _count_keyframes(schedule_string: str) -> int:
    """Count keyframes in a schedule string"""
    if not schedule_string or schedule_string == "0:(0)":
        return 1
    return len(schedule_string.split(','))

def _create_documentation_content(config) -> str:
    """Create documentation content"""
    return f"""
## Enhanced Features

### Memory-Efficient Processing
- **Chunked Analysis**: Processes audio in {config.audio.chunk_size}s segments to reduce memory usage
- **Intelligent Caching**: Content-based caching prevents redundant analysis
- **Configurable Limits**: Max duration of {config.audio.max_duration}s prevents excessive processing

### Advanced Audio Analysis
- **Multi-Algorithm Beat Detection**: Combines multiple techniques with confidence scoring
- **Spectral Feature Analysis**: Maps frequency content to visual properties
- **Energy Segmentation**: Divides audio into {config.audio.energy_segments} segments for precise mapping
- **Reactive Point Generation**: Identifies key moments for animation emphasis

### Intelligent Animation Generation
- **Smooth Interpolation**: Uses {config.animation.default_easing} easing for natural motion
- **Anti-Jitter Protection**: Prevents rapid movements that cause rendering artifacts
- **Energy-Responsive Scaling**: Animation intensity matches musical energy
- **Contextual Camera Movement**: Movement only added when musically appropriate

### Enhanced Lyric Processing
- **Whisper Integration**: Uses {config.lyrics.whisper_model_size} model for transcription
- **Emotion Classification**: Detects up to {config.lyrics.max_emotions_per_analysis} emotions with confidence scoring
- **Visual Element Mapping**: Extracts visual concepts from lyrics
- **Narrative Structure Analysis**: Identifies story progression patterns

## Usage Tips

### Audio Input
- Use high-quality files (320kbps MP3 or uncompressed WAV)
- Enable lyric analysis for vocal tracks
- Use memory-efficient processing for tracks longer than 2 minutes
- Set appropriate max duration based on your system capabilities

### Visual Prompts  
- Base prompt: Core visual description (objects, scene, style)
- Style prompt: Technical terms (lighting, camera, quality)
- Combine abstract concepts with specific visual elements
- Consider the emotional content when crafting prompts

### Performance Optimization
- Lower resolution/FPS for faster processing
- Disable lyric analysis for instrumental music
- Use "Fast Processing" preset for quick tests
- Enable chunked processing for long tracks

### Quality Settings
- Higher steps = better quality but slower generation  
- CFG Scale 7-9 recommended for most content
- Use appropriate sampler for your model
- Seed consistency allows reproducible results

## Configuration
Current configuration summary:
- Audio: {config.audio.max_duration}s max, chunking {'enabled' if config.audio.chunk_size else 'disabled'}
- Lyrics: {'Enabled' if config.lyrics.enable_transcription else 'Disabled'} with {config.lyrics.whisper_model_size} model
- Animation: {config.animation.default_easing} easing, rotation {'enabled' if config.animation.enable_rotation else 'disabled'}
- Integration: A1111 {'enabled' if config.a1111.enable_integration else 'disabled'}
"""
                        