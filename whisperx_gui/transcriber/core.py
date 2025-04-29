"""
Core transcription functionality for WhisperX.
"""

import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, Union, Callable
from datetime import timedelta

from .progress import ProgressEvent, ProgressCallback, null_progress
from .audio import extract_audio, duration_seconds

logger = logging.getLogger(__name__)

def format_timestamp(seconds: float) -> str:
    """HH:MM:SS formatter (zero-padded)."""
    return str(timedelta(seconds=int(seconds)))

def build_dia_pipeline(hf_token: str, device: Optional[str] = None) -> Any:
    """
    Build and return a speaker diarization pipeline.
    
    Args:
        hf_token: Hugging Face token for accessing pyannote models
        device: Device to run the pipeline on ("cuda" or "cpu"). 
                If None, will be auto-detected.
    
    Returns:
        DiarizationPipeline object
    """
    # Lazy import to avoid loading torch at module level
    try:
        import torch
        from whisperx.diarize import DiarizationPipeline
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
        logger.info(f"Creating diarization pipeline on {device}")
        
        return DiarizationPipeline(
            use_auth_token=hf_token,
            device=device
        )
    except ImportError as e:
        logger.error(f"Error importing required modules: {e}")
        raise
    except Exception as e:
        logger.error(f"Error building diarization pipeline: {e}")
        raise

def write_markdown(result: Dict[str, Any], md_path: Path) -> None:
    """
    Write transcription results to a markdown file.
    
    Args:
        result: Transcription result dictionary
        md_path: Output markdown file path
    """
    md_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Writing transcript to {md_path}")
    
    with md_path.open("w", encoding="utf-8") as f:
        # Full transcript (speaker‑labeled)
        f.write("# Full Transcript\n\n")
        last_speaker = None
        for seg in result["segments"]:
            speaker = seg.get("speaker", "UNKNOWN")
            text = seg["text"].strip()
            # Start a new paragraph when the speaker changes
            if speaker != last_speaker:
                if last_speaker is not None:
                    f.write("\n")         # blank line between speaker turns
                f.write(f"**{speaker}:** {text}")
            else:
                f.write(f" {text}")       # same speaker: append in the same line
            last_speaker = speaker
        f.write("\n\n")
        
        # Segment‑level with speaker labels
        f.write("# Timestamped Transcript\n\n")
        for idx, seg in enumerate(result["segments"], 1):
            start = format_timestamp(seg["start"])
            end = format_timestamp(seg["end"])
            speaker = seg.get("speaker", "UNKNOWN")
            f.write(f"## Segment {idx}: [{start} - {end}] ({speaker})\n\n")
            f.write(seg["text"].strip() + "\n\n")
            
            # Word timestamps - Modified to show speaker once
            if seg.get("words"):
                f.write("### Word-level timestamps\n\n")
                current_speaker = None
                for w in seg["words"]:
                    w_speaker = w.get("speaker", speaker)
                    # Only write speaker when it changes
                    if w_speaker != current_speaker:
                        f.write(f"\n**{w_speaker}:**\n")  # New speaker header
                        current_speaker = w_speaker
                    # Just timestamp and word for subsequent entries
                    f.write(f"- {w['word'].strip()} @ {format_timestamp(w['start'])}\n")
                f.write("\n")
        
        # Summary
        f.write("## Summary\n\n")
        f.write(f"Total segments: {len(result['segments'])}\n")
        if result["segments"]:
            total_dur = result["segments"][-1]["end"] - result["segments"][0]["start"]
            f.write(f"Total duration: {format_timestamp(total_dur)}\n")

def transcribe(
    media_path: Path,
    output_dir: Path,
    *,
    model_size: str = "medium",
    language: str = "en",
    hf_token: Optional[str] = None,
    min_speakers: int = 2,
    max_speakers: int = 4,
    overwrite: bool = False,
    progress: ProgressCallback = null_progress,
    dia_pipeline: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Process a media file through WhisperX transcription and (optionally) diarization.
    
    Args:
        media_path: Path to the video or audio file
        output_dir: Directory for transcript output
        model_size: WhisperX model size ("tiny", "base", "small", "medium", "large")
        language: Language code (e.g., "en", "fr")
        hf_token: Hugging Face token for diarization (if None and dia_pipeline is None, 
                 diarization will be skipped)
        min_speakers: Minimum number of speakers for diarization
        max_speakers: Maximum number of speakers for diarization
        overwrite: Whether to overwrite existing transcripts
        progress: Callback function for progress updates
        dia_pipeline: Pre-built diarization pipeline (optional)
        
    Returns:
        Dictionary with transcription results
    """
    # Lazy imports to avoid loading modules at module level
    try:
        import torch
        import whisperx
    except ImportError as e:
        error_msg = f"Required modules not found: {e}"
        logger.error(error_msg)
        progress({"step": "error", "pct": None, "msg": error_msg})
        raise
    
    # Prepare output path and check if already exists
    md_path = output_dir / f"{media_path.stem}.md"
    if md_path.exists() and not overwrite:
        logger.info(f"Skipping {media_path.name} (already transcribed)")
        progress({"step": "done", "pct": 100, "msg": f"Already transcribed: {media_path.name}"})
        return {"status": "skipped", "path": str(md_path)}
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Start timing
    start_time = time.time()
    
    try:
        # Get media duration for progress reporting
        media_duration = duration_seconds(media_path)
        logger.info(f"Media duration: {format_timestamp(media_duration)}")
        
        # Extract audio (if video)
        temp_audio = extract_audio(media_path)
        
        # Load model
        progress({"step": "load_model", "pct": 10, "msg": f"Loading WhisperX {model_size} model"})
        logger.info(f"Loading WhisperX {model_size} model on {device}")
        model = whisperx.load_model(
            model_size, 
            device, 
            compute_type="float16" if device == "cuda" else "int8"
        )
        
        # Transcribe
        progress({"step": "transcribe", "pct": 30, "msg": "Transcribing audio"})
        logger.debug("Starting transcription")
        result = model.transcribe(str(temp_audio), language=language, batch_size=16)
        logger.debug("Transcription complete")
        
        # Align
        progress({"step": "align", "pct": 50, "msg": "Aligning words with audio"})
        logger.debug(f"Loading alignment model for language: {language}")
        model_a, metadata = whisperx.load_align_model(language_code=language, device=device)
        logger.debug("Starting word alignment")
        result = whisperx.align(
            result["segments"], 
            model_a, 
            metadata,
            str(temp_audio), 
            device
        )
        logger.debug("Word alignment complete")
        
        # Speaker diarization (optional)
        if dia_pipeline or hf_token:
            progress({"step": "diarize", "pct": 70, "msg": "Identifying speakers"})
            logger.info("Running speaker diarization")
            
            try:
                # Use provided pipeline or create one
                pipeline = dia_pipeline
                if pipeline is None and hf_token:
                    pipeline = build_dia_pipeline(hf_token, device)
                
                # Run diarization
                dia_segments = pipeline(
                    str(temp_audio),
                    min_speakers=min_speakers,
                    max_speakers=max_speakers
                )
                result = whisperx.assign_word_speakers(dia_segments, result)
                logger.debug("Speaker diarization and assignment complete")
            except Exception as e:
                error_msg = f"Error during diarization: {e}"
                logger.error(error_msg)
                logger.warning("Skipping speaker assignment")
                # We continue without diarization rather than failing completely
        
        # Write markdown
        progress({"step": "write_md", "pct": 90, "msg": "Writing transcript"})
        write_markdown(result, md_path)
        
        # Clean up
        if temp_audio.name != media_path.name:
            temp_audio.unlink(missing_ok=True)
        
        # Report completion
        processing_time = time.time() - start_time
        logger.info(f"Finished in {processing_time:.1f}s")
        progress({"step": "done", "pct": 100, "msg": f"Completed in {processing_time:.1f}s"})
        
        # Add metadata to result
        result["processing_time"] = processing_time
        result["output_path"] = str(md_path)
        result["status"] = "success"
        
        return result
        
    except Exception as e:
        error_msg = f"Error processing {media_path}: {e}"
        logger.error(error_msg)
        progress({"step": "error", "pct": None, "msg": error_msg})
        raise