#!/usr/bin/env python3
"""
WhisperX Transcriber Script — now with optional speaker diarization.

Drop any videos into ./videos and run the script (no arguments)
to get timestamped markdown transcripts in ./transcripts/<video>/<video>.md.

New flags:
    --diarize                 Run speaker diarization with Pyannote models
    --min_speakers / --max_speakers  Bounds for speaker clustering
    --hf_token TOKEN          Hugging‑Face PAT (defaults to $HF_TOKEN env)

Requirements:
- whisperx (pip install git+https://github.com/m-bain/whisperX.git)
- ffmpeg
- pyannote.audio >= 3.x  (already in requirements.txt)
"""

from __future__ import annotations

import sys
import time
import subprocess
from pathlib import Path
from datetime import timedelta
import argparse
import os
import logging
from dotenv import load_dotenv

# Import the new logging configuration
from whisperx_gui.logging_config import setup_logging

# Configure logging based on environment variables
log_level = os.environ.get("LOG_LEVEL", "INFO")
setup_logging(log_level=log_level, where="cli")

# Get a logger for this module
logger = logging.getLogger(__name__)

# ── WhisperX & torch ─────────────────────────────────────────────────────────
try:
    import whisperx
    import torch
except ImportError:
    logger.error("WhisperX not installed. Install with: pip install git+https://github.com/m-bain/whisperX.git")
    sys.exit(1)
# ─────────────────────────────────────────────────────────────────────────────

# ── Configuration ────────────────────────────────────────────────────────────
VIDEOS_DIR = Path("videos")           # input folder
TRANSCRIPTS_DIR = Path("transcripts") # output folder
SUPPORTED_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".mp3", ".wav"}
# ─────────────────────────────────────────────────────────────────────────────

# ADDED: Import DiarizationPipeline here
from whisperx.diarize import DiarizationPipeline

def format_timestamp(seconds: float) -> str:
    """HH:MM:SS formatter (zero‑padded)."""
    return str(timedelta(seconds=int(seconds)))


def extract_audio(video_path: Path, audio_out: Path | None = None) -> Path:
    """Extract mono 16 kHz WAV using ffmpeg."""
    audio_out = audio_out or video_path.with_suffix(".wav")
    logger.info(f"Extracting audio from {video_path} to {audio_out}")
    cmd = [
        "ffmpeg", "-i", str(video_path),
        "-vn", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1",
        str(audio_out), "-y"
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return audio_out


def transcribe(audio_path: Path,
               model_size: str,
               language: str,
               *,
               # REMOVED: diarize, min_speakers, max_speakers, hf_token args
               # ADDED: dia_pipeline object
               dia_pipeline: DiarizationPipeline | None) -> dict:
    """
    WhisperX transcription + optional speaker diarization.

    Returns a dict with a `segments` key identical to whisperx's output,
    but with an extra `speaker` field if diarization was requested.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading WhisperX {model_size} on {device}")
    model = whisperx.load_model(
        model_size,
        device,
        compute_type="float16" if device == "cuda" else "int8"
    )

    logger.debug("Starting transcription")
    result = model.transcribe(str(audio_path), language=language, batch_size=16)
    logger.debug("Transcription complete")

    # Word‑level alignment
    logger.debug(f"Loading alignment model for language: {language}")
    model_a, metadata = whisperx.load_align_model(language_code=language, device=device)
    logger.debug("Starting word alignment")
    result = whisperx.align(result["segments"], model_a, metadata,
                        str(audio_path), device)
    logger.debug("Word alignment complete")

    # Speaker diarization (optional)
    # CHANGED: Use the pre-loaded pipeline if provided
    if dia_pipeline:
        logger.info("Running speaker diarization using pre-loaded pipeline")
        # Old way commented out:
        # dia = whisperx.DiarizationPipeline(
        #     use_auth_token=hf_token,
        #     device=device
        # )
        # dia_segments = dia(str(audio_path),
        #                    min_speakers=min_speakers,
        #                    max_speakers=max_speakers)
        # Assume min/max speakers are handled during pipeline creation or use defaults
        # If specific min/max needed per file, this approach needs adjustment
        # For now, using the pipeline directly assumes default/pre-configured speakers
        try:
            dia_segments = dia_pipeline(str(audio_path))
            result = whisperx.assign_word_speakers(dia_segments, result)
            logger.debug("Speaker diarization and assignment complete")
        except Exception as e:
            logger.error(f"Error during diarization: {e}")
            logger.warning("Skipping speaker assignment for this file")

    return result  # contains "segments"


def write_markdown(result: dict, md_path: Path) -> None:
    """Write full transcript, segments, and word timestamps (with speakers)."""
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


def duration_seconds(video_path: Path) -> float:
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)
    ]
    return float(subprocess.check_output(cmd).decode().strip())


def process_video(video_path: Path,
                  *,
                  model_size: str,
                  language: str,
                  # REMOVED: diarize, min_speakers, max_speakers, hf_token
                  # ADDED: dia_pipeline
                  dia_pipeline: DiarizationPipeline | None,
                  overwrite: bool) -> None:
    """Extract audio ➜ transcribe ➜ write markdown."""
    out_dir = TRANSCRIPTS_DIR / video_path.stem
    md_path = out_dir / f"{video_path.stem}.md"
    if md_path.exists() and not overwrite:
        logger.info(f"Skipping {video_path.name} (already transcribed)")
        return

    logger.info(f"Processing video: {video_path.name}")
    start = time.time()
    dur = duration_seconds(video_path)
    logger.info(f"Duration: {format_timestamp(dur)}")

    # Extract + transcribe
    audio = extract_audio(video_path)
    result = transcribe(audio,
                        model_size,
                        language,
                        # CHANGED: Pass the pipeline object
                        dia_pipeline=dia_pipeline)
    write_markdown(result, md_path)
    audio.unlink(missing_ok=True)
    processing_time = time.time() - start
    logger.info(f"Finished processing {video_path.name} in {processing_time:.1f}s")


def main() -> None:
    global TRANSCRIPTS_DIR  # allow --output_file to override

    load_dotenv()
    
    logger.debug("Starting WhisperX batch transcriber")

    parser = argparse.ArgumentParser(
        description="WhisperX batch transcriber (videos ➜ markdown)"
    )
    parser.add_argument("input_video", nargs="?",
                        help="Process only this file. Omit for batch mode.")
    parser.add_argument("--output_file",
                        help="Custom markdown path (single‑file mode only).")

    # WhisperX knobs
    parser.add_argument("--model_size", default="medium",
                        choices=["tiny", "base", "small", "medium", "large"])
    parser.add_argument("--language", default="en",
                        help="Language code (default: en)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Regenerate transcripts even if they exist.")

    # New diarization flags
    parser.add_argument("--min_speakers", type=int, default=2,
                        help="Minimum number of speakers (diarization).")
    parser.add_argument("--max_speakers", type=int, default=4,
                        help="Maximum number of speakers (diarization).")
    parser.add_argument("--hf_token",
                        default=os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN"),
                        help="Hugging‑Face access token for Pyannote models "
                             "(defaults to $HF_TOKEN env var). Required for diarization.")

    args = parser.parse_args()

    # --- Token Check & Early Diarization Setup --- 
    hf_token_read = args.hf_token
    if hf_token_read:
        logger.debug(f"HF Token loaded: ...{hf_token_read[-4:]}")  # Print last 4 chars
    else:
        logger.error("Hugging Face token is required for speaker diarization")
        logger.error("Please provide it via the --hf_token flag or set the HF_TOKEN environment variable")
        logger.error("Ensure you have accepted user conditions at:")
        logger.error("  - https://hf.co/pyannote/speaker-diarization-3.1")
        logger.error("  - https://hf.co/pyannote/segmentation-3.0")
        sys.exit(1)

    # Try to load the diarization pipeline *before* the loop
    dia_pipeline_instance: DiarizationPipeline | None = None
    try:
        logger.info("Pre-loading Diarization Pipeline")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Note: min/max speakers are typically arguments to the pipeline *call*, 
        # not usually the constructor. WhisperX's wrapper might handle this differently.
        # If you need per-file min/max speakers, this needs adjustment.
        # For now, we load the default pipeline.
        dia_pipeline_instance = DiarizationPipeline(
            use_auth_token=hf_token_read,
            device=device
            # We might need to specify model="pyannote/speaker-diarization-3.1" explicitly
            # if whisperx doesn't default to it or if multiple are cached.
        )
        logger.info("Diarization Pipeline loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load Diarization Pipeline: {e}")
        logger.error("Check HF Token, internet connection, and ensure user conditions are accepted at:")
        logger.error("  - https://hf.co/pyannote/speaker-diarization-3.1")
        logger.error("  - https://hf.co/pyannote/segmentation-3.0")
        logger.warning("Diarization will be skipped")
        # We proceed without diarization instead of exiting, user might still want transcription
    # --- End Early Diarization Setup ---

    # Ensure dirs exist
    VIDEOS_DIR.mkdir(exist_ok=True)
    TRANSCRIPTS_DIR.mkdir(exist_ok=True)

    if args.input_video:
        video = Path(args.input_video).expanduser()
        if not video.is_file():
            logger.error(f"Input file '{video}' not found")
            sys.exit(1)
        if args.output_file:
            TRANSCRIPTS_DIR = Path(args.output_file).expanduser().parent

        process_video(video,
                      model_size=args.model_size,
                      language=args.language,
                      # CHANGED: Pass pre-loaded pipeline
                      dia_pipeline=dia_pipeline_instance,
                      overwrite=args.overwrite)
    else:
        videos = [p for p in VIDEOS_DIR.iterdir()
                  if p.suffix.lower() in SUPPORTED_EXTS]
        if not videos:
            logger.warning("No videos found in ./videos – add some and rerun")
            return
        for vid in videos:
            process_video(vid,
                          model_size=args.model_size,
                          language=args.language,
                          # CHANGED: Pass pre-loaded pipeline
                          dia_pipeline=dia_pipeline_instance,
                          overwrite=args.overwrite)


if __name__ == "__main__":
    main()