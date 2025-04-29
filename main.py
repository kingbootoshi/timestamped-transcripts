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

# Import the logging configuration
from whisperx_gui.logging_config import setup_logging

# Configure logging based on environment variables
log_level = os.environ.get("LOG_LEVEL", "INFO")
setup_logging(log_level=log_level, where="cli")

# Get a logger for this module
logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────────────
VIDEOS_DIR = Path("videos")           # input folder
TRANSCRIPTS_DIR = Path("transcripts") # output folder
SUPPORTED_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".mp3", ".wav"}
# ─────────────────────────────────────────────────────────────────────────────

# Import from our new transcriber package
from whisperx_gui.transcriber import (
    transcribe, 
    build_dia_pipeline, 
    ProgressEvent,
    duration_seconds
)

def format_timestamp(seconds: float) -> str:
    """HH:MM:SS formatter (zero‑padded)."""
    return str(timedelta(seconds=int(seconds)))

def cli_progress_callback(event: ProgressEvent) -> None:
    """Progress callback for CLI usage."""
    # For CLI, just log the events
    if event["msg"]:
        if event["step"] == "error":
            logger.error(event["msg"])
        else:
            logger.info(event["msg"])

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
        logger.debug(f"HF Token loaded: ...{hf_token_read[-4:]}") # Print last 4 chars
    else:
        logger.warning("No Hugging Face token provided. Speaker diarization will be disabled.")
        logger.info("To enable diarization, provide a token via the --hf_token flag or set the HF_TOKEN environment variable.")
        logger.info("Ensure you have accepted user conditions at:")
        logger.info("  - https://hf.co/pyannote/speaker-diarization-3.1")
        logger.info("  - https://hf.co/pyannote/segmentation-3.0")

    # Try to load the diarization pipeline *before* the loop
    dia_pipeline_instance = None
    if hf_token_read:
        try:
            logger.info("Pre-loading Diarization Pipeline...")
            # Use our new build_dia_pipeline function
            dia_pipeline_instance = build_dia_pipeline(hf_token_read)
            logger.info("Diarization Pipeline loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load Diarization Pipeline: {e}")
            logger.error("Check HF Token, internet connection, and ensure user conditions are accepted at:")
            logger.error("  - https://hf.co/pyannote/speaker-diarization-3.1")
            logger.error("  - https://hf.co/pyannote/segmentation-3.0")
            logger.warning("Diarization will be skipped.")
            # We proceed without diarization

    # Ensure dirs exist
    VIDEOS_DIR.mkdir(exist_ok=True)
    TRANSCRIPTS_DIR.mkdir(exist_ok=True)

    if args.input_video:
        # Process a single file
        video = Path(args.input_video).expanduser()
        if not video.is_file():
            logger.error(f"Input file '{video}' not found.")
            sys.exit(1)
            
        output_dir = TRANSCRIPTS_DIR
        if args.output_file:
            output_path = Path(args.output_file).expanduser()
            output_dir = output_path.parent
            # For custom output, we pass the file name explicitly
            transcribe(
                video,
                output_dir,
                model_size=args.model_size,
                language=args.language,
                hf_token=hf_token_read,
                min_speakers=args.min_speakers,
                max_speakers=args.max_speakers,
                overwrite=args.overwrite,
                progress=cli_progress_callback,
                dia_pipeline=dia_pipeline_instance
            )
        else:
            # Standard output directory
            transcribe(
                video,
                TRANSCRIPTS_DIR,
                model_size=args.model_size,
                language=args.language,
                hf_token=hf_token_read,
                min_speakers=args.min_speakers,
                max_speakers=args.max_speakers,
                overwrite=args.overwrite,
                progress=cli_progress_callback,
                dia_pipeline=dia_pipeline_instance
            )
    else:
        # Batch processing
        videos = [p for p in VIDEOS_DIR.iterdir()
                  if p.suffix.lower() in SUPPORTED_EXTS]
        if not videos:
            logger.warning("No videos found in ./videos – add some and rerun.")
            return
            
        for vid in videos:
            logger.info(f"\n=== Processing {vid.name} ===")
            # Print video duration for user information
            try:
                dur = duration_seconds(vid)
                logger.info(f"Duration: {format_timestamp(dur)}")
            except Exception:
                logger.warning("Could not determine video duration.")
                
            # Process the video
            transcribe(
                vid,
                TRANSCRIPTS_DIR,
                model_size=args.model_size,
                language=args.language,
                hf_token=hf_token_read,
                min_speakers=args.min_speakers,
                max_speakers=args.max_speakers,
                overwrite=args.overwrite,
                progress=cli_progress_callback,
                dia_pipeline=dia_pipeline_instance
            )


if __name__ == "__main__":
    main()