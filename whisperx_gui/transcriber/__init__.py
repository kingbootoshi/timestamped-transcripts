"""
Domain logic for WhisperX transcription.

This module provides a clean API for transcription operations while encapsulating 
the heavy-lifting related to WhisperX.

Public API:
- transcribe: Main function for processing a video or audio file to transcript
- build_dia_pipeline: Create a speaker diarization pipeline
"""

from .audio import extract_audio, duration_seconds
from .progress import ProgressEvent
from .core import transcribe, build_dia_pipeline

__all__ = [
    "transcribe", 
    "build_dia_pipeline",
    "ProgressEvent", 
    "extract_audio", 
    "duration_seconds"
]