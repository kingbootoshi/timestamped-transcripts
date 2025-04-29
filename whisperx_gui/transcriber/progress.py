"""
Progress tracking types and utilities for WhisperX transcription.
"""

from typing import Literal, TypedDict, Optional, Callable

class ProgressEvent(TypedDict):
    """
    Event object for tracking transcription progress.
    
    Attributes:
        step: The current processing step
        pct: Optional percentage of completion (0-100)
        msg: Optional message with details about the current step
    """
    step: Literal["load_model", "transcribe", "align", 
                 "diarize", "write_md", "done", "error"]
    pct: Optional[float]
    msg: Optional[str]

# Type definition for progress callback functions
ProgressCallback = Callable[[ProgressEvent], None]

# Default no-op progress callback
def null_progress(_: ProgressEvent) -> None:
    """Default no-op progress callback."""
    pass