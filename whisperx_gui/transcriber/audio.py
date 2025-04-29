"""
Audio processing utilities for WhisperX transcription.
"""

import logging
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

def extract_audio(video_path: Path, audio_out: Optional[Path] = None) -> Path:
    """
    Extract mono 16 kHz WAV from video or audio file using ffmpeg.
    
    Args:
        video_path: Path to the input video or audio file
        audio_out: Optional output path for the audio file. If not provided,
                   it will use the same name as the input with .wav extension.
    
    Returns:
        Path to the extracted audio file
    """
    audio_out = audio_out or video_path.with_suffix(".wav")
    logger.info(f"Extracting audio from {video_path} to {audio_out}")
    
    cmd = [
        "ffmpeg", "-i", str(video_path),
        "-vn", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1",
        str(audio_out), "-y"
    ]
    
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return audio_out
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg error: {e.stderr.decode() if e.stderr else str(e)}")
        raise RuntimeError(f"Failed to extract audio from {video_path}") from e
    except Exception as e:
        logger.error(f"Error extracting audio: {e}")
        raise

def duration_seconds(media_path: Path) -> float:
    """
    Get the duration in seconds of a media file using ffprobe.
    
    Args:
        media_path: Path to the media file
        
    Returns:
        Duration in seconds
    """
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(media_path)
    ]
    
    try:
        result = subprocess.check_output(cmd).decode().strip()
        return float(result)
    except subprocess.CalledProcessError as e:
        logger.error(f"FFprobe error: {e.stderr.decode() if e.stderr else str(e)}")
        raise RuntimeError(f"Failed to get duration for {media_path}") from e
    except (ValueError, TypeError) as e:
        logger.error(f"Invalid duration format: {e}")
        raise RuntimeError(f"Failed to parse duration for {media_path}") from e
    except Exception as e:
        logger.error(f"Error getting duration: {e}")
        raise