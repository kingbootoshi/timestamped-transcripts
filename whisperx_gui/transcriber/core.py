import logging
from pathlib import Path
import torch
from typing import Dict, Any, Optional
from whisperx.diarize import DiarizationPipeline

logger = logging.getLogger(__name__)

class TranscriptionManager:
    """
    Core domain logic for WhisperX transcription operations.
    
    This class provides the core functionality needed for transcription, 
    separated from UI and worker concerns.
    """
    
    def __init__(self, model_size: str = "medium", language: str = "en"):
        """
        Initialize the transcription manager.
        
        Args:
            model_size: WhisperX model size ('tiny', 'base', 'small', 'medium', 'large')
            language: Language code for transcription
        """
        self.model_size = model_size
        self.language = language
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dia_pipeline: Optional[DiarizationPipeline] = None
        logger.debug(f"TranscriptionManager initialized with model size {model_size}, language {language}, device {self.device}")
        
    def load_models(self, hf_token: str) -> None:
        """
        Load WhisperX and diarization models.
        
        Args:
            hf_token: Hugging Face token for diarization model access
        """
        try:
            import whisperx
            logger.info(f"Loading WhisperX {self.model_size} model on {self.device}")
            self.model = whisperx.load_model(
                self.model_size,
                self.device,
                compute_type="float16" if self.device == "cuda" else "int8"
            )
            
            # Load alignment model
            logger.debug(f"Loading alignment model for language: {self.language}")
            self.model_align, self.metadata = whisperx.load_align_model(
                language_code=self.language, 
                device=self.device
            )
            
            # Load diarization pipeline if token provided
            if hf_token:
                logger.info("Loading diarization pipeline")
                self.dia_pipeline = DiarizationPipeline(
                    use_auth_token=hf_token,
                    device=self.device
                )
            else:
                logger.warning("No Hugging Face token provided, diarization will be disabled")
                
            logger.info("All models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
            
    def transcribe_audio(self, audio_path: Path, min_speakers: int = 2, max_speakers: int = 4) -> Dict[str, Any]:
        """
        Perform transcription and optional diarization on an audio file.
        
        Args:
            audio_path: Path to audio file
            min_speakers: Minimum number of speakers for diarization
            max_speakers: Maximum number of speakers for diarization
            
        Returns:
            Dictionary with transcription results
        """
        if not hasattr(self, 'model'):
            raise RuntimeError("Models not loaded. Call load_models() first.")
            
        logger.info(f"Transcribing audio: {audio_path}")
        
        try:
            import whisperx
            
            # Transcribe
            logger.debug("Starting transcription")
            result = self.model.transcribe(str(audio_path), language=self.language, batch_size=16)
            logger.debug("Transcription complete")
            
            # Word-level alignment
            logger.debug("Starting word alignment")
            result = whisperx.align(
                result["segments"], 
                self.model_align, 
                self.metadata,
                str(audio_path), 
                self.device
            )
            logger.debug("Word alignment complete")
            
            # Speaker diarization (optional)
            if self.dia_pipeline:
                logger.info("Running speaker diarization")
                try:
                    dia_segments = self.dia_pipeline(
                        str(audio_path),
                        min_speakers=min_speakers,
                        max_speakers=max_speakers
                    )
                    result = whisperx.assign_word_speakers(dia_segments, result)
                    logger.debug("Speaker diarization and assignment complete")
                except Exception as e:
                    logger.error(f"Error during diarization: {e}")
                    logger.warning("Skipping speaker assignment")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in transcription: {e}")
            raise