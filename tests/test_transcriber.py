"""
Unit tests for the whisperx_gui.transcriber module.

These tests mock WhisperX dependencies to test the transcription
workflow without requiring actual models or media files.
"""

import sys
import unittest
from unittest.mock import patch, MagicMock, call
import tempfile
from pathlib import Path
import time
import importlib

# Add the project root to sys.path to allow importing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the module under test
from whisperx_gui.transcriber import transcribe, ProgressEvent

class TestTranscriber(unittest.TestCase):
    """Test case for the transcriber module."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_video = Path(self.temp_dir.name) / "test_video.mp4"
        self.test_video.touch()  # Create empty file
        self.output_dir = Path(self.temp_dir.name) / "output"
        self.output_dir.mkdir()
        
        # Mock dependencies
        self.mock_whisperx = MagicMock()
        self.mock_torch = MagicMock()
        
        # Set up model mocks
        self.mock_model = MagicMock()
        self.mock_align_model = MagicMock()
        self.mock_metadata = MagicMock()
        self.mock_dia_pipeline = MagicMock()
        
        # Configure mock returns
        self.mock_whisperx.load_model.return_value = self.mock_model
        self.mock_whisperx.load_align_model.return_value = (self.mock_align_model, self.mock_metadata)
        
        self.mock_model.transcribe.return_value = {
            "segments": [
                {"start": 0.0, "end": 5.0, "text": "Test transcription."}
            ]
        }
        
        self.mock_whisperx.align.return_value = {
            "segments": [
                {
                    "start": 0.0,
                    "end": 5.0, 
                    "text": "Test transcription.",
                    "words": [
                        {"word": "Test", "start": 0.0, "end": 1.0},
                        {"word": "transcription", "start": 1.5, "end": 4.5},
                    ]
                }
            ]
        }
        
        self.mock_dia_pipeline.return_value = {"diarization": "data"}
        
        self.mock_whisperx.assign_word_speakers.return_value = {
            "segments": [
                {
                    "start": 0.0,
                    "end": 5.0, 
                    "text": "Test transcription.",
                    "speaker": "SPEAKER_01",
                    "words": [
                        {"word": "Test", "start": 0.0, "end": 1.0, "speaker": "SPEAKER_01"},
                        {"word": "transcription", "start": 1.5, "end": 4.5, "speaker": "SPEAKER_01"},
                    ]
                }
            ]
        }
        
        # Patch subprocess for audio extraction
        self.mock_subprocess = MagicMock()
        self.mock_subprocess.run.return_value = MagicMock()
        self.mock_subprocess.check_output.return_value = b"10.5"  # Duration in seconds
        
        # Define patches
        # Important: We're patching the imports used in the transcribe function
        # not the module attributes, since the imports are lazy (done inside the function)
        self.patches = [
            patch('builtins.__import__', self._mock_import),
            patch("whisperx_gui.transcriber.audio.subprocess", self.mock_subprocess),
        ]
        
        # Apply patches
        for p in self.patches:
            p.start()
            
        # Progress tracking
        self.progress_events = []
        
    def _mock_import(self, name, *args, **kwargs):
        """Mock import to return our mocks for certain modules."""
        if name == 'torch':
            return self.mock_torch
        elif name == 'whisperx':
            return self.mock_whisperx
        else:
            # For all other imports, use the real __import__
            return self.__real_import(name, *args, **kwargs)
            
    def __real_import(self, name, *args, **kwargs):
        """The real __import__ function."""
        # Save the original __import__
        orig_import = __builtins__['__import__']
        # Remove our patch temporarily
        __builtins__['__import__'] = orig_import
        try:
            # Import the module
            result = __import__(name, *args, **kwargs)
            return result
        finally:
            # Restore our patch
            __builtins__['__import__'] = self._mock_import
        
    def tearDown(self):
        """Tear down test fixtures."""
        # Restore patches
        for p in self.patches:
            p.stop()
            
        # Clean up temp directory
        self.temp_dir.cleanup()
        
    def progress_callback(self, event: ProgressEvent):
        """Track progress events."""
        self.progress_events.append(event)
    
    def test_transcribe_happy_path(self):
        """Test the normal transcription flow with all steps succeeding."""
        # Arrange
        self.mock_torch.cuda.is_available.return_value = True  # Pretend we have CUDA
        
        # Act
        result = transcribe(
            self.test_video,
            self.output_dir,
            model_size="tiny",
            language="en",
            hf_token="fake_token",
            progress=self.progress_callback
        )
        
        # Assert
        self.assertEqual(result["status"], "success")
        self.assertTrue(Path(result["output_path"]).exists())
        
        # Check progress events order and completeness
        self.assertEqual(len(self.progress_events), 6)
        steps = [event["step"] for event in self.progress_events]
        self.assertEqual(steps, [
            "load_model", 
            "transcribe", 
            "align", 
            "diarize", 
            "write_md", 
            "done"
        ])
        
        # Verify the expected function calls
        self.mock_whisperx.load_model.assert_called_once()
        self.mock_model.transcribe.assert_called_once()
        self.mock_whisperx.load_align_model.assert_called_once()
        self.mock_whisperx.align.assert_called_once()
        self.mock_whisperx.assign_word_speakers.assert_called_once()
    
    def test_transcribe_no_diarization(self):
        """Test transcription without diarization (no HF token)."""
        # Act
        result = transcribe(
            self.test_video,
            self.output_dir,
            model_size="tiny",
            language="en",
            hf_token=None,  # No diarization
            progress=self.progress_callback
        )
        
        # Assert
        self.assertEqual(result["status"], "success")
        
        # Check progress events order and completeness
        self.assertEqual(len(self.progress_events), 5)  # One less (no diarize step)
        steps = [event["step"] for event in self.progress_events]
        self.assertEqual(steps, [
            "load_model", 
            "transcribe", 
            "align", 
            "write_md", 
            "done"
        ])
        
        # Verify no diarization was attempted
        self.mock_whisperx.assign_word_speakers.assert_not_called()
    
    def test_transcribe_already_exists(self):
        """Test that transcription is skipped if file exists and overwrite=False."""
        # Arrange
        output_file = self.output_dir / f"{self.test_video.stem}.md"
        output_file.touch()  # Create the file to trigger the "already exists" check
        
        # Act
        result = transcribe(
            self.test_video,
            self.output_dir,
            overwrite=False,  # Don't overwrite
            progress=self.progress_callback
        )
        
        # Assert
        self.assertEqual(result["status"], "skipped")
        self.assertEqual(len(self.progress_events), 1)  # Only the "done" event
        self.assertEqual(self.progress_events[0]["step"], "done")
        
        # Verify no model loading or processing was attempted
        self.mock_whisperx.load_model.assert_not_called()
    
    def test_transcribe_error_handling(self):
        """Test proper error handling and progress events on failure."""
        # Arrange
        self.mock_model.transcribe.side_effect = RuntimeError("Out of memory")
        
        # Act/Assert
        with self.assertRaises(RuntimeError):
            transcribe(
                self.test_video,
                self.output_dir,
                progress=self.progress_callback
            )
        
        # Check that error was reported in progress events
        self.assertTrue(any(event["step"] == "error" for event in self.progress_events))
        error_event = next(event for event in self.progress_events if event["step"] == "error")
        self.assertIn("Out of memory", error_event["msg"])
    
    def test_diarization_error_recovery(self):
        """Test that the process continues even if diarization fails."""
        # Arrange - Create a mock for the diarization pipeline call
        # that will raise an exception
        dia_pipeline = MagicMock()
        dia_pipeline.side_effect = Exception("Diarization error")
        
        # Act
        result = transcribe(
            self.test_video,
            self.output_dir,
            hf_token="fake_token",
            progress=self.progress_callback,
            dia_pipeline=dia_pipeline
        )
        
        # Assert
        self.assertEqual(result["status"], "success")  # Should still succeed
        
        # Check progress events - should still have all steps
        steps = [event["step"] for event in self.progress_events]
        self.assertIn("diarize", steps)  # Diarization was attempted
        self.assertIn("write_md", steps)  # But we still wrote the output
        self.assertIn("done", steps)      # And completed

    def test_import_time(self):
        """Test that importing the module is fast (< 500ms) due to lazy imports."""
        # Arrange - Make sure we have a clean import state
        # We'll use time to measure import time
        
        # Act
        if "whisperx_gui.transcriber" in sys.modules:
            del sys.modules["whisperx_gui.transcriber"]
        
        start_time = time.time()
        import whisperx_gui.transcriber
        import_time = time.time() - start_time
        
        # Assert
        self.assertLess(import_time, 0.5)  # Less than 500ms
        
        # Clean up
        del sys.modules["whisperx_gui.transcriber"]

if __name__ == "__main__":
    unittest.main()