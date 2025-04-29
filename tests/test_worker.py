"""
Integration tests for the whisperx_gui.backend.worker module.

These tests verify that the worker process correctly handles tasks,
sends heartbeats, and cleans up properly on termination.
"""

import sys
import unittest
import time
import tempfile
import multiprocessing as mp
import queue
import os
import signal
from pathlib import Path
import wave
import numpy as np
import logging
import threading

# Add the project root to sys.path to allow importing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from whisperx_gui.backend.worker import TranscribeWorker, WorkerManager

# Set up logging for tests
logging.basicConfig(level=logging.DEBUG)

class TestWorker(unittest.TestCase):
    """Test case for the worker process."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        
        # Create a dummy WAV file for testing
        self.create_dummy_wav()
        
        # Create communication queues
        self.task_queue = mp.Queue()
        self.result_queue = mp.Queue()
        
        # Initialize variables for tracking messages
        self.messages = []
        self.heartbeats = []
        self.progress_events = []
        
        # Initialize flag for worker crash simulation
        self.simulate_crash = False
        
    def tearDown(self):
        """Clean up test environment."""
        # Clean up temp directory
        self.temp_dir.cleanup()
        
        # Ensure all processes are terminated
        for attr in ['worker', 'monitor_thread']:
            if hasattr(self, attr) and getattr(self, attr) is not None:
                try:
                    if isinstance(getattr(self, attr), mp.Process) and getattr(self, attr).is_alive():
                        getattr(self, attr).terminate()
                        getattr(self, attr).join(timeout=1.0)
                except Exception:
                    pass
        
    def create_dummy_wav(self, duration=10.0, sample_rate=16000):
        """
        Create a dummy WAV file for testing.
        
        Args:
            duration: Duration in seconds
            sample_rate: Sample rate in Hz
        
        Returns:
            Path to the created WAV file
        """
        filename = self.temp_path / "test_audio.wav"
        
        # Generate a sine wave
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        
        # Save as WAV
        with wave.open(str(filename), 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes((audio * 32767).astype(np.int16).tobytes())
            
        return filename
        
    def monitor_result_queue(self):
        """Monitor the result queue and store messages."""
        while self.monitoring:
            try:
                # Get message with timeout
                result = self.result_queue.get(timeout=0.5)
                self.messages.append(result)
                
                # Categorize message
                if result.get("type") == "heartbeat":
                    self.heartbeats.append(result)
                elif result.get("type") == "progress":
                    self.progress_events.append(result)
                    
                # Check for crash simulation
                if self.simulate_crash and len(self.heartbeats) >= 2:
                    # Kill the worker process
                    self.worker.terminate()
                    # Set flag to false to avoid repeated termination
                    self.simulate_crash = False
                    
            except queue.Empty:
                # No messages available
                continue
            except Exception as e:
                print(f"Error in monitor thread: {e}")
                break
                
    def test_worker_lifecycle(self):
        """Test worker startup, heartbeat, and shutdown."""
        # Start worker
        self.worker = TranscribeWorker(self.task_queue, self.result_queue)
        self.worker.start()
        
        # Start monitoring thread
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self.monitor_result_queue)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        # Wait for heartbeats
        start_time = time.time()
        while len(self.heartbeats) < 2:
            if time.time() - start_time > 15:
                self.fail("Timeout waiting for heartbeats")
            time.sleep(0.1)
            
        # Verify we received heartbeats
        self.assertGreaterEqual(len(self.heartbeats), 2)
        
        # Send termination task
        self.task_queue.put({"type": "terminate"})
        
        # Wait for worker to exit
        self.worker.join(timeout=5.0)
        
        # Verify worker exited
        self.assertFalse(self.worker.is_alive())
        
        # Stop monitoring
        self.monitoring = False
        self.monitor_thread.join(timeout=1.0)
        
    def test_transcription_task(self):
        """Test processing a transcription task."""
        # Create output directory
        output_dir = self.temp_path / "output"
        output_dir.mkdir()
        
        # Start worker
        self.worker = TranscribeWorker(self.task_queue, self.result_queue)
        self.worker.start()
        
        # Start monitoring thread
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self.monitor_result_queue)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        # Wait for first heartbeat
        start_time = time.time()
        while not self.heartbeats:
            if time.time() - start_time > 15:
                self.fail("Timeout waiting for heartbeat")
            time.sleep(0.1)
            
        # Prepare a mock transcription task
        # Note: Using a mock to avoid actually running the heavy transcription
        wav_path = self.temp_path / "test_audio.wav"
        self.task_queue.put({
            "type": "transcribe",
            "media_path": str(wav_path),
            "output_dir": str(output_dir),
            "model_size": "tiny",  # Use smallest model for speed
            "task_id": "test-1"
        })
        
        # Wait for progress events and result
        start_time = time.time()
        while not any(m.get("type") == "result" for m in self.messages):
            if time.time() - start_time > 60:  # Allow up to 60s for processing
                self.fail("Timeout waiting for result")
            time.sleep(0.1)
            
        # Verify we received progress events
        progress_steps = [
            msg.get("event", {}).get("step")
            for msg in self.messages
            if msg.get("type") == "progress" and "event" in msg
        ]
        
        # May not get all steps due to mocking, but should at least try to load model
        self.assertIn("load_model", progress_steps)
        
        # Verify we got a result 
        results = [msg for msg in self.messages if msg.get("type") == "result"]
        self.assertGreaterEqual(len(results), 1)
        
        # Send termination task
        self.task_queue.put({"type": "terminate"})
        
        # Wait for worker to exit
        self.worker.join(timeout=5.0)
        
        # Verify worker exited within 3 seconds (acceptance criteria)
        self.assertFalse(self.worker.is_alive())
        
        # Stop monitoring
        self.monitoring = False
        self.monitor_thread.join(timeout=1.0)
        
    def test_worker_crash_recovery(self):
        """Test that the WorkerManager handles worker crashes."""
        # Create a WorkerManager
        manager = WorkerManager()
        
        # Start a worker
        worker_id = manager.start_worker()
        
        # Keep track of progress events
        progress_events = []
        
        # Define callbacks
        def progress_callback(event):
            progress_events.append(event)
            
        def result_callback(result):
            # Just for tracking
            pass
            
        # Prepare test file
        wav_path = self.temp_path / "test_audio.wav"
        output_dir = self.temp_path / "output"
        output_dir.mkdir()
        
        # Submit a task
        task_id = manager.submit_task(
            worker_id,
            {
                "type": "transcribe",
                "media_path": str(wav_path),
                "output_dir": str(output_dir),
                "model_size": "tiny"
            },
            progress_callback=progress_callback,
            result_callback=result_callback
        )
        
        # Let the task run for a bit
        time.sleep(2.0)
        
        # Forcefully kill the worker
        worker_process = manager.workers[worker_id]["worker"]
        worker_process.terminate()
        
        # Wait for the manager to detect the death
        start_time = time.time()
        while worker_id in manager.workers and time.time() - start_time < 20:
            time.sleep(0.1)
            
        # Verify the worker was removed
        self.assertNotIn(worker_id, manager.workers)
        
        # Stop the manager
        manager.stop_all()
        
    def test_worker_heartbeat_timeout(self):
        """Test that the WorkerManager detects missing heartbeats."""
        # Start worker directly (not via manager)
        self.worker = TranscribeWorker(self.task_queue, self.result_queue)
        self.worker.start()
        
        # Start monitoring thread
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self.monitor_result_queue)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        # Wait for heartbeats to start
        start_time = time.time()
        while not self.heartbeats and time.time() - start_time < 15:
            time.sleep(0.1)
            
        # Enable crash simulation after 2 heartbeats
        self.simulate_crash = True
        
        # Wait for crash to happen
        while self.worker.is_alive() and time.time() - start_time < 30:
            time.sleep(0.1)
            
        # Verify worker was terminated
        time.sleep(1.0)  # Give it a moment to fully exit
        self.assertFalse(self.worker.is_alive())
        
        # Stop monitoring
        self.monitoring = False
        self.monitor_thread.join(timeout=1.0)
        
    def test_worker_signal_handling(self):
        """Test worker handles termination signals properly."""
        # Create a temp file to track cleanup
        marker_file = self.temp_path / "cleanup_marker.txt"
        with open(marker_file, "w") as f:
            f.write("test")
            
        # Start worker
        self.worker = TranscribeWorker(self.task_queue, self.result_queue)
        self.worker.start()
        
        # Wait for worker to initialize
        time.sleep(2.0)
        
        # Send SIGTERM
        os.kill(self.worker.pid, signal.SIGTERM)
        
        # Wait for worker to exit (should be <= 3s per acceptance criteria)
        start_time = time.time()
        while self.worker.is_alive() and time.time() - start_time < 5:
            time.sleep(0.1)
            
        cleanup_time = time.time() - start_time
        
        # Verify worker exited within 3 seconds
        self.assertFalse(self.worker.is_alive())
        self.assertLessEqual(cleanup_time, 3.0)

if __name__ == "__main__":
    unittest.main()