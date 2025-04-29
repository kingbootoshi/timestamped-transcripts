import multiprocessing as mp
import logging
import os
import queue
from pathlib import Path
import time
from typing import Dict, Any, Optional, Callable

from whisperx_gui.logging_config import setup_logging
from whisperx_gui.transcriber.core import TranscriptionManager

class TranscriptionWorker:
    """
    Worker process for handling transcription in the background.
    
    This class is designed to run in a separate process, communicate via queues,
    and forward logs back to the main process.
    """
    
    def __init__(self, 
                 task_queue: mp.Queue, 
                 result_queue: mp.Queue,
                 log_queue: Optional[queue.SimpleQueue] = None):
        """
        Initialize the worker with communication queues.
        
        Args:
            task_queue: Queue for receiving tasks from main process
            result_queue: Queue for sending results back to main process
            log_queue: Queue for forwarding logs to main process
        """
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.log_queue = log_queue
        self.transcription_manager = None
        
    def setup_logging(self) -> None:
        """Configure worker-specific logging that forwards to main process."""
        log_level = os.environ.get("LOG_LEVEL", "INFO")
        self.log_queue = setup_logging(log_level=log_level, where="worker")
        
    def initialize_transcriber(self, model_size: str, language: str, hf_token: str) -> None:
        """
        Initialize the transcription manager with the specified parameters.
        
        Args:
            model_size: WhisperX model size
            language: Language code
            hf_token: Hugging Face token
        """
        logger = logging.getLogger(__name__)
        logger.info(f"Initializing transcriber with model_size={model_size}, language={language}")
        self.transcription_manager = TranscriptionManager(model_size, language)
        self.transcription_manager.load_models(hf_token)
        
    def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a transcription task.
        
        Args:
            task: Dictionary containing task parameters
            
        Returns:
            Dictionary with results and metadata
        """
        logger = logging.getLogger(__name__)
        
        task_type = task.get("type")
        if task_type == "initialize":
            logger.info("Processing initialization task")
            self.initialize_transcriber(
                task.get("model_size", "medium"),
                task.get("language", "en"),
                task.get("hf_token", "")
            )
            return {"status": "initialized"}
            
        elif task_type == "transcribe":
            if not self.transcription_manager:
                return {"status": "error", "error": "Transcriber not initialized"}
                
            audio_path = Path(task.get("audio_path", ""))
            if not audio_path.exists():
                return {"status": "error", "error": f"Audio file not found: {audio_path}"}
                
            logger.info(f"Processing transcription task for {audio_path}")
            start_time = time.time()
            
            try:
                result = self.transcription_manager.transcribe_audio(
                    audio_path,
                    min_speakers=task.get("min_speakers", 2),
                    max_speakers=task.get("max_speakers", 4)
                )
                
                processing_time = time.time() - start_time
                logger.info(f"Transcription completed in {processing_time:.1f}s")
                
                return {
                    "status": "success",
                    "result": result,
                    "processing_time": processing_time,
                    "audio_path": str(audio_path)
                }
                
            except Exception as e:
                logger.error(f"Error in transcription: {e}")
                return {
                    "status": "error",
                    "error": str(e),
                    "audio_path": str(audio_path)
                }
                
        else:
            logger.warning(f"Unknown task type: {task_type}")
            return {"status": "error", "error": f"Unknown task type: {task_type}"}
            
    def run(self) -> None:
        """
        Main worker loop. Continuously processes tasks from the queue.
        """
        # Set up worker-specific logging
        self.setup_logging()
        logger = logging.getLogger(__name__)
        logger.info("Transcription worker started")
        
        while True:
            try:
                # Get a task from the queue
                task = self.task_queue.get()
                
                # Check for termination signal
                if task is None or task.get("type") == "terminate":
                    logger.info("Received termination signal, shutting down worker")
                    break
                    
                # Process the task
                result = self.process_task(task)
                
                # Send the result back
                self.result_queue.put(result)
                
            except Exception as e:
                logger.error(f"Error in worker: {e}")
                # Send error result back
                self.result_queue.put({
                    "status": "error",
                    "error": str(e)
                })
                
        logger.info("Worker process terminated")


def start_worker_process(task_queue: mp.Queue, 
                         result_queue: mp.Queue,
                         log_queue: Optional[queue.SimpleQueue] = None) -> mp.Process:
    """
    Create and start a worker process.
    
    Args:
        task_queue: Queue for tasks
        result_queue: Queue for results
        log_queue: Queue for logs
        
    Returns:
        Started worker process
    """
    def worker_target():
        worker = TranscriptionWorker(task_queue, result_queue, log_queue)
        worker.run()
        
    process = mp.Process(target=worker_target)
    process.daemon = True
    process.start()
    return process


class QueueListener:
    """
    A utility class to listen to a log queue and handle incoming log records.
    
    This class should be run in the main process to receive logs from worker processes.
    """
    
    def __init__(self, log_queue: queue.SimpleQueue, callback: Optional[Callable] = None):
        """
        Initialize the queue listener.
        
        Args:
            log_queue: Queue containing log records from workers
            callback: Optional callback function to handle new log entries
        """
        self.log_queue = log_queue
        self.callback = callback
        self.running = False
        self.logger = logging.getLogger(__name__)
        
    def start(self) -> None:
        """Start listening to the queue in a separate thread."""
        import threading
        self.running = True
        self.thread = threading.Thread(target=self._listen)
        self.thread.daemon = True
        self.thread.start()
        self.logger.debug("Log queue listener started")
        
    def stop(self) -> None:
        """Stop the listener thread."""
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=1.0)
        self.logger.debug("Log queue listener stopped")
        
    def _listen(self) -> None:
        """Listen for records and process them."""
        while self.running:
            try:
                # Non-blocking get with timeout
                try:
                    record = self.log_queue.get(block=True, timeout=0.5)
                    self._process_record(record)
                except queue.Empty:
                    continue
            except Exception as e:
                self.logger.error(f"Error in log queue listener: {e}")
                
    def _process_record(self, record) -> None:
        """Process a log record by logging it to the main process logger."""
        # First, log the record in the main process
        logger = logging.getLogger(record.name)
        logger.handle(record)
        
        # If a callback is provided, call it with the record
        if self.callback:
            try:
                self.callback(record)
            except Exception as e:
                self.logger.error(f"Error in callback: {e}")