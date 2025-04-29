"""
Worker process for background transcription tasks.

This module provides a multiprocessing-based worker that isolates GPU/RAM
usage from the main GUI process, communicating through JSON messages.
"""

import multiprocessing as mp
import logging
import os
import queue
import signal
import sys
import threading
import time
import json
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List, Union, Set
import traceback

from whisperx_gui.logging_config import setup_logging
from whisperx_gui.transcriber import (
    transcribe, 
    build_dia_pipeline, 
    ProgressEvent
)

# Default heartbeat interval (seconds)
HEARTBEAT_INTERVAL = 5.0
# Child processes to track for cleanup
CHILD_PROCESSES: Set[int] = set()

class TranscribeWorker(mp.Process):
    """
    Worker process for handling transcription in the background.
    
    This process is designed to run separately from the GUI process
    to isolate memory usage and prevent UI freezing.
    """
    
    def __init__(self, 
                 task_queue: mp.Queue, 
                 result_queue: mp.Queue,
                 log_queue: Optional[queue.SimpleQueue] = None):
        """
        Initialize the worker process.
        
        Args:
            task_queue: Queue for receiving tasks from main process
            result_queue: Queue for sending results and heartbeats to main process
            log_queue: Queue for forwarding logs to main process
        """
        super().__init__()
        
        # Set daemon to False to allow graceful shutdown
        self.daemon = False
        
        # Communication queues
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.log_queue = log_queue
        
        # State
        self.dia_pipeline = None
        self.running = False
        self.current_task_id = None
        self.child_pids = set()
        
    def setup_logging(self) -> None:
        """Configure worker-specific logging that forwards to main process."""
        log_level = os.environ.get("LOG_LEVEL", "INFO")
        self.log_queue = setup_logging(log_level=log_level, where="worker")
        
    def setup_signal_handlers(self) -> None:
        """Set up signal handlers for graceful shutdown."""
        # Handle termination signals
        signal.signal(signal.SIGTERM, self._handle_terminate)
        signal.signal(signal.SIGINT, self._handle_terminate)
        
    def _handle_terminate(self, signum: int, frame) -> None:
        """
        Handle termination signals by stopping gracefully.
        
        Args:
            signum: Signal number
            frame: Current stack frame
        """
        logger = logging.getLogger(__name__)
        logger.info(f"Received signal {signum}, shutting down worker")
        self.stop()
        
    def start_heartbeat(self) -> None:
        """Start the heartbeat thread."""
        self.heartbeat_thread = threading.Thread(target=self._heartbeat_loop)
        self.heartbeat_thread.daemon = True
        self.heartbeat_thread.start()
        
    def _heartbeat_loop(self) -> None:
        """Send periodic heartbeats to the main process."""
        logger = logging.getLogger(__name__)
        logger.debug("Heartbeat thread started")
        
        while self.running:
            try:
                # Send a heartbeat to the main process
                self.result_queue.put({
                    "type": "heartbeat",
                    "timestamp": time.time(),
                    "task_id": self.current_task_id
                })
                logger.debug("Heartbeat sent")
                
                # Sleep for the heartbeat interval
                time.sleep(HEARTBEAT_INTERVAL)
            except Exception as e:
                logger.error(f"Error in heartbeat thread: {e}")
                # If the queue is closed or there's another critical issue,
                # we'll exit the loop
                if not self.running:
                    break
                    
        logger.debug("Heartbeat thread stopped")
        
    def track_child_process(self, pid: int) -> None:
        """
        Track a child process for cleanup during shutdown.
        
        Args:
            pid: Process ID to track
        """
        global CHILD_PROCESSES
        CHILD_PROCESSES.add(pid)
        self.child_pids.add(pid)
        
    def cleanup_child_processes(self) -> None:
        """
        Terminate any running child processes (e.g., ffmpeg).
        """
        logger = logging.getLogger(__name__)
        
        # Try to kill tracked child processes
        for pid in self.child_pids.copy():
            try:
                logger.info(f"Terminating child process {pid}")
                os.kill(pid, signal.SIGTERM)
                self.child_pids.remove(pid)
                if pid in CHILD_PROCESSES:
                    CHILD_PROCESSES.remove(pid)
            except ProcessLookupError:
                # Process already gone
                self.child_pids.remove(pid)
                if pid in CHILD_PROCESSES:
                    CHILD_PROCESSES.remove(pid)
            except Exception as e:
                logger.error(f"Error terminating process {pid}: {e}")
        
    def initialize_diarization(self, hf_token: str) -> bool:
        """
        Initialize the diarization pipeline.
        
        Args:
            hf_token: Hugging Face token
            
        Returns:
            True if successful, False otherwise
        """
        logger = logging.getLogger(__name__)
        logger.info("Initializing diarization pipeline")
        
        try:
            # Use the transcriber's function to build the pipeline
            self.dia_pipeline = build_dia_pipeline(hf_token)
            logger.info("Diarization pipeline initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize diarization pipeline: {e}")
            return False
            
    def progress_callback(self, event: ProgressEvent) -> None:
        """
        Handle progress events from transcription.
        
        This callback sends progress updates to the main process.
        
        Args:
            event: Progress event data
        """
        logger = logging.getLogger(__name__)
        
        # Log the event
        if event["msg"]:
            if event["step"] == "error":
                logger.error(event["msg"])
            else:
                logger.info(event["msg"])
        
        # Forward the event to the main process
        try:
            self.result_queue.put({
                "type": "progress",
                "task_id": self.current_task_id,
                "event": event
            })
        except Exception as e:
            logger.error(f"Error sending progress update: {e}")
        
    def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a transcription task.
        
        Args:
            task: Dictionary containing task parameters
            
        Returns:
            Dictionary with results and metadata
        """
        logger = logging.getLogger(__name__)
        
        # Store the current task ID for heartbeats
        self.current_task_id = task.get("task_id")
        
        task_type = task.get("type")
        if task_type == "initialize_diarization":
            logger.info("Initializing diarization pipeline")
            success = self.initialize_diarization(task.get("hf_token", ""))
            return {
                "type": "result",
                "task_id": self.current_task_id,
                "status": "initialized" if success else "error",
                "error": None if success else "Failed to initialize diarization pipeline"
            }
            
        elif task_type == "transcribe":
            media_path = Path(task.get("media_path", ""))
            if not media_path.exists():
                return {
                    "type": "result",
                    "task_id": self.current_task_id,
                    "status": "error", 
                    "error": f"Media file not found: {media_path}"
                }
                
            output_dir = Path(task.get("output_dir", "transcripts"))
            output_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Processing transcription task for {media_path}")
            
            try:
                # Use the transcriber's function directly
                result = transcribe(
                    media_path,
                    output_dir,
                    model_size=task.get("model_size", "medium"),
                    language=task.get("language", "en"),
                    hf_token=task.get("hf_token"),
                    min_speakers=task.get("min_speakers", 2),
                    max_speakers=task.get("max_speakers", 4),
                    overwrite=task.get("overwrite", False),
                    progress=self.progress_callback,
                    dia_pipeline=self.dia_pipeline
                )
                
                # Send only essential information back, not the entire result
                # which could be large and cause serialization issues
                response = {
                    "type": "result",
                    "task_id": self.current_task_id,
                    "status": result.get("status", "unknown"),
                    "output_path": result.get("output_path", None),
                    "processing_time": result.get("processing_time", 0)
                }
                
                if result.get("status") == "skipped":
                    response["path"] = result.get("path")
                    
                return response
                
            except Exception as e:
                error_msg = str(e)
                stack_trace = traceback.format_exc()
                logger.error(f"Error in transcription: {error_msg}\n{stack_trace}")
                return {
                    "type": "result",
                    "task_id": self.current_task_id,
                    "status": "error",
                    "error": error_msg,
                    "media_path": str(media_path)
                }
                
        elif task_type == "terminate":
            logger.info("Received termination request")
            self.running = False
            return {
                "type": "result",
                "task_id": self.current_task_id,
                "status": "terminated"
            }
        else:
            logger.warning(f"Unknown task type: {task_type}")
            return {
                "type": "result",
                "task_id": self.current_task_id,
                "status": "error", 
                "error": f"Unknown task type: {task_type}"
            }
            
    def run(self) -> None:
        """Main worker process entry point."""
        try:
            # Set up worker-specific logging
            self.setup_logging()
            logger = logging.getLogger(__name__)
            logger.info(f"Worker process started (PID: {os.getpid()})")
            
            # Set up signal handlers
            self.setup_signal_handlers()
            
            # Start running
            self.running = True
            
            # Start heartbeat thread
            self.start_heartbeat()
            
            # Process tasks until terminated
            while self.running:
                try:
                    # Get a task from the queue with timeout
                    try:
                        task = self.task_queue.get(timeout=1.0)
                    except queue.Empty:
                        # No tasks available, continue the loop
                        continue
                    
                    # Check for termination task
                    if task is None or task.get("type") == "terminate":
                        logger.info("Received termination task, shutting down worker")
                        self.running = False
                        
                        # Acknowledge termination
                        self.result_queue.put({
                            "type": "result",
                            "task_id": task.get("task_id") if task else None,
                            "status": "terminated"
                        })
                        break
                    
                    # Process the task
                    result = self.process_task(task)
                    
                    # Send the result back
                    self.result_queue.put(result)
                    
                except Exception as e:
                    error_msg = str(e)
                    stack_trace = traceback.format_exc()
                    logger.error(f"Error in worker main loop: {error_msg}\n{stack_trace}")
                    
                    # Send error result back
                    try:
                        self.result_queue.put({
                            "type": "result",
                            "task_id": self.current_task_id,
                            "status": "error",
                            "error": error_msg
                        })
                    except Exception as send_err:
                        logger.error(f"Failed to send error message: {send_err}")
                
            logger.info("Worker process terminated normally")
            
        except Exception as e:
            # Critical error in worker setup
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            if hasattr(self, 'result_queue'):
                try:
                    self.result_queue.put({
                        "type": "critical_error",
                        "error": error_msg,
                        "stack_trace": stack_trace
                    })
                except Exception:
                    pass  # Can't do anything if queue is closed
            
            # Try to log if possible
            try:
                logger = logging.getLogger(__name__)
                logger.critical(f"Critical error in worker process: {error_msg}\n{stack_trace}")
            except Exception:
                # Last resort: print to stderr
                print(f"Critical error in worker: {error_msg}\n{stack_trace}", file=sys.stderr)
                
        finally:
            # Cleanup
            self.stop()
            
    def stop(self) -> None:
        """Stop the worker process cleanly."""
        # Stop main loop
        self.running = False
        
        # Terminate child processes
        self.cleanup_child_processes()
        
        # Log shutdown if possible
        try:
            logger = logging.getLogger(__name__)
            logger.info("Worker process stopping")
        except Exception:
            pass


def start_worker() -> Dict[str, Any]:
    """
    Start a transcription worker process and return communication queues.
    
    Returns:
        Dictionary with the worker process, task queue, and result queue
    """
    # Create communication queues
    task_queue = mp.Queue()
    result_queue = mp.Queue()
    
    # Create and start worker
    worker = TranscribeWorker(task_queue, result_queue)
    worker.start()
    
    return {
        "worker": worker,
        "task_queue": task_queue,
        "result_queue": result_queue
    }


class WorkerManager:
    """
    Manages worker processes for transcription tasks.
    
    Provides high-level interface for starting, monitoring, and stopping workers,
    as well as handling communication with the worker processes.
    """
    
    def __init__(self, max_workers: int = 1):
        """
        Initialize the worker manager.
        
        Args:
            max_workers: Maximum number of worker processes to run
        """
        self.logger = logging.getLogger(__name__)
        self.max_workers = max_workers
        self.workers = {}
        self.next_task_id = 0
        self.task_callbacks = {}
        self.last_heartbeats = {}
        self.monitor_thread = None
        self.running = False
        
    def start_monitor(self) -> None:
        """Start the worker monitor thread."""
        if self.monitor_thread is not None and self.monitor_thread.is_alive():
            return
            
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
    def _monitor_loop(self) -> None:
        """Monitor workers for results and heartbeats."""
        while self.running:
            current_time = time.time()
            
            # Check for results from all workers
            for worker_id, worker_info in list(self.workers.items()):
                if not worker_info["worker"].is_alive():
                    self.logger.warning(f"Worker {worker_id} has died unexpectedly")
                    self._handle_worker_death(worker_id)
                    continue
                    
                # Check for results
                try:
                    # Non-blocking check for results
                    while True:
                        try:
                            result = worker_info["result_queue"].get_nowait()
                            self._handle_worker_message(worker_id, result)
                        except queue.Empty:
                            break
                except Exception as e:
                    self.logger.error(f"Error checking results from worker {worker_id}: {e}")
                
                # Check for heartbeat timeout
                last_heartbeat = self.last_heartbeats.get(worker_id, 0)
                if current_time - last_heartbeat > 15.0:  # 15 seconds without heartbeat
                    self.logger.warning(f"Worker {worker_id} heartbeat timeout")
                    self._handle_worker_timeout(worker_id)
            
            # Sleep before next check
            time.sleep(0.1)
            
    def _handle_worker_message(self, worker_id: str, message: Dict[str, Any]) -> None:
        """
        Handle a message from a worker.
        
        Args:
            worker_id: Worker ID that sent the message
            message: Message from the worker
        """
        message_type = message.get("type")
        
        if message_type == "heartbeat":
            # Update last heartbeat time
            self.last_heartbeats[worker_id] = time.time()
            self.logger.debug(f"Heartbeat from worker {worker_id}")
            
        elif message_type == "progress":
            # Forward progress event to callback
            task_id = message.get("task_id")
            if task_id in self.task_callbacks:
                callback = self.task_callbacks[task_id].get("progress")
                if callback:
                    try:
                        callback(message.get("event", {}))
                    except Exception as e:
                        self.logger.error(f"Error in progress callback: {e}")
                        
        elif message_type == "result":
            # Task completed, call result callback
            task_id = message.get("task_id")
            if task_id in self.task_callbacks:
                callback = self.task_callbacks[task_id].get("result")
                if callback:
                    try:
                        callback(message)
                    except Exception as e:
                        self.logger.error(f"Error in result callback: {e}")
                
                # Clean up callbacks
                del self.task_callbacks[task_id]
                
        elif message_type == "critical_error":
            # Critical error in worker
            self.logger.error(f"Critical error in worker {worker_id}: {message.get('error')}")
            self.logger.debug(f"Stack trace: {message.get('stack_trace')}")
            
            # Try to restart the worker
            self._restart_worker(worker_id)
            
    def _handle_worker_death(self, worker_id: str) -> None:
        """
        Handle a worker process that has died unexpectedly.
        
        Args:
            worker_id: ID of the dead worker
        """
        # Fail any pending tasks
        for task_id, callbacks in list(self.task_callbacks.items()):
            if callbacks.get("worker_id") == worker_id:
                result_callback = callbacks.get("result")
                if result_callback:
                    try:
                        result_callback({
                            "type": "result",
                            "task_id": task_id,
                            "status": "error",
                            "error": "Worker process died unexpectedly"
                        })
                    except Exception as e:
                        self.logger.error(f"Error in result callback: {e}")
                
                # Clean up callback
                del self.task_callbacks[task_id]
        
        # Clean up worker resources
        try:
            worker_info = self.workers.pop(worker_id)
            worker_info["worker"].terminate()
        except Exception as e:
            self.logger.error(f"Error cleaning up worker {worker_id}: {e}")
            
        # Remove heartbeat record
        if worker_id in self.last_heartbeats:
            del self.last_heartbeats[worker_id]
            
    def _handle_worker_timeout(self, worker_id: str) -> None:
        """
        Handle a worker that has stopped sending heartbeats.
        
        Args:
            worker_id: ID of the timed-out worker
        """
        self.logger.warning(f"Worker {worker_id} has stopped sending heartbeats")
        
        # Check if the worker is still alive
        worker_info = self.workers.get(worker_id)
        if worker_info and worker_info["worker"].is_alive():
            # Try to terminate the worker
            try:
                worker_info["worker"].terminate()
            except Exception as e:
                self.logger.error(f"Error terminating worker {worker_id}: {e}")
                
        # Handle as worker death
        self._handle_worker_death(worker_id)
        
    def _restart_worker(self, worker_id: str) -> None:
        """
        Restart a worker process.
        
        Args:
            worker_id: ID of the worker to restart
        """
        # Clean up old worker
        try:
            if worker_id in self.workers:
                worker_info = self.workers.pop(worker_id)
                if worker_info["worker"].is_alive():
                    worker_info["worker"].terminate()
        except Exception as e:
            self.logger.error(f"Error cleaning up worker {worker_id} for restart: {e}")
            
        # Start a new worker
        try:
            worker_result = start_worker()
            self.workers[worker_id] = worker_result
            self.last_heartbeats[worker_id] = time.time()
            self.logger.info(f"Worker {worker_id} restarted")
        except Exception as e:
            self.logger.error(f"Error restarting worker {worker_id}: {e}")
            
    def start_worker(self) -> str:
        """
        Start a new worker process.
        
        Returns:
            Worker ID
        """
        worker_id = f"worker-{int(time.time())}-{len(self.workers)}"
        
        try:
            worker_result = start_worker()
            self.workers[worker_id] = worker_result
            self.last_heartbeats[worker_id] = time.time()
            self.logger.info(f"Started worker {worker_id}")
            
            # Start monitor if not already running
            self.start_monitor()
            
            return worker_id
        except Exception as e:
            self.logger.error(f"Error starting worker: {e}")
            raise
            
    def stop_worker(self, worker_id: str) -> None:
        """
        Stop a worker process.
        
        Args:
            worker_id: ID of the worker to stop
        """
        if worker_id not in self.workers:
            self.logger.warning(f"Worker {worker_id} not found")
            return
            
        worker_info = self.workers[worker_id]
        
        # Send termination task
        try:
            worker_info["task_queue"].put({"type": "terminate"})
        except Exception as e:
            self.logger.error(f"Error sending termination task to worker {worker_id}: {e}")
            
        # Wait for worker to terminate
        try:
            worker_info["worker"].join(timeout=3.0)
        except Exception as e:
            self.logger.error(f"Error waiting for worker {worker_id} to terminate: {e}")
            
        # Force termination if not exited
        if worker_info["worker"].is_alive():
            try:
                worker_info["worker"].terminate()
                self.logger.warning(f"Forced termination of worker {worker_id}")
            except Exception as e:
                self.logger.error(f"Error forcing termination of worker {worker_id}: {e}")
                
        # Clean up worker
        self.workers.pop(worker_id, None)
        if worker_id in self.last_heartbeats:
            del self.last_heartbeats[worker_id]
            
    def submit_task(self, 
                    worker_id: str, 
                    task: Dict[str, Any],
                    progress_callback: Optional[Callable[[ProgressEvent], None]] = None,
                    result_callback: Optional[Callable[[Dict[str, Any]], None]] = None) -> str:
        """
        Submit a task to a worker.
        
        Args:
            worker_id: ID of the worker to run the task
            task: Task to submit
            progress_callback: Callback for progress updates
            result_callback: Callback for the final result
            
        Returns:
            Task ID
        """
        if worker_id not in self.workers:
            raise ValueError(f"Worker {worker_id} not found")
            
        # Generate a task ID
        task_id = str(self.next_task_id)
        self.next_task_id += 1
        
        # Add task ID to the task
        task["task_id"] = task_id
        
        # Store callbacks
        self.task_callbacks[task_id] = {
            "worker_id": worker_id,
            "progress": progress_callback,
            "result": result_callback
        }
        
        # Submit the task
        worker_info = self.workers[worker_id]
        worker_info["task_queue"].put(task)
        
        return task_id
        
    def stop_all(self) -> None:
        """Stop all worker processes."""
        self.logger.info("Stopping all workers")
        
        # Stop monitor thread
        self.running = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=1.0)
            
        # Stop all workers
        for worker_id in list(self.workers.keys()):
            self.stop_worker(worker_id)
            
        # Clear all task callbacks
        self.task_callbacks.clear()
        
        self.logger.info("All workers stopped")