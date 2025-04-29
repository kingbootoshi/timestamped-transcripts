#!/usr/bin/env python3
"""
WhisperX GUI Application Entry Point

This script launches the WhisperX GUI application, setting up the
necessary multiprocessing support and handling startup/shutdown.
"""

import os
import sys
import logging
import signal
import time
import multiprocessing as mp
from whisperx_gui.logging_config import setup_logging
from whisperx_gui.gui.app import WhisperXGUI

# Set up signal handlers
def handle_sigterm(signum, frame):
    """Handle SIGTERM by performing a clean shutdown."""
    if 'app' in globals():
        logger.info("Received termination signal, shutting down gracefully")
        app.shutdown()
    sys.exit(0)

if __name__ == "__main__":
    # Set the start method for multiprocessing
    # 'spawn' is safer than 'fork' for GUI applications
    if sys.platform != 'win32':  # Not needed on Windows
        mp.set_start_method('spawn')
    
    # Configure logging based on environment variables
    log_level = os.environ.get("LOG_LEVEL", "INFO")
    setup_logging(log_level=log_level, where="gui")
    
    logger = logging.getLogger(__name__)
    logger.info("Starting WhisperX GUI Application")
    
    # Set up signal handlers
    signal.signal(signal.SIGTERM, handle_sigterm)
    signal.signal(signal.SIGINT, handle_sigterm)
    
    try:
        # Initialize the GUI
        app = WhisperXGUI()
        
        # Run the GUI application
        app.run()
        
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, shutting down")
        if 'app' in locals():
            app.shutdown()
    except Exception as e:
        logger.exception(f"Error running WhisperX GUI: {e}")
        sys.exit(1)
    finally:
        # Ensure clean exit
        logger.info("WhisperX GUI Application exited")
        # Force any remaining worker processes to terminate
        time.sleep(0.5)  # Give them a moment to clean up
        sys.exit(0)