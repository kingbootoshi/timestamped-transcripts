#!/usr/bin/env python3
"""
WhisperX GUI Application Entry Point

This script launches the WhisperX GUI application.
"""

import os
import sys
import logging
from whisperx_gui.logging_config import setup_logging
from whisperx_gui.gui.app import WhisperXGUI

if __name__ == "__main__":
    # Configure logging based on environment variables
    log_level = os.environ.get("LOG_LEVEL", "INFO")
    setup_logging(log_level=log_level, where="gui")
    
    logger = logging.getLogger(__name__)
    logger.info("Starting WhisperX GUI Application")
    
    try:
        # Initialize and run the GUI
        app = WhisperXGUI()
        app.run()
    except Exception as e:
        logger.exception(f"Error running WhisperX GUI: {e}")
        sys.exit(1)