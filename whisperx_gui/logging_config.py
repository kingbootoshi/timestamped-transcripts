import logging
import json
import os
import queue
from logging.handlers import QueueHandler, QueueListener, RotatingFileHandler
from typing import Optional, Literal

def setup_logging(
    log_level: str = "INFO",
    where: Literal["cli", "gui", "worker"] = "cli"
) -> Optional[queue.SimpleQueue]:
    """
    Set up structured logging based on the environment.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        where: Context of execution ('cli', 'gui', or 'worker')
        
    Returns:
        SimpleQueue if where='worker', otherwise None
    """
    # Convert string level to actual level
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Format for console output
    fmt = "%(asctime)s %(levelname)s %(name)s %(message)s"
    
    # Configure root logger
    root = logging.getLogger()
    root.setLevel(numeric_level)
    
    # Remove existing handlers
    while root.handlers:
        root.handlers.pop()

    # Worker processes use a queue to forward logs to the parent process
    if where == "worker":
        q = queue.SimpleQueue()
        root.addHandler(QueueHandler(q))
        # Listener started by parent
        return q

    # Human-readable console output
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(fmt))
    root.addHandler(console)

    # JSON-structured file logging
    file_h = RotatingFileHandler(
        "whisperx.log", 
        maxBytes=5_000_000, 
        backupCount=3
    )
    file_h.setFormatter(
        logging.Formatter(
            json.dumps({
                "time": "%(asctime)s",
                "lvl": "%(levelname)s",
                "src": "%(name)s",
                "msg": "%(message)s"
            })
        )
    )
    root.addHandler(file_h)
    
    return None