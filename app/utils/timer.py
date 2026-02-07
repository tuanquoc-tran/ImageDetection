"""
Performance Timer Utility
Provides simple performance profiling for production code.
"""

import time
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)


class PerformanceTimer:
    """Simple performance timer for profiling inspection cycles."""
    
    def __init__(self, name="Operation"):
        """
        Initialize timer.
        
        Args:
            name: Name of operation being timed
        """
        self.name = name
        self.start_time = None
        self.elapsed_ms = 0.0
    
    def start(self):
        """Start timer."""
        self.start_time = time.perf_counter()
    
    def stop(self):
        """Stop timer and return elapsed time in milliseconds."""
        if self.start_time is None:
            logger.warning(f"Timer '{self.name}' stopped before starting")
            return 0.0
        
        self.elapsed_ms = (time.perf_counter() - self.start_time) * 1000
        return self.elapsed_ms
    
    def get_elapsed(self):
        """Get elapsed time without stopping."""
        if self.start_time is None:
            return 0.0
        return (time.perf_counter() - self.start_time) * 1000
    
    def log_elapsed(self, level=logging.DEBUG):
        """Log elapsed time."""
        if self.elapsed_ms > 0:
            logger.log(level, f"{self.name}: {self.elapsed_ms:.2f} ms")


@contextmanager
def timed_operation(name="Operation", log_level=logging.DEBUG):
    """
    Context manager for timing operations.
    
    Usage:
        with timed_operation("Image Processing"):
            process_image()
    """
    timer = PerformanceTimer(name)
    timer.start()
    try:
        yield timer
    finally:
        timer.stop()
        timer.log_elapsed(log_level)
