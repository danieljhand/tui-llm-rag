"""
Performance tracking utilities for the RAG application.

Provides decorators for measuring and displaying execution time of key operations
like embedding generation and LLM inference.
"""
import time
import os
import functools
from rich.console import Console
from typing import Any

# Rich console for formatted output
console = Console()
# Global flag to enable/disable performance tracking (controlled via env var)
ENABLE_TRACKING = os.getenv("ENABLE_PERFORMANCE_TRACKING", "false").lower() == "true"

def trace_performance(label: str):
    """Decorator factory for tracing function execution time.
    
    Measures elapsed time using high-resolution perf_counter and displays
    the duration in milliseconds if ENABLE_PERFORMANCE_TRACKING is true.
    
    Args:
        label: Descriptive label for the operation being traced
        
    Returns:
        Decorator function that wraps the target function
        
    Example:
        @trace_performance("Database Query")
        def fetch_data():
            # ... operation ...
            
    Note:
        Uses functools.wraps to preserve original function metadata
    """
    def decorator(func):
        @functools.wraps(func)  # Preserve original function's name and docstring
        def wrapper(*args, **kwargs) -> Any:
            # Record start time with high precision
            start_time = time.perf_counter()
            try:
                # Execute the wrapped function
                result = func(*args, **kwargs)
                return result
            finally:
                # Always calculate and display timing, even if function raises
                end_time = time.perf_counter()
                duration_ms = (end_time - start_time) * 1000
                # Only display if tracking is enabled (avoids console clutter)
                if ENABLE_TRACKING:
                    console.print(f"[bold cyan][PERF][/bold cyan] {label}: {int(duration_ms)} ms")
        return wrapper
    return decorator
