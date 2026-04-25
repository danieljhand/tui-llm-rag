import time
import os
import functools
from rich.console import Console
from typing import Any

console = Console()
ENABLE_TRACKING = os.getenv("ENABLE_PERFORMANCE_TRACKING", "false").lower() == "true"

def trace_performance(label: str):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                end_time = time.perf_counter()
                duration_ms = (end_time - start_time) * 1000
                if ENABLE_TRACKING:
                    console.print(f"[bold cyan][PERF][/bold cyan] {label}: {int(duration_ms)} ms")
        return wrapper
    return decorator
