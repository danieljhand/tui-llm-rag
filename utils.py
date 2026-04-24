import time
import os
import functools
from rich.console import Console
from typing import Any, Tuple, Union

console = Console()
ENABLE_TRACKING = os.getenv("ENABLE_PERFORMANCE_TRACKING", "decode").lower() == "true"

def trace_performance(label: str):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Union[Any, Tuple[Any, float]]:
            if not ENABLE_TRACKING:
                return func(*args, **kwargs)
            
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            
            duration_ms = (end_time - start_time) * 1000
            console.print(f"[bold cyan][PERF][/bold cyan] {label}: {int(duration_ms)} ms")
            
            return result, duration_ms
        return wrapper
    return decorator
