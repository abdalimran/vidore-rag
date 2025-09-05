"""Timer utility for measuring execution time."""

import time
import functools
from contextlib import ContextDecorator
from typing import Callable, Any


class Timer(ContextDecorator):
    """
    A versatile timer that can be used as:
    - A decorator to time functions.
    - A context manager to time code blocks.

    Example:
        @Timer()
        def slow_func():
            time.sleep(2)

        with Timer("Block Timer"):
            time.sleep(1.5)
    """

    def __init__(self, name: str | None = None, logger: Callable[[str], Any] = print):
        self.name = name
        self.logger = logger
        self.start_time: float | None = None
        self.elapsed: float | None = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if self.start_time is not None:
            self.elapsed = time.perf_counter() - self.start_time
            label = f"[{self.name}] " if self.name else ""
            self.logger(f"{label}Elapsed time: {self.elapsed:.6f} seconds")
        return False  # Don't suppress exceptions

    def __call__(self, func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        return wrapper
