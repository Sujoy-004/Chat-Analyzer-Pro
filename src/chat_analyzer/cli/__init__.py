"""CLI Package

Command-line interface for chat-analyzer (D-01 console-script target).
Importing this package pulls only typer and the standard library.
"""

from .main import app

__all__ = ["app"]
