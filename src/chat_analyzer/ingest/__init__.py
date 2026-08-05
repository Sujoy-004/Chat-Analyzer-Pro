"""
Ingestion Package

Contains file ingestion, dependency detection, and supported-format helpers
for chat data (txt/json/pdf with optional OCR).
"""

from .ingestion import (
    get_dependency_status,
    get_supported_formats,
    process_uploaded_file,
)

__all__ = [
    'get_dependency_status',
    'get_supported_formats',
    'process_uploaded_file'
]
