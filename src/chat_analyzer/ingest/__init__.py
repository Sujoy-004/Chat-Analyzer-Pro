"""
Ingestion Package

Contains file ingestion, dependency detection, and supported-format helpers
for chat data (txt/json/pdf with optional OCR).
"""

from .ingestion import (
    process_uploaded_file,
    get_dependency_status,
    get_supported_formats
)

__all__ = [
    'process_uploaded_file',
    'get_dependency_status',
    'get_supported_formats'
]
