# src/ingest/ingestion.py
"""
Complete ingestion module for Chat-Analyzer-Pro.
Production-ready with robust error handling and graceful dependency fallbacks.

Public API:
    process_uploaded_file(uploaded_file) -> (messages: List[dict], media_ocr: List[dict])
    parse_whatsapp_text(text) -> List[dict]
    parse_json_chat(bytes_or_str) -> List[dict] | None
    
Features:
- Handles all major file formats (TXT, JSON, ZIP, images, PDFs, media)
- OCR text extraction from images when available
- PDF text extraction with OCR fallback
- Graceful handling of missing dependencies
- Comprehensive error handling and logging
"""

import json
import logging
import os
import re
import uuid
import zipfile
from datetime import datetime
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

# Set up logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Dependency availability flags
DEPENDENCIES = {
    'PIL': False,
    'pytesseract': False,
    'pdfplumber': False,
    'pdf2image': False
}

# Try to import optional dependencies
try:
    from PIL import Image, ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    DEPENDENCIES['PIL'] = True
except ImportError:
    logger.info("PIL not available - image processing disabled")

try:
    import pytesseract
    DEPENDENCIES['pytesseract'] = True
except ImportError:
    logger.info("pytesseract not available - OCR disabled")

try:
    import pdfplumber
    DEPENDENCIES['pdfplumber'] = True
except ImportError:
    logger.info("pdfplumber not available - PDF text extraction limited")

try:
    from pdf2image import convert_from_bytes
    DEPENDENCIES['pdf2image'] = True
except ImportError:
    logger.info("pdf2image not available - PDF OCR fallback disabled")

# Constants
WHATSAPP_DATE_PATTERNS = [
    "%d/%m/%y, %I:%M %p",
    "%d/%m/%Y, %I:%M %p",
    "%d/%m/%y, %H:%M",
    "%d/%m/%Y, %H:%M",
    "%m/%d/%y, %I:%M %p",
    "%m/%d/%Y, %I:%M %p",
    "%d/%m/%y %I:%M %p",
    "%d/%m/%Y %I:%M %p",
]

WAP_MESSAGE_RE = re.compile(
    r"""^
    (\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}),?\s*
    (\d{1,2}:\d{2}(?::\d{2})?(?:\s?[APap][Mm])?)\s*-\s*
    ([^:]+):\s*
    (.*)
    $""",
    re.VERBOSE,
)

# Supported file types
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif', '.tiff', '.tif'}
AUDIO_EXTENSIONS = {'.opus', '.mp3', '.wav', '.m4a', '.aac', '.ogg', '.flac'}
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.3gp'}
MEDIA_EXTENSIONS = AUDIO_EXTENSIONS | VIDEO_EXTENSIONS


def get_dependency_status() -> Dict[str, bool]:
    """Get status of all optional dependencies."""
    return DEPENDENCIES.copy()


def try_parse_datetime(date_str: str) -> Tuple[str, str]:
    """
    Parse WhatsApp-like date+time string into ISO date and HH:MM.
    Returns (date_iso, time_hm). On failure, returns (original, '').
    """
    s = date_str.strip()
    candidates = [s, s.replace(" - ", ", "), s.replace(" ,", ",")]
    
    for candidate in candidates:
        for fmt in WHATSAPP_DATE_PATTERNS:
            try:
                dt = datetime.strptime(candidate, fmt)
                return dt.strftime("%Y-%m-%d"), dt.strftime("%H:%M")
            except ValueError:
                continue
    
    # Regex fallback
    match = re.match(
        r"^(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})[, ]+\s*(\d{1,2}:\d{2}(?:[:\d{2}]?)\s*(?:AM|PM|am|pm)?)", 
        s
    )
    if match:
        try:
            part = f"{match.group(1)}, {match.group(2)}"
            for fmt in WHATSAPP_DATE_PATTERNS:
                try:
                    dt = datetime.strptime(part, fmt)
                    return dt.strftime("%Y-%m-%d"), dt.strftime("%H:%M")
                except ValueError:
                    continue
        except Exception:
            pass
    
    return s, ""


def parse_whatsapp_text(text: str) -> List[Dict[str, Any]]:
    """
    Parse WhatsApp .txt export into structured message list.
    Handles multiline messages and various date formats.
    """
    messages = []
    current_message = None
    
    for line in text.splitlines():
        line = line.rstrip()
        if not line:
            if current_message:
                current_message["text"] += "\n"
            continue
            
        match = WAP_MESSAGE_RE.match(line)
        if match:
            # Save previous message
            if current_message:
                messages.append(current_message)
            
            # Start new message
            date_raw, time_raw, author, message_text = match.groups()
            current_message = {
                "date_raw": date_raw.strip(),
                "time_raw": time_raw.strip(),
                "author": author.strip(),
                "text": message_text.strip(),
                "raw_line": line,
                "source_hint": "whatsapp_txt",
            }
        else:
            # Continuation line or orphan
            if current_message:
                current_message["text"] += "\n" + line.strip()
            else:
                # Orphan line
                messages.append({
                    "date_raw": "",
                    "time_raw": "",
                    "author": "unknown",
                    "text": line.strip(),
                    "raw_line": line,
                    "source_hint": "orphan_text",
                })
    
    # Don't forget the last message
    if current_message:
        messages.append(current_message)
    
    return messages


def parse_json_chat(data_input: Any) -> Optional[List[Dict[str, Any]]]:
    """
    Parse JSON chat exports (Telegram, etc.).
    Handles bytes, strings, and file-like objects.
    """
    try:
        if isinstance(data_input, (bytes, bytearray)):
            data = json.loads(data_input.decode("utf-8", errors="ignore"))
        elif isinstance(data_input, str):
            data = json.loads(data_input)
        elif hasattr(data_input, "read"):
            raw = data_input.read()
            data = json.loads(raw.decode("utf-8", errors="ignore"))
        else:
            return None
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        logger.warning(f"JSON parsing failed: {e}")
        return None
    
    # Extract messages from various JSON structures
    if isinstance(data, dict):
        if "messages" in data and isinstance(data["messages"], list):
            return data["messages"]
        if "chats" in data and isinstance(data["chats"], list):
            # Flatten multiple chats
            messages = []
            for chat in data["chats"]:
                if isinstance(chat, dict) and "messages" in chat:
                    messages.extend(chat["messages"])
            return messages if messages else None
    elif isinstance(data, list):
        return data
    
    return None


def ocr_image_bytes(image_bytes: bytes, lang: str = "eng") -> str:
    """
    Extract text from image bytes using OCR.
    Returns empty string if OCR unavailable or fails.
    """
    if not (DEPENDENCIES['PIL'] and DEPENDENCIES['pytesseract']):
        logger.warning("OCR dependencies not available")
        return ""
    
    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        text = pytesseract.image_to_string(image, lang=lang)
        return text.strip()
    except Exception as e:
        logger.error(f"OCR failed: {e}")
        return ""


def extract_text_from_pdf(pdf_bytes: bytes, ocr_lang: str = "eng") -> str:
    """
    Extract text from PDF with OCR fallback for image-based PDFs.
    """
    if not DEPENDENCIES['pdfplumber']:
        logger.warning("pdfplumber not available for PDF processing")
        return ""
    
    pages_text = []
    
    try:
        with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
            for i, page in enumerate(pdf.pages, 1):
                try:
                    text = page.extract_text()
                    if text and text.strip():
                        pages_text.append(f"[page:{i}]\n{text}")
                    else:
                        # Try OCR fallback
                        ocr_text = _ocr_pdf_page(pdf_bytes, i, ocr_lang)
                        pages_text.append(f"[page:{i}][ocr]\n{ocr_text}")
                except Exception as e:
                    logger.warning(f"Failed to process PDF page {i}: {e}")
                    ocr_text = _ocr_pdf_page(pdf_bytes, i, ocr_lang)
                    pages_text.append(f"[page:{i}][ocr]\n{ocr_text}")
    except Exception as e:
        logger.error(f"PDF processing failed: {e}")
        return ""
    
    return "\n\n".join(pages_text)


def _ocr_pdf_page(pdf_bytes: bytes, page_num: int, lang: str) -> str:
    """OCR fallback for PDF pages."""
    if not (DEPENDENCIES['pdf2image'] and DEPENDENCIES['PIL'] and DEPENDENCIES['pytesseract']):
        return ""
    
    try:
        images = convert_from_bytes(pdf_bytes, first_page=page_num, last_page=page_num)
        if images:
            return pytesseract.image_to_string(images[0], lang=lang).strip()
    except Exception as e:
        logger.warning(f"PDF OCR failed for page {page_num}: {e}")
    
    return ""


def extract_media_metadata(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    """Extract basic metadata from media files."""
    file_size = len(file_bytes)
    file_ext = filename.lower().split('.')[-1]
    
    # Classify media type
    if f'.{file_ext}' in AUDIO_EXTENSIONS:
        media_type = "audio"
    elif f'.{file_ext}' in VIDEO_EXTENSIONS:
        media_type = "video"
    else:
        media_type = "unknown"
    
    return {
        "filename": filename,
        "file_type": file_ext,
        "media_type": media_type,
        "file_size_bytes": file_size,
        "file_size_mb": round(file_size / (1024 * 1024), 2),
        "file_size_human": _format_file_size(file_size)
    }


def _format_file_size(size_bytes: int) -> str:
    """Convert bytes to human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def normalize_message(raw_msg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize raw message to standard schema:
    {
        "uid": str,
        "date": "YYYY-MM-DD",
        "time": "HH:MM", 
        "author": str,
        "text": str,
        "source": str,
        "media": [],
        "meta": {}
    }
    """
    # Generate unique ID
    uid = raw_msg.get("uid") or raw_msg.get("id") or str(uuid.uuid4())
    
    # Determine source
    source = raw_msg.get("source") or raw_msg.get("source_hint") or "unknown"
    
    # Parse date and time
    date_field, time_field = "", ""
    if raw_msg.get("date"):
        date_field = raw_msg["date"]
    elif raw_msg.get("date_raw"):
        date_field, time_field = try_parse_datetime(
            f"{raw_msg.get('date_raw', '')}, {raw_msg.get('time_raw', '')}".strip(", ")
        )
    elif raw_msg.get("datetime"):
        try:
            dt = datetime.fromisoformat(str(raw_msg["datetime"]).replace('Z', '+00:00'))
            date_field = dt.strftime("%Y-%m-%d")
            time_field = dt.strftime("%H:%M")
        except (ValueError, TypeError):
            date_field = str(raw_msg["datetime"])
    
    # Extract other fields
    author = raw_msg.get("author") or raw_msg.get("sender") or raw_msg.get("from") or "unknown"
    text = raw_msg.get("text") or raw_msg.get("message") or raw_msg.get("body") or ""
    media = raw_msg.get("media") or raw_msg.get("attachments") or []
    meta = raw_msg.get("meta") or {}
    
    # Add preview to metadata for debugging
    if "raw_line" in raw_msg:
        meta["_raw_preview"] = raw_msg["raw_line"][:500]
    elif text:
        meta["_raw_preview"] = text[:200]
    
    return {
        "uid": uid,
        "date": date_field,
        "time": time_field,
        "author": author,
        "text": text,
        "source": source,
        "media": media,
        "meta": meta,
    }


def _read_file_content(uploaded_file: Any) -> Tuple[str, bytes]:
    """Extract filename and content from uploaded file object."""
    if hasattr(uploaded_file, "name") and hasattr(uploaded_file, "read"):
        # Streamlit UploadedFile-like object
        return uploaded_file.name, uploaded_file.read()
    elif isinstance(uploaded_file, (tuple, list)) and len(uploaded_file) == 2:
        # (filename, bytes) tuple
        return uploaded_file[0], uploaded_file[1]
    elif isinstance(uploaded_file, str) and os.path.exists(uploaded_file):
        # File path
        with open(uploaded_file, "rb") as f:
            return os.path.basename(uploaded_file), f.read()
    else:
        raise ValueError("Unsupported uploaded_file type")


def process_uploaded_file(uploaded_file: Any) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Main ingestion function. Processes any supported file type.
    
    Args:
        uploaded_file: File object with .name and .read() methods, or (name, bytes) tuple
        
    Returns:
        Tuple of (normalized_messages, media_analysis_results)
    """
    try:
        filename, content = _read_file_content(uploaded_file)
    except Exception as e:
        logger.error(f"Failed to read file: {e}")
        return [], [{"file": "unknown", "note": f"File reading error: {e}"}]
    
    lower_filename = filename.lower()
    raw_messages = []
    media_results = []
    
    try:
        # ZIP files - extract and process contents
        if lower_filename.endswith('.zip'):
            raw_messages, media_results = _process_zip_file(content, filename)
        
        # Text files - WhatsApp exports
        elif lower_filename.endswith('.txt'):
            raw_messages = _process_text_file(content, filename)
        
        # JSON files - Telegram/other chat exports  
        elif lower_filename.endswith('.json'):
            raw_messages = _process_json_file(content, filename)
        
        # Image files - OCR extraction
        elif any(lower_filename.endswith(ext) for ext in IMAGE_EXTENSIONS):
            raw_messages, media_results = _process_image_file(content, filename)
        
        # PDF files - text extraction
        elif lower_filename.endswith('.pdf'):
            raw_messages, media_results = _process_pdf_file(content, filename)
        
        # Media files - metadata extraction
        elif any(lower_filename.endswith(ext) for ext in MEDIA_EXTENSIONS):
            media_results = _process_media_file(content, filename)
        
        # Unknown file type - try as text
        else:
            raw_messages = _process_unknown_file(content, filename)
    
    except Exception as e:
        logger.error(f"Processing failed for {filename}: {e}")
        media_results.append({"file": filename, "note": f"Processing error: {e}"})
    
    # Normalize all messages
    normalized_messages = []
    for raw_msg in raw_messages:
        try:
            normalized = normalize_message(raw_msg)
            normalized_messages.append(normalized)
        except Exception as e:
            logger.warning(f"Message normalization failed: {e}")
            # Fallback normalization
            normalized_messages.append({
                "uid": str(uuid.uuid4()),
                "date": "",
                "time": "",
                "author": raw_msg.get("author", "unknown"),
                "text": str(raw_msg.get("text", ""))[:1000],
                "source": raw_msg.get("source_hint", "unknown"),
                "media": [],
                "meta": {"_normalization_error": True}
            })
    
    return normalized_messages, media_results


def _process_zip_file(content: bytes, filename: str) -> Tuple[List[Dict], List[Dict]]:
    """Process ZIP archive contents."""
    messages, media = [], []
    
    try:
        with zipfile.ZipFile(BytesIO(content)) as zf:
            for member in zf.namelist():
                if member.endswith('/'):
                    continue
                    
                try:
                    member_content = zf.read(member)
                    member_name = os.path.basename(member)
                    member_lower = member_name.lower()
                    
                    # Process based on file type
                    if member_lower.endswith('.txt'):
                        member_messages = _process_text_file(member_content, member_name)
                        messages.extend(member_messages)
                    
                    elif member_lower.endswith('.json'):
                        member_messages = _process_json_file(member_content, member_name)
                        messages.extend(member_messages)
                    
                    elif any(member_lower.endswith(ext) for ext in IMAGE_EXTENSIONS):
                        member_messages, member_media = _process_image_file(member_content, member_name)
                        messages.extend(member_messages)
                        media.extend(member_media)
                    
                    elif member_lower.endswith('.pdf'):
                        member_messages, member_media = _process_pdf_file(member_content, member_name)
                        messages.extend(member_messages)
                        media.extend(member_media)
                    
                    elif any(member_lower.endswith(ext) for ext in MEDIA_EXTENSIONS):
                        member_media = _process_media_file(member_content, member_name)
                        media.extend(member_media)
                    
                    else:
                        media.append({"file": member_name, "note": "Unsupported file type in ZIP"})
                
                except Exception as e:
                    logger.warning(f"Failed to process ZIP member {member}: {e}")
                    media.append({"file": member, "note": f"Processing error: {e}"})
    
    except zipfile.BadZipFile:
        media.append({"file": filename, "note": "Invalid or corrupted ZIP file"})
    
    return messages, media


def _process_text_file(content: bytes, filename: str) -> List[Dict]:
    """Process text file as WhatsApp export."""
    try:
        text = content.decode('utf-8', errors='ignore')
        messages = parse_whatsapp_text(text)
        for msg in messages:
            msg["source_hint"] = "whatsapp_txt"
        return messages
    except Exception as e:
        logger.error(f"Text file processing failed: {e}")
        return []


def _process_json_file(content: bytes, filename: str) -> List[Dict]:
    """Process JSON file as chat export."""
    messages = parse_json_chat(content)
    if messages:
        for msg in messages:
            if isinstance(msg, dict):
                msg["source_hint"] = "json"
        return [msg for msg in messages if isinstance(msg, dict)]
    return []


def _process_image_file(content: bytes, filename: str) -> Tuple[List[Dict], List[Dict]]:
    """Process image file with OCR."""
    ocr_text = ocr_image_bytes(content)
    media_result = {"file": filename, "ocr": ocr_text}
    
    messages = []
    if ocr_text.strip():
        messages.append({
            "date_raw": "",
            "time_raw": "",
            "author": "unknown",
            "text": ocr_text,
            "raw_line": "",
            "source_hint": "ocr_image",
        })
    
    return messages, [media_result]


def _process_pdf_file(content: bytes, filename: str) -> Tuple[List[Dict], List[Dict]]:
    """Process PDF file with text extraction."""
    pdf_text = extract_text_from_pdf(content)
    media_result = {"file": filename, "extracted_text": pdf_text}
    
    messages = []
    if pdf_text.strip():
        messages.append({
            "date_raw": "",
            "time_raw": "",
            "author": "unknown", 
            "text": pdf_text,
            "raw_line": "",
            "source_hint": "pdf",
            "meta": {"filename": filename}
        })
    
    return messages, [media_result]


def _process_media_file(content: bytes, filename: str) -> List[Dict]:
    """Process media file - extract metadata only."""
    metadata = extract_media_metadata(content, filename)
    return [{
        "file": filename,
        "note": f"Media file: {metadata['media_type']} ({metadata['file_size_human']})",
        "metadata": metadata
    }]


def _process_unknown_file(content: bytes, filename: str) -> List[Dict]:
    """Try to process unknown file as text."""
    try:
        text = content.decode('utf-8', errors='ignore')
        if text.strip():
            # Try WhatsApp parsing first
            messages = parse_whatsapp_text(text)
            if messages:
                for msg in messages:
                    msg["source_hint"] = "whatsapp_txt_generic"
                return messages
            
            # Fallback: treat as single text block
            return [{
                "date_raw": "",
                "time_raw": "",
                "author": "unknown",
                "text": text.strip(),
                "raw_line": "",
                "source_hint": "generic_text",
            }]
    except UnicodeDecodeError:
        pass
    
    return []


# Utility function for debugging
def get_supported_formats() -> Dict[str, List[str]]:
    """Return dict of supported file formats by category."""
    return {
        "chat_exports": ["txt", "json"],
        "archives": ["zip"],
        "images": list(ext.lstrip('.') for ext in IMAGE_EXTENSIONS),
        "documents": ["pdf"],
        "audio": list(ext.lstrip('.') for ext in AUDIO_EXTENSIONS),
        "video": list(ext.lstrip('.') for ext in VIDEO_EXTENSIONS),
    }


if __name__ == "__main__":
    # Basic test/demo
    import sys
    from pprint import pprint
    
    print("Chat Analyzer Pro - Ingestion Module")
    print("Dependencies:", get_dependency_status())
    print("Supported formats:", get_supported_formats())
    
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
        try:
            messages, media = process_uploaded_file(filepath)
            print(f"\nProcessed {filepath}:")
            print(f"Messages: {len(messages)}")
            print(f"Media items: {len(media)}")
            
            if messages:
                print("\nFirst few messages:")
                pprint(messages[:3])
            
            if media:
                print("\nMedia analysis:")
                pprint(media[:3])
                
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
