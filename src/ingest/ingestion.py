# src/ingest/ingestion.py
"""
Central ingestion module for Chat-Analyzer-Pro.

Public API:
    process_uploaded_file(uploaded_file) -> (messages: List[dict], media_ocr: List[dict])
    parse_whatsapp_text(text) -> List[dict]
    parse_json_chat(bytes_or_str) -> List[dict] | None
    ocr_image_bytes(bts) -> str
    extract_text_from_pdf(bts) -> str
    normalize_msg(raw: dict) -> dict

Notes:
- Requires external system packages: tesseract-ocr, poppler-utils (for pdf->image).
- Python packages: pillow, pytesseract, pdfplumber, pdf2image
  Install: pip install pillow pytesseract pdfplumber pdf2image
- If tesseract is not on PATH, set pytesseract.pytesseract.tesseract_cmd
  before calling OCR functions.
"""

from typing import Tuple, List, Dict, Optional, Any, BinaryIO
from io import BytesIO
import zipfile
import json
import re
import uuid
import logging
import os
from datetime import datetime

# Optional heavy deps; import only at top so import errors are visible early.
try:
    from PIL import Image, ImageFile
    PIL_AVAILABLE = True
    ImageFile.LOAD_TRUNCATED_IMAGES = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

try:
    from pdf2image import convert_from_bytes
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# -------------------------
# Helpers: date parsing
# -------------------------
WHATSAPP_DATE_PATTERNS = [
    "%d/%m/%y, %I:%M %p",     # e.g., 1/2/20, 9:41 PM
    "%d/%m/%Y, %I:%M %p",    # 01/02/2020, 9:41 PM
    "%d/%m/%y, %H:%M",       # 1/2/20, 21:41
    "%d/%m/%Y, %H:%M",
    "%m/%d/%y, %I:%M %p",    # US style exported sometimes
    "%m/%d/%Y, %I:%M %p",
    # fallback patterns without comma
    "%d/%m/%y %I:%M %p",
    "%d/%m/%Y %I:%M %p",
]


def try_parse_datetime(date_str: str) -> Tuple[str, str]:
    """
    Attempt to parse WhatsApp-like date+time string into ISO date and HH:MM.
    Returns (date_iso, time_hm). On failure, returns (original, '').
    """
    s = date_str.strip()
    # Try common separators: ", " or " - "
    candidates = [s, s.replace(" - ", ", "), s.replace(" ,", ",")]
    for cand in candidates:
        for fmt in WHATSAPP_DATE_PATTERNS:
            try:
                dt = datetime.strptime(cand, fmt)
                return dt.strftime("%Y-%m-%d"), dt.strftime("%H:%M")
            except Exception:
                continue
    # last ditch: try splitting into date and time with regex of numeric parts
    m = re.match(r"^(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})[, ]+\s*(\d{1,2}:\d{2}(?:[:\d{2}]?)\s*(?:AM|PM|am|pm)?)", s)
    if m:
        try:
            part = m.group(1) + ", " + m.group(2)
            for fmt in WHATSAPP_DATE_PATTERNS:
                try:
                    dt = datetime.strptime(part, fmt)
                    return dt.strftime("%Y-%m-%d"), dt.strftime("%H:%M")
                except Exception:
                    continue
        except Exception:
            pass
    # Fail: return raw date in date field for traceability
    return s, ""


# -------------------------
# Parsers
# -------------------------
WAP_MESSAGE_RE = re.compile(
    r"""^
    (\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}),?\s*    # date part (group 1)
    (\d{1,2}:\d{2}(?:\s?[APap][Mm])?)\s*-\s*   # time part (group 2)
    ([^:]+):\s*                                # author (group 3)
    (.*)                                       # message (group 4)
    $""",
    re.VERBOSE,
)


def parse_whatsapp_text(text: str) -> List[Dict[str, Any]]:
    """
    Parse exported WhatsApp .txt content into a list of raw message dicts.
    Handles multi-line messages by detecting lines that start new messages.
    Returns list of dicts with keys: date_raw, time_raw, author, text, raw_line.
    """
    messages: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None

    # Normalize line endings
    lines = text.splitlines()
    for line in lines:
        line = line.rstrip("\r\n")
        if not line:
            # preserve empty lines as continuation if in current message
            if current:
                current["text"] += "\n"
            continue
        m = WAP_MESSAGE_RE.match(line)
        if m:
            # push previous
            if current:
                messages.append(current)
            date_raw = m.group(1).strip()
            time_raw = m.group(2).strip()
            author = m.group(3).strip()
            msg = m.group(4).strip()
            current = {
                "date_raw": date_raw,
                "time_raw": time_raw,
                "author": author,
                "text": msg,
                "raw_line": line,
                "source_hint": "whatsapp_txt",
            }
        else:
            # continuation line: append to current message text (if exists), else ignore
            if current:
                current["text"] += "\n" + line.strip()
            else:
                # orphan line: store as unknown source
                messages.append({
                    "date_raw": "",
                    "time_raw": "",
                    "author": "unknown",
                    "text": line.strip(),
                    "raw_line": line,
                    "source_hint": "orphan_text",
                })
    if current:
        messages.append(current)
    return messages


def parse_json_chat(bytes_or_str: Any) -> Optional[List[Dict[str, Any]]]:
    """
    Try to parse JSON chat exports. Accepts bytes or str.
    Heuristics:
      - If top-level dict with key 'messages' -> return that list
      - If top-level list -> return it
      - Otherwise return None
    """
    try:
        if isinstance(bytes_or_str, (bytes, bytearray)):
            data = json.loads(bytes_or_str.decode("utf-8", errors="ignore"))
        elif isinstance(bytes_or_str, str):
            data = json.loads(bytes_or_str)
        else:
            # maybe file-like: read
            if hasattr(bytes_or_str, "read"):
                raw = bytes_or_str.read()
                data = json.loads(raw.decode("utf-8", errors="ignore"))
            else:
                return None
    except Exception:
        return None

    if isinstance(data, dict):
        if "messages" in data and isinstance(data["messages"], list):
            return data["messages"]
        # some JSON exports have 'chats' -> list of chat objects
        if "chats" in data and isinstance(data["chats"], list):
            # try flattening
            msgs = []
            for c in data["chats"]:
                if isinstance(c, dict) and "messages" in c and isinstance(c["messages"], list):
                    msgs.extend(c["messages"])
            if msgs:
                return msgs
    elif isinstance(data, list):
        return data
    return None


# -------------------------
# OCR & PDF
# -------------------------
def ocr_image_bytes(bts: bytes, lang: str = "eng") -> str:
    """
    Run pytesseract OCR on image bytes and return extracted text (str).
    """
    if not PIL_AVAILABLE or not TESSERACT_AVAILABLE:
        logger.warning("PIL or pytesseract not available, skipping OCR")
        return ""
    
    try:
        img = Image.open(BytesIO(bts)).convert("RGB")
        text = pytesseract.image_to_string(img, lang=lang)
        return text or ""
    except Exception as exc:
        logger.exception("ocr_image_bytes failed: %s", exc)
        return ""


def extract_text_from_pdf(bts: bytes, ocr_lang: str = "eng") -> str:
    """
    Extract text from PDF bytes.
    First attempt: pdfplumber text extraction per page.
    If a page has empty text, convert that page to image and OCR it.
    Returns a string with page separators.
    """
    if not PDFPLUMBER_AVAILABLE:
        logger.warning("pdfplumber not available, skipping PDF text extraction")
        return ""
    
    pages_text: List[str] = []
    try:
        with pdfplumber.open(BytesIO(bts)) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                try:
                    txt = page.extract_text() or ""
                    if txt.strip():
                        pages_text.append(f"[page:{i}]\n" + txt)
                    else:
                        # fallback to OCR on the specific page
                        if PDF2IMAGE_AVAILABLE and PIL_AVAILABLE and TESSERACT_AVAILABLE:
                            try:
                                images = convert_from_bytes(bts, first_page=i, last_page=i)
                                if images:
                                    ocr_text = pytesseract.image_to_string(images[0], lang=ocr_lang)
                                    pages_text.append(f"[page:{i}][ocr]\n" + (ocr_text or ""))
                                else:
                                    pages_text.append(f"[page:{i}]\n")
                            except Exception:
                                pages_text.append(f"[page:{i}]\n")
                        else:
                            pages_text.append(f"[page:{i}]\n")
                except Exception:
                    logger.exception("Failed page extract; falling back to OCR for page %s", i)
                    if PDF2IMAGE_AVAILABLE and PIL_AVAILABLE and TESSERACT_AVAILABLE:
                        try:
                            images = convert_from_bytes(bts, first_page=i, last_page=i)
                            if images:
                                ocr_text = pytesseract.image_to_string(images[0], lang=ocr_lang)
                                pages_text.append(f"[page:{i}][ocr]\n" + (ocr_text or ""))
                            else:
                                pages_text.append(f"[page:{i}]\n")
                        except Exception:
                            pages_text.append(f"[page:{i}]\n")
                    else:
                        pages_text.append(f"[page:{i}]\n")
    except Exception:
        # total fallback: convert all pages to images and OCR
        logger.exception("pdfplumber failed; converting all pages to images for OCR")
        if PDF2IMAGE_AVAILABLE and PIL_AVAILABLE and TESSERACT_AVAILABLE:
            try:
                images = convert_from_bytes(bts)
                for i, img in enumerate(images, start=1):
                    ocr_text = pytesseract.image_to_string(img, lang=ocr_lang)
                    pages_text.append(f"[page:{i}][ocr]\n" + (ocr_text or ""))
            except Exception:
                logger.exception("complete PDF -> image OCR fallback failed")
                return ""
        else:
            logger.warning("PDF2IMAGE, PIL, or pytesseract not available for PDF OCR fallback")
            return ""
    return "\n\n".join(pages_text)


def extract_media_metadata(bts: bytes, filename: str) -> Dict[str, Any]:
    """
    Extract basic metadata from media files.
    """
    file_size = len(bts)
    file_ext = filename.lower().split('.')[-1]
    
    metadata = {
        "filename": filename,
        "file_type": file_ext,
        "file_size_bytes": file_size,
        "file_size_mb": round(file_size / (1024*1024), 2),
        "file_size_human": _human_readable_size(file_size)
    }
    
    # Basic file type classification
    audio_formats = ['opus', 'mp3', 'wav', 'm4a', 'aac', 'ogg', 'flac']
    video_formats = ['mp4', 'avi', 'mov', 'mkv', 'webm', 'flv', '3gp']
    
    if file_ext in audio_formats:
        metadata["media_type"] = "audio"
    elif file_ext in video_formats:
        metadata["media_type"] = "video"
    else:
        metadata["media_type"] = "unknown"
    
    return metadata


def _human_readable_size(size_bytes: int) -> str:
    """Convert bytes to human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


# -------------------------
# Normalization & utilities
# -------------------------
def normalize_msg(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a raw parsed message to canonical schema:
    {
      "uid": str,
      "date": "YYYY-MM-DD" or raw date string,
      "time": "HH:MM" or "",
      "author": str,
      "text": str,
      "source": "whatsapp|telegram|json|ocr|pdf|unknown",
      "media": [ ... ],
      "meta": { ... }
    }
    """
    uid = raw.get("uid") or raw.get("id") or str(uuid.uuid4())
    src_hint = raw.get("source") or raw.get("source_hint") or raw.get("source_type") or "unknown"
    # normalize whatsapp-style raw date/time if present
    date_field = ""
    time_field = ""
    if raw.get("date"):
        date_field = raw.get("date")
    elif raw.get("date_raw"):
        date_field, time_field = try_parse_datetime(f"{raw.get('date_raw')}, {raw.get('time_raw')}".strip(", "))
        # try_parse returns raw string in date_field if fails
        if time_field == "":
            # attempt to parse time if still available
            _, t = try_parse_datetime(raw.get("date_raw"))
            time_field = t
    else:
        # attempt date-time from a single combined field
        if raw.get("datetime"):
            try:
                dt = datetime.fromisoformat(raw.get("datetime"))
                date_field = dt.strftime("%Y-%m-%d")
                time_field = dt.strftime("%H:%M")
            except Exception:
                date_field = raw.get("datetime")
    author = raw.get("author") or raw.get("sender") or raw.get("from") or "unknown"
    text = raw.get("text") or raw.get("message") or raw.get("body") or ""
    media = raw.get("media") or raw.get("attachments") or []
    meta = raw.get("meta") or {}

    normalized = {
        "uid": uid,
        "date": date_field,
        "time": time_field,
        "author": author,
        "text": text,
        "source": src_hint,
        "media": media,
        "meta": meta,
    }
    return normalized


# -------------------------
# Main entrypoint
# -------------------------
def _read_file_content(uploaded_file: Any) -> Tuple[str, bytes]:
    """
    Accept uploaded_file objects used by Streamlit (have .name and .read())
    or simple (name, bytes) tuples. Return (filename, bytes).
    """
    if isinstance(uploaded_file, (tuple, list)) and len(uploaded_file) == 2:
        return uploaded_file[0], uploaded_file[1]
    if hasattr(uploaded_file, "name") and hasattr(uploaded_file, "read"):
        name = getattr(uploaded_file, "name")
        data = uploaded_file.read()
        return name, data
    if isinstance(uploaded_file, str) and os.path.exists(uploaded_file):
        with open(uploaded_file, "rb") as f:
            return os.path.basename(uploaded_file), f.read()
    raise ValueError("Unsupported uploaded_file type. Must be stream-like with .name & .read() or (name, bytes).")


def process_uploaded_file(uploaded_file: Any) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Centralized ingestion for a single uploaded file.
    Accepts streamlit UploadedFile-like object (has .name and .read) or (name, bytes).
    Returns (messages, media_ocr_list).

    messages: list of normalized messages (canonical schema)
    media_ocr_list: list of dicts like {"file": filename, "ocr": text, "note": optional}
    """
    filename, content = _read_file_content(uploaded_file)
    lower = filename.lower()
    messages_raw: List[Dict[str, Any]] = []
    media_ocr: List[Dict[str, Any]] = []

    try:
        # ZIP handling
        if lower.endswith(".zip"):
            try:
                with zipfile.ZipFile(BytesIO(content)) as z:
                    for member in z.namelist():
                        if member.endswith("/"):
                            continue
                        try:
                            with z.open(member) as f:
                                b = f.read()
                        except Exception:
                            logger.exception("Failed to read zipped member %s", member)
                            continue
                        mname = os.path.basename(member)
                        mlow = mname.lower()
                        if mlow.endswith(".txt"):
                            try:
                                txt = b.decode("utf-8", errors="ignore")
                            except Exception:
                                txt = b.decode("latin-1", errors="ignore")
                            parsed = parse_whatsapp_text(txt)
                            for p in parsed:
                                p["source_hint"] = "whatsapp_txt"
                                messages_raw.append(p)
                        elif mlow.endswith(".json"):
                            parsed = parse_json_chat(b)
                            if parsed:
                                # try to ensure each item is a dict
                                for it in parsed:
                                    if isinstance(it, dict):
                                        it["source_hint"] = "json"
                                        messages_raw.append(it)
                        elif mlow.endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tiff", ".tif")):
                            ocr_text = ocr_image_bytes(b)
                            media_ocr.append({"file": member, "ocr": ocr_text})
                            if ocr_text.strip():
                                messages_raw.append({
                                    "date_raw": "",
                                    "time_raw": "",
                                    "author": "unknown",
                                    "text": ocr_text,
                                    "raw_line": "",
                                    "source_hint": "ocr_from_zip",
                                })
                        elif mlow.endswith(".pdf"):
                            pdf_text = extract_text_from_pdf(b)
                            if pdf_text.strip():
                                messages_raw.append({
                                    "date_raw": "",
                                    "time_raw": "",
                                    "author": "unknown",
                                    "text": pdf_text,
                                    "raw_line": "",
                                    "source_hint": "pdf_in_zip",
                                    "meta": {"filename": member}
                                })
                        elif mlow.endswith((".opus", ".mp4", ".avi", ".mov", ".m4a", ".wav", ".mp3", ".aac", ".ogg", ".flac", ".mkv", ".webm", ".flv", ".3gp")):
                            # Media file - extract metadata only
                            metadata = extract_media_metadata(b, member)
                            media_ocr.append({
                                "file": member,
                                "note": f"Media file detected: {metadata['media_type']} ({metadata['file_size_human']})",
                                "metadata": metadata
                            })
                        else:
                            # unsupported binary inside zip; record filename
                            media_ocr.append({"file": member, "note": "unsupported file type in zip"})
            except zipfile.BadZipFile:
                logger.exception("Uploaded zip file is corrupted or not a zip.")
                # treat as generic file below
        
        # Plain text files
        elif lower.endswith(".txt"):
            try:
                txt = content.decode("utf-8", errors="ignore")
            except Exception:
                txt = content.decode("latin-1", errors="ignore")
            parsed = parse_whatsapp_text(txt)
            for p in parsed:
                p["source_hint"] = "whatsapp_txt"
                messages_raw.append(p)
        
        # JSON chat
        elif lower.endswith(".json"):
            parsed = parse_json_chat(content)
            if parsed:
                for it in parsed:
                    if isinstance(it, dict):
                        it["source_hint"] = "json"
                        messages_raw.append(it)
            else:
                # attempt to decode as text and parse whatsapp style
                try:
                    txt = content.decode("utf-8", errors="ignore")
                    parsed = parse_whatsapp_text(txt)
                    if parsed:
                        for p in parsed:
                            p["source_hint"] = "whatsapp_txt_from_json_ext"
                            messages_raw.append(p)
                except Exception:
                    pass
        
        # Images (enhanced support)
        elif lower.endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tiff", ".tif")):
            ocr_text = ocr_image_bytes(content)
            media_ocr.append({"file": filename, "ocr": ocr_text})
            if ocr_text.strip():
                messages_raw.append({
                    "date_raw": "",
                    "time_raw": "",
                    "author": "unknown",
                    "text": ocr_text,
                    "raw_line": "",
                    "source_hint": "ocr_image",
                })
        
        # PDF
        elif lower.endswith(".pdf"):
            pdf_text = extract_text_from_pdf(content)
            if pdf_text.strip():
                messages_raw.append({
                    "date_raw": "",
                    "time_raw": "",
                    "author": "unknown",
                    "text": pdf_text,
                    "raw_line": "",
                    "source_hint": "pdf",
                    "meta": {"filename": filename}
                })
        
        # Media files (enhanced support)
        elif lower.endswith((".opus", ".mp4", ".avi", ".mov", ".m4a", ".wav", ".mp3", ".aac", ".ogg", ".flac", ".mkv", ".webm", ".flv", ".3gp")):
            metadata = extract_media_metadata(content, filename)
            media_ocr.append({
                "file": filename,
                "note": f"Media file detected: {metadata['media_type']} ({metadata['file_size_human']})",
                "metadata": metadata
            })
        
        else:
            # Generic attempt: try decode as text and parse whatsapp-like
            try:
                txt = content.decode("utf-8", errors="ignore")
                parsed = parse_whatsapp_text(txt)
                if parsed:
                    for p in parsed:
                        p["source_hint"] = "whatsapp_txt_generic"
                        messages_raw.append(p)
                else:
                    # if not whatsapp-like, still return decoded text as a single record
                    if txt.strip():
                        messages_raw.append({
                            "date_raw": "",
                            "time_raw": "",
                            "author": "unknown",
                            "text": txt,
                            "raw_line": "",
                            "source_hint": "generic_text",
                        })
            except Exception:
                # binary unknown
                media_ocr.append({"file": filename, "note": "unsupported binary and not text-decodable"})
    except Exception as exc:
        logger.exception("process_uploaded_file failed for %s: %s", filename, exc)
        media_ocr.append({"file": filename, "note": f"ingestion error: {exc}"})

    # Normalize messages
    normalized_msgs: List[Dict[str, Any]] = []
    for raw in messages_raw:
        try:
            norm = normalize_msg(raw)
            # attach meta about original raw for traceability
            norm_meta = dict(norm.get("meta", {}))
            norm_meta["_raw_preview"] = (raw.get("raw_line") or (raw.get("text")[:200] if raw.get("text") else ""))[:1000]
            norm["meta"] = norm_meta
            normalized_msgs.append(norm)
        except Exception:
            logger.exception("normalize_msg failed for raw: %s", raw)
            # fallback: wrap raw as text
            normalized_msgs.append({
                "uid": str(uuid.uuid4()),
                "date": "",
                "time": "",
                "author": raw.get("author", "unknown") if isinstance(raw, dict) else "unknown",
                "text": str(raw)[:2000],
                "source": raw.get("source_hint", "unknown") if isinstance(raw, dict) else "unknown",
                "media": [],
                "meta": {"_normalize_error": True}
            })

    return normalized_msgs, media_ocr


# -------------------------
# If run as script, simple demo
# -------------------------
if __name__ == "__main__":
    import sys
    from pprint import pprint

    if len(sys.argv) < 2:
        print("Usage: python ingestion.py <file>")
        sys.exit(1)
    fn = sys.argv[1]
    with open(fn, "rb") as fh:
        msgs, media = process_uploaded_file((os.path.basename(fn), fh.read()))
    print("MESSAGES:", len(msgs))
    pprint(msgs[:5])
    print("MEDIA OCR:", len(media))
    pprint(media[:5])
