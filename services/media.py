"""
Media service.
Downloads WhatsApp media, processes images/audio, and extracts text content.
"""
from __future__ import annotations

import io
import logging
import mimetypes
from pathlib import Path

logger = logging.getLogger(__name__)


SUPPORTED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
SUPPORTED_AUDIO_TYPES = {"audio/ogg", "audio/mpeg", "audio/mp4", "audio/wav"}
SUPPORTED_DOCUMENT_TYPES = {"application/pdf", "text/plain"}


async def handle_image(
    image_bytes: bytes,
    mime_type: str = "image/jpeg",
) -> str:
    """
    Describe an image using Claude's vision capability.
    Returns a text description.
    """
    import base64
    from config import get_settings
    settings = get_settings()

    b64 = base64.standard_b64encode(image_bytes).decode()
    
    from services.ai import _client
    client = _client()
    response = await client.chat.completions.create(
        model=settings.ocr_model,
        max_tokens=512,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64}"}},
                    {"type": "text", "text": "Describe this image briefly and extract any text you see."},
                ],
            }
        ],
    )
    return response.choices[0].message.content or ""


async def handle_audio(audio_bytes: bytes, mime_type: str = "audio/ogg") -> str:
    """Transcribe audio to text via Whisper."""
    from services.ai import transcribe_audio
    return await transcribe_audio(audio_bytes, mime_type)


async def handle_document(doc_bytes: bytes, mime_type: str = "application/pdf") -> str:
    """Extract text from a document."""
    if mime_type == "text/plain":
        return doc_bytes.decode("utf-8", errors="replace")

    if mime_type == "application/pdf":
        try:
            import pdfplumber
            import io
            with pdfplumber.open(io.BytesIO(doc_bytes)) as pdf:
                pages_text = [page.extract_text() or "" for page in pdf.pages]
                raw_text = "\n".join(pages_text).strip()
                
                # Extract first page as physical image and force Vision OCR on it
                im = pdf.pages[0].to_image(resolution=100).original
                img_io = io.BytesIO()
                im.save(img_io, format="JPEG")
                img_bytes = img_io.getvalue()
                
            # Run the derived image bytes back through your dedicated OCR handler
            ocr_text = await handle_image(img_bytes, "image/jpeg")
            
            # Pure Vision OCR response only
            return ocr_text.strip()
            
        except ImportError:
            logger.warning("pdfplumber not installed; returning raw bytes hint")
            return "[PDF content – install pdfplumber to extract text]"
        except Exception as exc:
            logger.error("Error processing PDF OCR: %s", exc)
            return "[PDF Processing Error]"

    return f"[Unsupported document type: {mime_type}]"


async def process_media(
    media_bytes: bytes,
    mime_type: str,
) -> str:
    """Route media to the correct handler and return a text representation."""
    if mime_type in SUPPORTED_IMAGE_TYPES:
        return await handle_image(media_bytes, mime_type)
    if mime_type in SUPPORTED_AUDIO_TYPES:
        return await handle_audio(media_bytes, mime_type)
    if mime_type in SUPPORTED_DOCUMENT_TYPES:
        return await handle_document(media_bytes, mime_type)

    logger.warning("Unsupported media type: %s", mime_type)
    return f"[Received a file of type {mime_type} – I can't process this yet.]"


def guess_mime_type(filename: str) -> str:
    mime, _ = mimetypes.guess_type(filename)
    return mime or "application/octet-stream"