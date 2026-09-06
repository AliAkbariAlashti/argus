import base64
import io
import logging

import cv2
from PIL import Image

log = logging.getLogger("qwenvl.imaging")


def frame_to_pil(frame):
    """BGR numpy frame -> RGB PIL image. Resolution/VRAM budget is enforced
    by the VLM processor's min_pixels/max_pixels (see vlm.py), not here."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def image_to_data_uri(img: Image.Image, max_width: int = 320) -> str | None:
    """Small JPEG data URI of a frame, so the UI can show what the model
    actually saw. Downscaled — it's a thumbnail, and it gets persisted on
    every chat turn and monitor event."""
    try:
        thumb = img.copy()
        if thumb.width > max_width:
            ratio = max_width / thumb.width
            thumb = thumb.resize((max_width, max(1, int(thumb.height * ratio))))
        buf = io.BytesIO()
        thumb.save(buf, format="JPEG", quality=70)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:  # noqa: BLE001
        log.exception("Failed to build snapshot thumbnail")
        return None
