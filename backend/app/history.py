"""Rule-based natural-language search over already-logged events.

No model involved — every event (from the VLM monitor, or CPU tools' own
motion/face/people/object detectors) already carries a camera, category,
timestamp and snapshot. "When did you last see a person on camera 1" is a
structured filter over that table, not a generation task, so this parses
the question into filter parameters directly instead of calling any model.
"""
import re
from datetime import datetime, timedelta, timezone

from . import yolo

# Words that signal "most recent match only" rather than a list.
_LAST_WORDS = {"last", "latest", "most recent", "recently", "recent"}

# Relative time windows, longest phrase first so "last hour" doesn't get
# eaten by a shorter "hour" match.
_TIME_WINDOWS = [
    (r"\btoday\b", lambda now: now.replace(hour=0, minute=0, second=0, microsecond=0)),
    (r"\byesterday\b", lambda now: (now - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)),
    (r"\bthis week\b", lambda now: now - timedelta(days=now.weekday())),
    (r"\blast (?:24 hours|day)\b", lambda now: now - timedelta(hours=24)),
    (r"\blast hour\b", lambda now: now - timedelta(hours=1)),
    (r"\blast week\b", lambda now: now - timedelta(weeks=1)),
    (r"\blast (\d+) (minute|hour|day|week)s?\b", None),  # handled specially below
]

_STOPWORDS = {
    "the", "a", "an", "on", "in", "at", "did", "do", "does", "when", "what",
    "was", "were", "is", "are", "you", "i", "we", "see", "seen", "saw",
    "show", "showed", "detect", "detected", "detection", "of", "to", "me",
    "any", "there", "camera", "cameras", "source", "sources", "for",
    "happened", "happen", "happening", "occurred", "occur", "activity",
    "event", "events", "log", "history", "please", "and", "or", "with",
} | _LAST_WORDS


def _match_time_window(question: str, now: datetime):
    numeric = re.search(r"\blast (\d+) (minute|hour|day|week)s?\b", question)
    if numeric:
        amount, unit = int(numeric.group(1)), numeric.group(2)
        delta = {"minute": timedelta(minutes=amount), "hour": timedelta(hours=amount),
                 "day": timedelta(days=amount), "week": timedelta(weeks=amount)}[unit]
        return now - delta
    for pattern, resolver in _TIME_WINDOWS:
        if resolver is None:
            continue
        if re.search(pattern, question):
            return resolver(now)
    return None


def _normalize(text: str) -> str:
    """Zone tags use hyphens ("data-center"); questions use spaces ("data center")."""
    return text.lower().strip().replace("-", " ")


def _match_camera(question: str, cameras: list[dict]):
    """Best-effort match against camera name or zone tags. Longest match
    wins so "server room" beats a stray "room"."""
    lowered = _normalize(question)
    best = None
    for cam in cameras:
        candidates = [cam["name"]] + (cam.get("zone_tags") or [])
        for candidate in candidates:
            candidate = _normalize(candidate)
            if candidate and candidate in lowered:
                if best is None or len(candidate) > len(best[1]):
                    best = (cam["id"], candidate)
    return best


def _match_severity(question: str):
    for severity in ("critical", "warning", "info"):
        if re.search(rf"\b{severity}\b", question, re.IGNORECASE):
            return severity
    return None


def _extract_keyword(question: str, camera_match_text: str | None, severity: str | None):
    """What's left after removing stopwords, time phrases, the matched
    camera name, and the matched severity word — the actual thing being
    searched for ("person", "car"). Severity is its own filter already;
    leaving it in as a keyword would also require it to appear in the
    event's category/summary text, which it never does."""
    cleaned = _normalize(question)
    cleaned = re.sub(r"\blast (\d+) (minute|hour|day|week)s?\b", " ", cleaned)
    for pattern, _ in _TIME_WINDOWS:
        cleaned = re.sub(pattern, " ", cleaned)
    if camera_match_text:
        cleaned = cleaned.replace(camera_match_text, " ")
    if severity:
        cleaned = re.sub(rf"\b{severity}\b", " ", cleaned)
    cleaned = re.sub(r"[^\w\s]", " ", cleaned)
    words = [w for w in cleaned.split() if w and w not in _STOPWORDS]
    if not words:
        return None
    # Prefer a known YOLO class if one of the remaining words matches —
    # gives an exact, reliable keyword instead of a loose phrase.
    for word in words:
        if word in yolo.CLASSES:
            return word
    return " ".join(words)


def parse_question(question: str, cameras: list[dict], now: datetime | None = None):
    """Returns filter params for /api/events: camera_id, q, severity, since,
    and limit (1 if the question asks for the most recent occurrence)."""
    now = now or datetime.now(timezone.utc)
    lowered = _normalize(question)
    match = _match_camera(question, cameras)
    camera_id, camera_match_text = match if match else (None, None)

    since = _match_time_window(lowered, now)
    severity = _match_severity(lowered)
    keyword = _extract_keyword(question, camera_match_text, severity)
    wants_last = any(word in lowered for word in _LAST_WORDS) or "when" in lowered

    return {
        "camera_id": camera_id,
        "q": keyword,
        "severity": severity,
        "since": since.isoformat() if since else None,
        "limit": 1 if wants_last else 50,
    }
