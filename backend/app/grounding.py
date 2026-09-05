import re

_WORD_RE = re.compile(r"[a-z0-9]+")

# Phrases that signal a purely factual question about the camera fleet
# itself (count / list / online status) which has a correct, deterministic
# answer in the database — asking the VLM to guess this from a single image
# it can't see the fleet from is how you get hallucinated answers.
_METADATA_PATTERNS = [
    r"\bhow many cameras?\b",
    r"\bhow many feeds?\b",
    r"\blist (all )?(the )?cameras?\b",
    r"\bwhich cameras? (are|is)\b",
    r"\ball cameras?\b.*\b(online|offline|list|name)",
    r"\bcameras? (do we|are there)\b",
    r"\bwhat cameras? (do|are)\b",
]
_METADATA_RE = re.compile("|".join(_METADATA_PATTERNS), re.IGNORECASE)

# Phrases signaling the question wants to be checked against every camera's
# current view, not just one ("is anyone in the building", "check everywhere").
_FLEET_VISION_PATTERNS = [
    r"\ball cameras?\b",
    r"\bevery camera\b",
    r"\bany camera\b",
    r"\banywhere\b",
    r"\bacross (the|all) (building|site|cameras?)\b",
    r"\bwhole (building|site|facility)\b",
    r"\bentire (building|site|facility)\b",
]
_FLEET_VISION_RE = re.compile("|".join(_FLEET_VISION_PATTERNS), re.IGNORECASE)


def is_metadata_question(question: str) -> bool:
    return bool(_METADATA_RE.search(question))


def is_fleet_vision_question(question: str) -> bool:
    return bool(_FLEET_VISION_RE.search(question))


def _tokenize(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


# Words too common to indicate which camera someone means. Without this,
# a question sharing filler words with a camera's description can outrank
# an actual name match.
_STOPWORDS = {
    "a", "an", "and", "any", "anything", "are", "at", "be", "can", "do",
    "does", "for", "happening", "has", "have", "how", "i", "in", "is",
    "it", "many", "me", "much", "of", "on", "or", "see", "show", "someone",
    "something", "that", "the", "there", "they", "this", "to", "us", "we",
    "what", "when", "where", "which", "who", "why", "with", "you", "camera",
    "cameras", "feed", "feeds", "view", "now", "right", "currently",
}


def _content_tokens(text: str) -> set[str]:
    return _tokenize(text) - _STOPWORDS


def camera_match_score(question: str, camera) -> float:
    """How strongly a question refers to this camera, from its metadata.
    Fields are weighted by how deliberately they identify a camera: an
    explicit name beats a zone tag, which beats an incidental word shared
    with the free-text description."""
    q_tokens = _content_tokens(question)
    if not q_tokens:
        return 0.0

    q_text = question.lower()
    score = 0.0

    # Whole name appearing verbatim ("the east corridor") is the strongest
    # possible signal.
    name_lower = (camera.name or "").lower().strip()
    if name_lower and name_lower in q_text:
        score += 10.0

    # Partial name overlap ("corridor" for "East Corridor") — covers the
    # common case of referring to a camera by one distinctive word.
    name_tokens = _content_tokens(camera.name or "")
    score += 3.0 * len(q_tokens & name_tokens)

    # Zone tags are explicitly curated for this purpose.
    for tag in camera.zone_tags or []:
        tag_norm = tag.replace("-", " ").lower().strip()
        if not tag_norm:
            continue
        if tag_norm in q_text:
            score += 2.5
        else:
            score += 1.5 * len(q_tokens & _content_tokens(tag_norm))

    score += 1.5 * len(q_tokens & _content_tokens(camera.location or ""))
    # Description is prose, so treat overlap as weak corroboration only.
    score += 0.5 * len(q_tokens & _content_tokens(camera.description or ""))

    return score


# Below this, a "match" is more likely to be an incidental shared word than
# a real reference to the camera.
MATCH_THRESHOLD = 1.4


def best_matching_camera(question: str, cameras: list):
    """Returns the single best-matching camera, or None if nothing clears
    the confidence threshold."""
    scored = [(camera_match_score(question, c), c) for c in cameras]
    scored = [s for s in scored if s[0] >= MATCH_THRESHOLD]
    if not scored:
        return None
    scored.sort(key=lambda s: s[0], reverse=True)
    return scored[0][1]


def build_grounded_prompt(camera, question: str, frame_count: int = 1) -> str:
    """Wraps the user's question with a short grounding preamble describing
    the camera, so the VLM's answer is contextualized without needing a
    separate retrieval/routing model call. When several frames are attached
    they are consecutive moments, so say so — otherwise the model describes
    them as unrelated images instead of reading movement from them."""
    tags = ", ".join(camera.zone_tags or []) or "none"

    if frame_count > 1:
        frames_note = (
            f"The {frame_count} images are consecutive frames from this camera, "
            "about a second apart, oldest first — use them to judge movement "
            "and describe the current situation (the last frame is now).\n"
        )
    else:
        frames_note = ""

    return (
        f"You are viewing a live feed from camera \"{camera.name}\" "
        f"located at {camera.location}. Zone tags: {tags}. "
        f"Camera notes: {camera.description or 'none'}.\n"
        f"{frames_note}\n"
        f"Question: {question}"
    )


def build_fleet_prompt(cameras: list, question: str) -> str:
    """Wraps a question with a preamble listing every camera whose current
    frame is attached (in the same order), so the model can attribute what
    it sees in each image to the right camera by name."""
    lines = [
        f"{i + 1}. \"{cam.name}\" — {cam.location} (tags: {', '.join(cam.zone_tags or []) or 'none'})"
        for i, cam in enumerate(cameras)
    ]
    listing = "\n".join(lines)
    return (
        "You are viewing live feeds from multiple CCTV cameras at once. "
        "Each image below is the current frame from one camera, in this order:\n"
        f"{listing}\n\n"
        "When answering, refer to cameras by name and say which camera(s) "
        "your observation comes from.\n\n"
        f"Question: {question}"
    )


def metadata_answer(question: str, camera_dicts: list[dict]) -> str:
    """Deterministic answer for questions about the fleet itself (count,
    listing, online status) that don't require looking at any image.
    Expects dicts shaped like main._camera_out() (has an "online" key)."""
    if not camera_dicts:
        return "There are no cameras configured yet."

    lines = [
        f"- {cam['name']} ({cam['location']}) — {'online' if cam['online'] else 'offline'}"
        for cam in camera_dicts
    ]
    return (
        f"There {'is' if len(camera_dicts) == 1 else 'are'} {len(camera_dicts)} "
        f"camera{'s' if len(camera_dicts) != 1 else ''} configured:\n" + "\n".join(lines)
    )
