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


def camera_match_score(question: str, camera) -> int:
    """Cheap keyword overlap between a question and a camera's metadata
    (name, location, zone tags, description). Higher is a better match.
    Used to find which camera(s) a free-form question like "what's in the
    parking lot" refers to, without hitting the VLM at all for routing."""
    q_tokens = _tokenize(question)
    if not q_tokens:
        return 0

    haystack = " ".join(
        [camera.name, camera.location, camera.description, " ".join(camera.zone_tags or [])]
    )
    hay_tokens = _tokenize(haystack)

    # zone tags can be multi-word ("parking-lot" / "parking lot"); match those as substrings too
    score = len(q_tokens & hay_tokens)
    q_text = question.lower()
    for tag in camera.zone_tags or []:
        tag_norm = tag.replace("-", " ").lower()
        if tag_norm and tag_norm in q_text:
            score += 2
    return score


def best_matching_camera(question: str, cameras: list):
    """Returns the single best-matching camera, or None if nothing scores > 0."""
    scored = [(camera_match_score(question, c), c) for c in cameras]
    scored = [s for s in scored if s[0] > 0]
    if not scored:
        return None
    scored.sort(key=lambda s: s[0], reverse=True)
    return scored[0][1]


def build_grounded_prompt(camera, question: str) -> str:
    """Wraps the user's question with a short grounding preamble describing
    the camera, so the VLM's answer is contextualized without needing a
    separate retrieval/routing model call."""
    tags = ", ".join(camera.zone_tags or []) or "none"
    return (
        f"You are viewing a live feed from camera \"{camera.name}\" "
        f"located at {camera.location}. Zone tags: {tags}. "
        f"Camera notes: {camera.description or 'none'}.\n\n"
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
