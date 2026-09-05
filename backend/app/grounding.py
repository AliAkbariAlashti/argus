import re

_WORD_RE = re.compile(r"[a-z0-9]+")


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
