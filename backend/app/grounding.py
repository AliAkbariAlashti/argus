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

# Phrases that are unambiguously about the whole fleet, even if the question
# also happens to mention a place ("is anyone in the server room or anywhere
# else"). These force a fleet search regardless of camera matching.
_FLEET_VISION_PATTERNS = [
    r"\ball (the )?cameras?\b",
    r"\bevery camera\b",
    r"\bany camera\b",
    r"\bany (of the )?(feeds?|views?)\b",
    r"\banywhere\b",
    r"\bacross (the|all) (building|site|cameras?)\b",
    r"\b(whole|entire) (building|site|facility|place)\b",
]
_FLEET_VISION_RE = re.compile("|".join(_FLEET_VISION_PATTERNS), re.IGNORECASE)

# Questions that ask the system to *locate* something — the answer IS which
# camera, so a focused camera can never satisfy them; checking the fleet is
# not optional. This is the case that used to silently collapse to one
# camera and give a confidently wrong answer ("where is the crowd shown?").
_LOCATE_PATTERNS = [
    r"\bwhere\b",
    r"\bwhich (camera|feed|one|area|zone|room|place|location)\b",
    r"\bfind\b",
    r"\bsearch\b",
    r"\blocate\b",
    r"\bwho (is|are)\b",
]
_LOCATE_RE = re.compile("|".join(_LOCATE_PATTERNS), re.IGNORECASE)

# Questions that ask about presence/count in whatever is currently being
# looked at ("is there anyone?", "how many people?"). These have a perfectly
# good answer scoped to one feed — if the user has a camera focused, that IS
# the scope they mean, same as the existing suggested-prompt chips. Only
# default to searching the fleet when nothing is focused, unlike _LOCATE_RE
# above which always means "check everywhere."
_PRESENCE_PATTERNS = [
    r"\bany(one|body|thing)\b",
    r"\bis there\b",
    r"\bare there\b",
    r"\bhow many people\b",
    r"\bcount\b",
]
_PRESENCE_RE = re.compile("|".join(_PRESENCE_PATTERNS), re.IGNORECASE)


def is_metadata_question(question: str) -> bool:
    return bool(_METADATA_RE.search(question))


def is_explicit_fleet_question(question: str) -> bool:
    """Question names the fleet outright — always search everything."""
    return bool(_FLEET_VISION_RE.search(question))


def is_locate_question(question: str) -> bool:
    """Question asks *where/which camera*, so the answer is meaningless
    against a single feed — the fleet must always be checked."""
    return bool(_LOCATE_RE.search(question))


def is_presence_question(question: str) -> bool:
    """Question asks about presence/count in whatever is being looked at.
    Answerable from one feed; only forces a fleet search when no camera is
    focused, unlike a locate question."""
    return bool(_PRESENCE_RE.search(question))


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


# A camera has to be named deliberately — by its name or an explicit zone
# tag — to narrow a question to it. Incidental overlap with a camera's prose
# description is not enough; that's what used to route "where is the crowd?"
# to a single hallway camera and answer it confidently and uselessly.
NAMED_CAMERA_THRESHOLD = 2.5


def route_question(question: str, cameras: list, focused_camera_id=None):
    """Decides what a question should be answered against.

    Returns (scope, camera) where scope is:
      "metadata" — answerable from the DB alone, no vision needed.
      "single"   — one camera, in `camera`.
      "fleet"    — every camera.

    The default is deliberately "fleet": this is a fleet monitoring tool, so
    searching everything is the sane default and a single camera is the
    narrow case the user opts into by naming a place. Getting this backwards
    means any phrasing we failed to anticipate silently answers from one
    arbitrary camera — which reads as the product not understanding the
    question at all.
    """
    if is_metadata_question(question):
        return "metadata", None

    # Explicit "all cameras / anywhere" always wins, even if a place is named.
    if is_explicit_fleet_question(question):
        return "fleet", None

    def _focused_camera():
        if not focused_camera_id:
            return None
        return next((c for c in cameras if c.id == focused_camera_id), None)

    match = best_matching_camera(question, cameras)
    named = (
        match is not None
        and camera_match_score(question, match) >= NAMED_CAMERA_THRESHOLD
    )

    if named:
        # A named place plus a locating question ("where else is anyone?")
        # still wants the fleet; a named place alone narrows to that camera.
        if is_locate_question(question) and _mentions_other_scope(question):
            return "fleet", None
        return "single", match

    # No camera named. "Where/which camera/find" only has an answer against
    # the whole fleet — a focused camera can't satisfy "where", since the
    # question IS asking which camera.
    if is_locate_question(question):
        return "fleet", None

    # "Is there anyone?" / "how many people?" without a named place: this
    # has a perfectly good answer scoped to whatever the user is currently
    # looking at, so a focused camera wins here — same as the ambiguous
    # case below. Only search the whole fleet if nothing is focused.
    if is_presence_question(question):
        focused = _focused_camera()
        return ("single", focused) if focused else ("fleet", None)

    # Genuinely ambiguous ("what is happening?"). If the user is looking at
    # a camera, answer about that one — otherwise search everything rather
    # than picking one arbitrarily.
    focused = _focused_camera()
    if focused:
        return "single", focused

    return "fleet", None


_OTHER_SCOPE_RE = re.compile(
    r"\b(else|other|others|too|also|rest|remaining)\b", re.IGNORECASE
)


def _mentions_other_scope(question: str) -> bool:
    """"...anywhere else", "any other camera" — widens a named-camera
    question back out to the fleet."""
    return bool(_OTHER_SCOPE_RE.search(question))


def build_grounded_prompt(camera, question: str, frame_count: int = 1) -> str:
    """Wraps the user's question with a short grounding preamble describing
    the camera, so the VLM's answer is contextualized without needing a
    separate retrieval/routing model call. When several frames are attached
    they're sent as a native video clip (see vlm.ask's `fps` path) so the
    model already knows their real spacing/order via M-RoPE — this note only
    adds task framing (which moment "now" is), not timing, which used to be
    a guessed sentence here."""
    tags = ", ".join(camera.zone_tags or []) or "none"

    if frame_count > 1:
        frames_note = (
            "This is a short live clip from the camera, oldest frame first — "
            "use it to judge movement and describe the current situation "
            "(the last frame is now).\n"
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
    it sees in each image to the right camera by name.

    The output instructions matter as much as the listing: without them the
    model tends to merge several feeds into one vague paragraph, which loses
    the single most useful part of the answer — *which camera*."""
    lines = [
        f"Image {i + 1} = camera \"{cam.name}\" ({cam.location}"
        + (f"; tags: {', '.join(cam.zone_tags)}" if cam.zone_tags else "")
        + ")"
        for i, cam in enumerate(cameras)
    ]
    listing = "\n".join(lines)
    n = len(cameras)

    return (
        f"You are a CCTV monitoring analyst reviewing {n} security cameras at "
        "the same site. Each attached image is the current frame from a "
        "different camera:\n"
        f"{listing}\n\n"
        "Instructions:\n"
        f"- Examine all {n} images before answering.\n"
        "- Always identify cameras by name, never as \"image 1\" or \"the first image\".\n"
        "- If the answer is on some cameras but not others, say which ones, "
        "and briefly note the cameras where it is absent.\n"
        "- If it appears on no camera, say so plainly.\n"
        "- Be specific and concise. Do not describe cameras that are not "
        "relevant to the question.\n"
        "- End your reply with a final line naming ONLY the camera(s) your "
        "answer is actually about (the ones with the relevant finding, not "
        "ones you only mentioned as clear), in exactly this format and "
        "nothing else on that line: [[cameras: Name One, Name Two]]. Use "
        "the exact camera names from the list above. If the finding is on "
        "no camera, write [[cameras: none]].\n\n"
        f"Question: {question}"
    )


_CAMERA_TAG_RE = re.compile(r"\[\[cameras:\s*(.*?)\s*\]\]\s*$", re.IGNORECASE | re.DOTALL)


def extract_primary_cameras(answer: str, cameras: list) -> tuple[str, list]:
    """Splits the model's trailing [[cameras: ...]] tag off a fleet answer
    and resolves it against the actual camera list by name.

    Returns (display_text, matched_cameras) — display_text has the tag
    stripped (never shown to the user), and matched_cameras is empty if the
    tag is missing, unparseable, or names no real camera (the caller should
    fall back to some other default, e.g. the first camera, in that case)."""
    match = _CAMERA_TAG_RE.search(answer)
    if not match:
        return answer, []

    display_text = answer[: match.start()].rstrip()
    if match.group(1).strip().lower() == "none":
        return display_text, []

    named = [n.strip().lower() for n in match.group(1).split(",") if n.strip()]
    matched = [c for c in cameras if (c.name or "").strip().lower() in named]
    return display_text, matched


def build_event_prompt(camera, watch_for: list[str] | None = None) -> str:
    """Structured classification prompt for the background monitor — JSON,
    not prose, so the caller can act on booleans instead of parsing natural
    language. Kept intentionally narrow (person/vehicle/weapon) rather than
    open-ended, since this runs unattended on a timer, not in response to a
    specific user question.

    `watch_for` folds an operator's custom alert-rule targets for this
    camera into this same call, instead of one extra VLM call per rule."""
    watch_block = ""
    if watch_for:
        quoted = ", ".join(f'"{w}"' for w in watch_for)
        watch_block = (
            "\nAlso specifically check for these operator-defined "
            f"conditions: {quoted}. List, in \"custom_matches\", exactly "
            "which of these strings (verbatim, from the list above) are "
            "currently visibly true in the frame — empty list if none "
            "apply.\n"
        )
    return (
        f"You are a CCTV monitoring analyst reviewing camera \"{camera.name}\" "
        f"at {camera.location}.\n"
        "Respond with ONLY a single JSON object, no other text, in exactly "
        "this shape:\n"
        '{"person_present": bool, "vehicle_present": bool, '
        '"weapon_visible": bool, "custom_matches": [string], '
        '"summary": "one short sentence"}\n'
        "\"weapon_visible\" means a firearm, knife, or other weapon is "
        "clearly visible — leave it false unless you are reasonably "
        "confident, since this is what triggers a real alert."
        f"{watch_block}"
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
