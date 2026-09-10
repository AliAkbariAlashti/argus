"""Deterministic natural-language planning over structured observations."""
import re
from datetime import datetime, timedelta, timezone

from sqlalchemy import func

from . import history, yolo
from .models import Event, Observation


_HISTORICAL_RE = re.compile(
    r"\b(last|latest|recent|recently|today|yesterday|history|historical|event|events|"
    r"detected|detection|seen|saw|entered|exited|crossed|crossing|stayed|dwell|ago)\b",
    re.IGNORECASE,
)
_CURRENT_RE = re.compile(r"\b(now|currently|right now|at the moment)\b", re.IGNORECASE)
_VEHICLES = ["bicycle", "car", "motorcycle", "bus", "truck"]


def _object_types(question):
    text = question.lower().replace("-", " ")
    aliases = {
        "people": ["person"], "persons": ["person"], "someone": ["person"], "anyone": ["person"],
        "vehicles": _VEHICLES, "vehicle": _VEHICLES,
    }
    for alias, classes in aliases.items():
        if re.search(rf"\b{re.escape(alias)}\b", text):
            return classes
    for name in sorted(yolo.CLASSES, key=len, reverse=True):
        forms = [name, name + "s"]
        if name.endswith("y"):
            forms.append(name[:-1] + "ies")
        if any(re.search(rf"\b{re.escape(form)}\b", text) for form in forms):
            return [name]
    return []


def _minimum_dwell_seconds(question):
    match = re.search(r"\b(?:for|over|longer than|at least)\s+(\d+)\s*(second|minute|hour)s?\b", question, re.IGNORECASE)
    if not match:
        return None
    amount = int(match.group(1))
    return amount * {"second": 1, "minute": 60, "hour": 3600}[match.group(2).lower()]


def plan_observation_query(question, cameras, forced_camera_id=None, now=None):
    """Return a query plan only when structured history can answer safely."""
    if _CURRENT_RE.search(question) or not _HISTORICAL_RE.search(question):
        return None
    now = now or datetime.now(timezone.utc)
    camera_dicts = [{"id": c.id, "name": c.name, "zone_tags": c.zone_tags or []} for c in cameras]
    parsed = history.parse_question(question, camera_dicts, now=now)
    lowered = question.lower()
    until = now.replace(hour=0, minute=0, second=0, microsecond=0).isoformat() if re.search(r"\byesterday\b", lowered) else None
    if parsed["since"] is None and re.search(r"\brecent(?:ly)?\b", lowered):
        parsed["since"] = (now - timedelta(hours=24)).isoformat()
    if re.search(r"\b(how many|count|number of)\b", lowered):
        intent = "count"
    elif re.search(r"\b(which cameras?|where)\b", lowered):
        intent = "cameras"
    elif re.search(r"\b(how long|longest|dwell)\b", lowered):
        intent = "dwell"
    elif re.search(r"\b(last|latest|most recent|when)\b", lowered):
        intent = "latest"
    else:
        intent = "list"
    kind = None
    if re.search(r"\b(crossed|crossing|entered|exited)\b", lowered):
        kind = "line_crossing"
    elif re.search(r"\b(stayed|dwell)\b", lowered):
        kind = "dwell"
    return {
        "intent": intent,
        "camera_id": forced_camera_id or parsed["camera_id"],
        "object_types": _object_types(question),
        "kind": kind,
        "since": parsed["since"],
        "until": until,
        "minimum_dwell_seconds": _minimum_dwell_seconds(question),
    }


def observation_query(db, plan):
    query = db.query(Observation)
    if plan["camera_id"]:
        query = query.filter(Observation.camera_id == plan["camera_id"])
    if plan["object_types"]:
        query = query.filter(Observation.object_type.in_(plan["object_types"]))
    if plan["kind"]:
        query = query.filter(Observation.kind == plan["kind"])
    if plan["since"]:
        since = datetime.fromisoformat(plan["since"])
        query = query.filter(Observation.observed_at >= (since if since.tzinfo else since.replace(tzinfo=timezone.utc)))
    if plan["until"]:
        until = datetime.fromisoformat(plan["until"])
        query = query.filter(Observation.observed_at < (until if until.tzinfo else until.replace(tzinfo=timezone.utc)))
    if plan["minimum_dwell_seconds"] is not None:
        query = query.filter(Observation.dwell_seconds >= plan["minimum_dwell_seconds"])
    return query


def _noun(plan):
    if not plan["object_types"]:
        return "object"
    if len(plan["object_types"]) > 1:
        return "vehicle"
    return plan["object_types"][0]


def _plural(noun):
    return "people" if noun == "person" else noun + "s"


def answer_observation_question(db, question, cameras, forced_camera_id=None):
    plan = plan_observation_query(question, cameras, forced_camera_id)
    if plan is None:
        return None
    query = observation_query(db, plan)
    newest = query.order_by(Observation.observed_at.desc(), Observation.id.desc()).first()
    camera_names = {camera.id: camera.name for camera in cameras}
    noun = _noun(plan)
    if newest is None:
        return {"answer": "No matching structured observations were found.", "cameras_used": [], "snapshot": None, "plan": plan}

    if plan["intent"] == "count":
        if plan["kind"] == "line_crossing":
            crossings = query.count()
            answer = f"I found {crossings} matching line-crossing event{'' if crossings == 1 else 's'}."
        else:
            tracked = query.filter(Observation.track_id.isnot(None)).with_entities(func.count(func.distinct(Observation.track_id))).scalar() or 0
            untracked = query.filter(Observation.track_id.is_(None)).count()
            if tracked:
                answer = f"I found {tracked} unique tracked {noun if tracked == 1 else _plural(noun)}"
                if untracked:
                    answer += f" plus {untracked} untracked detection observation{'' if untracked == 1 else 's'}"
                answer += "."
            else:
                answer = f"I found {untracked} matching {noun} detection observation{'' if untracked == 1 else 's'}."
        rows = query.with_entities(Observation.camera_id).distinct().limit(500).all()
        used = [row[0] for row in rows]
    elif plan["intent"] == "cameras":
        rows = query.with_entities(Observation.camera_id, func.count(Observation.id)).group_by(Observation.camera_id).order_by(func.count(Observation.id).desc()).all()
        used = [row[0] for row in rows]
        answer = "Matching observations were found on: " + ", ".join(f"{camera_names.get(camera_id, camera_id)} ({count})" for camera_id, count in rows) + "."
    elif plan["intent"] == "dwell":
        longest = query.order_by(Observation.dwell_seconds.is_(None), Observation.dwell_seconds.desc(), Observation.observed_at.desc()).first()
        used = [longest.camera_id]
        answer = f"The longest matching {noun} dwell observation was {longest.dwell_seconds or 0:g} seconds on {camera_names.get(longest.camera_id, longest.camera_id)}, at {longest.to_dict()['observed_at']}."
        newest = longest
    elif plan["intent"] == "latest":
        used = [newest.camera_id]
        detail = f", crossing {newest.direction.replace('_', '→') if newest.direction else ''}" if newest.kind == "line_crossing" else ""
        answer = f"The latest matching {noun} observation was on {camera_names.get(newest.camera_id, newest.camera_id)} at {newest.to_dict()['observed_at']}{detail}."
    else:
        rows = query.order_by(Observation.observed_at.desc(), Observation.id.desc()).limit(20).all()
        used = list(dict.fromkeys(row.camera_id for row in rows))
        answer = f"I found {len(rows)} recent matching observation{'' if len(rows) == 1 else 's'} across " + ", ".join(camera_names.get(camera_id, camera_id) for camera_id in used) + "."

    event = db.get(Event, newest.event_id) if newest.event_id is not None else None
    return {"answer": answer, "cameras_used": used, "snapshot": event.snapshot if event else None, "plan": plan}
