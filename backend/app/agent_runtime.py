"""Bounded, model-driven Argus agent with typed operational tools."""
import json
import logging
from datetime import datetime, timedelta, timezone

import httpx

from .config import AGENT_API_KEY, AGENT_BASE_URL, AGENT_MAX_STEPS, AGENT_MODEL, AGENT_TIMEOUT_SECONDS
from .cpu import CpuSettings
from .imaging import frame_to_pil, image_to_data_uri
from .models import Camera, CpuConfig, Event, Observation

log = logging.getLogger("qwenvl.agent_runtime")

SYSTEM_PROMPT = """You are Argus, an operator for a CCTV system. Talk naturally and help the user investigate and operate their camera network. Use tools whenever the answer depends on cameras, current scenes, health, events, or observations. You may call several tools before answering. Never claim you saw something unless a tool returned evidence. State camera names and timestamps when available. Explain uncertainty briefly. Do not invent camera IDs, records, or tool results. Current UTC time: {now}."""


TOOLS = [
    {"type": "function", "function": {"name": "list_cameras", "description": "List configured cameras, locations, tags, source types, and current online state.", "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}},
    {"type": "function", "function": {"name": "get_camera_health", "description": "Get live stream and CPU-analysis health for all cameras or selected camera IDs.", "parameters": {"type": "object", "properties": {"camera_ids": {"type": "array", "items": {"type": "string"}, "maxItems": 20}}, "additionalProperties": False}}},
    {"type": "function", "function": {"name": "search_observations", "description": "Search structured detections, line crossings, dwell records, actions, zones, and tracks.", "parameters": {"type": "object", "properties": {"camera_ids": {"type": "array", "items": {"type": "string"}, "maxItems": 20}, "object_type": {"type": "string"}, "kind": {"type": "string", "enum": ["object_detection", "line_crossing", "dwell"]}, "zone": {"type": "string"}, "action": {"type": "string"}, "since_minutes": {"type": "integer", "minimum": 1, "maximum": 10080}, "limit": {"type": "integer", "minimum": 1, "maximum": 100}}, "additionalProperties": False}}},
    {"type": "function", "function": {"name": "search_events", "description": "Search operator-facing activity events and alerts.", "parameters": {"type": "object", "properties": {"camera_ids": {"type": "array", "items": {"type": "string"}, "maxItems": 20}, "category": {"type": "string"}, "severity": {"type": "string", "enum": ["info", "warning", "critical"]}, "since_minutes": {"type": "integer", "minimum": 1, "maximum": 10080}, "limit": {"type": "integer", "minimum": 1, "maximum": 100}}, "additionalProperties": False}}},
    {"type": "function", "function": {"name": "inspect_live_cameras", "description": "Ask the vision model to inspect current frames. Use only when pixels must be examined, after identifying relevant cameras.", "parameters": {"type": "object", "required": ["camera_ids", "question"], "properties": {"camera_ids": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 4}, "question": {"type": "string", "maxLength": 1000}}, "additionalProperties": False}}},
]


class AgentUnavailable(RuntimeError):
    pass


class AgentRuntime:
    def __init__(self, base_url=AGENT_BASE_URL, model=AGENT_MODEL, api_key=AGENT_API_KEY, max_steps=AGENT_MAX_STEPS):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.max_steps = max_steps

    @property
    def configured(self):
        return bool(self.base_url and self.model)

    def public_status(self):
        return {"configured": self.configured, "base_url": self.base_url, "model": self.model, "max_steps": self.max_steps}

    def _url(self):
        base = self.base_url[:-3] if self.base_url.endswith("/v1") else self.base_url
        return base + "/v1/chat/completions"

    def _complete(self, messages):
        if not self.configured:
            raise AgentUnavailable("No Argus instruction model is configured.")
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        payload = {"model": self.model, "messages": messages, "tools": TOOLS, "tool_choice": "auto", "temperature": 0.1, "stream": False}
        try:
            response = httpx.post(self._url(), headers=headers, json=payload, timeout=AGENT_TIMEOUT_SECONDS)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]
        except (httpx.HTTPError, KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise AgentUnavailable(f"Instruction model unavailable: {exc}") from None

    @staticmethod
    def _since(query, column, minutes):
        return query.filter(column >= datetime.now(timezone.utc) - timedelta(minutes=minutes)) if minutes else query

    def _run_tool(self, name, args, db, registry, cpu_monitor, vlm):
        cameras = db.query(Camera).order_by(Camera.created_at).all()
        by_id = {camera.id: camera for camera in cameras}
        requested = args.get("camera_ids") or []
        if requested and any(camera_id not in by_id for camera_id in requested):
            return {"error": "One or more camera IDs do not exist."}, [], None
        selected = [by_id[camera_id] for camera_id in requested] if requested else cameras
        if name == "list_cameras":
            data = [{**camera.to_dict(), "online": bool(registry.get(camera.id) and registry.get(camera.id).is_online())} for camera in cameras]
            return {"cameras": data}, [camera.id for camera in cameras], None
        if name == "get_camera_health":
            data = []
            for camera in selected:
                feed = registry.get(camera.id)
                config = db.get(CpuConfig, camera.id)
                state = cpu_monitor.status(camera.id)
                data.append({"camera_id": camera.id, "name": camera.name, "online": bool(feed and feed.is_online()), "fps": feed.get_fps() if feed else 0, "cpu_enabled": CpuSettings(**(config.settings if config else {})).enabled, "cpu_status": state})
            return {"cameras": data}, [camera.id for camera in selected], None
        if name == "search_observations":
            query = db.query(Observation)
            if requested: query = query.filter(Observation.camera_id.in_(requested))
            for key, column in (("object_type", Observation.object_type), ("kind", Observation.kind), ("zone", Observation.zone), ("action", Observation.action)):
                if args.get(key): query = query.filter(column.ilike(args[key]))
            query = self._since(query, Observation.observed_at, args.get("since_minutes"))
            rows = query.order_by(Observation.observed_at.desc()).limit(min(args.get("limit", 25), 100)).all()
            data = []
            for row in rows:
                item = row.to_dict()
                if item.get("attributes", {}).get("snapshot"):
                    item["attributes"] = {**item["attributes"], "snapshot": "available"}
                data.append(item)
            evidence = rows[0].attributes.get("snapshot") if rows and rows[0].attributes else None
            return {"count": len(rows), "observations": data}, list(dict.fromkeys(row.camera_id for row in rows)), evidence
        if name == "search_events":
            query = db.query(Event)
            if requested: query = query.filter(Event.camera_id.in_(requested))
            if args.get("category"): query = query.filter(Event.category.ilike(args["category"]))
            if args.get("severity"): query = query.filter(Event.severity == args["severity"])
            query = self._since(query, Event.created_at, args.get("since_minutes"))
            rows = query.order_by(Event.created_at.desc()).limit(min(args.get("limit", 25), 100)).all()
            data = []
            for row in rows:
                item = row.to_dict()
                item["snapshot"] = "available" if item.get("snapshot") else None
                data.append(item)
            return {"count": len(rows), "events": data}, list(dict.fromkeys(row.camera_id for row in rows)), rows[0].snapshot if rows else None
        if name == "inspect_live_cameras":
            if not vlm.ready:
                return {"error": vlm.error or "Vision model is not ready."}, [], None
            images, used, labels = [], [], []
            for camera in selected[:4]:
                feed = registry.get(camera.id)
                frame = feed.get_frame() if feed else None
                if frame is not None:
                    images.append(frame_to_pil(frame)); used.append(camera.id); labels.append(camera.name)
            if not images:
                return {"error": "Selected cameras have no current frames."}, [], None
            prompt = "Camera frames are ordered as: " + ", ".join(labels) + ". Answer only from these frames. " + args["question"]
            answer = vlm.ask(images, prompt, max_new_tokens=512)
            return {"answer": answer, "camera_ids": used, "camera_names": labels, "observed_at": datetime.now(timezone.utc).isoformat()}, used, image_to_data_uri(images[0])
        return {"error": f"Unknown tool: {name}"}, [], None

    def answer(self, question, history, db, registry, cpu_monitor, vlm, on_status=None):
        messages = [{"role": "system", "content": SYSTEM_PROMPT.format(now=datetime.now(timezone.utc).isoformat())}]
        messages.extend({"role": item["role"], "content": item["text"]} for item in history)
        messages.append({"role": "user", "content": question})
        trace, cameras_used, snapshot = [], [], None
        for step in range(self.max_steps):
            message = self._complete(messages)
            calls = message.get("tool_calls") or []
            if not calls:
                content = message.get("content")
                if not isinstance(content, str) or not content.strip():
                    raise AgentUnavailable("Instruction model returned no answer.")
                return {"answer": content.strip(), "cameras_used": list(dict.fromkeys(cameras_used)), "snapshot": snapshot, "tool_trace": trace, "steps": step + 1}
            messages.append(message)
            for call in calls:
                function = call.get("function") or {}
                name = function.get("name", "")
                try: args = json.loads(function.get("arguments") or "{}")
                except json.JSONDecodeError: args = {}
                if not isinstance(args, dict):
                    args = {}
                if on_status: on_status(f"Using {name.replace('_', ' ')}…")
                result, used, evidence = self._run_tool(name, args, db, registry, cpu_monitor, vlm)
                trace.append({"tool": name, "arguments": args, "result": result})
                cameras_used.extend(used)
                snapshot = snapshot or evidence
                messages.append({"role": "tool", "tool_call_id": call.get("id", f"step-{step}"), "name": name, "content": json.dumps(result, default=str)[:30000]})
        raise AgentUnavailable(f"Agent exceeded its {self.max_steps}-step limit.")


agent_runtime = AgentRuntime()
