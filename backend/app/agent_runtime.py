"""Bounded, model-driven Argus agent with typed operational tools."""
import json
import logging
import time
from datetime import datetime, timedelta, timezone

import httpx
from sqlalchemy import func

from .config import AGENT_API_KEY, AGENT_BASE_URL, AGENT_MAX_STEPS, AGENT_MODEL, AGENT_TIMEOUT_SECONDS
from .agent_actions import create_proposal
from .cpu import CpuSettings
from .imaging import frame_to_pil, image_to_data_uri
from .models import AgentToolRun, AlertRule, Camera, CpuConfig, Event, Observation
from .entity_matching import suggest_matches
from .visual_search import find_by_text, find_similar

log = logging.getLogger("qwenvl.agent_runtime")

SYSTEM_PROMPT = """You are Argus, an operator for a CCTV system. Talk naturally and help the user investigate and operate their camera network. Use tools whenever the answer depends on cameras, current scenes, health, events, observations, rules, or configuration. Start by loading up to four relevant tools from the catalog, then call them as needed. You may load more tools in a later step. Never claim you saw something unless a tool returned evidence. State camera names and timestamps when available. Explain uncertainty briefly. Do not invent camera IDs, records, or tool results. Any state-changing tool creates a proposal only; tell the user what you prepared and that they must approve its confirmation card. Current UTC time: {now}.\nTool catalog:\n{catalog}"""


TOOLS = [
    {"type": "function", "function": {"name": "list_cameras", "description": "List configured cameras, locations, tags, source types, and current online state.", "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}},
    {"type": "function", "function": {"name": "get_camera_health", "description": "Get live stream and CPU-analysis health for all cameras or selected camera IDs.", "parameters": {"type": "object", "properties": {"camera_ids": {"type": "array", "items": {"type": "string"}, "maxItems": 20}}, "additionalProperties": False}}},
    {"type": "function", "function": {"name": "search_observations", "description": "Search structured detections, line crossings, dwell records, actions, zones, and tracks.", "parameters": {"type": "object", "properties": {"camera_ids": {"type": "array", "items": {"type": "string"}, "maxItems": 20}, "object_type": {"type": "string"}, "kind": {"type": "string", "enum": ["object_detection", "line_crossing", "dwell"]}, "zone": {"type": "string"}, "action": {"type": "string"}, "since_minutes": {"type": "integer", "minimum": 1, "maximum": 10080}, "limit": {"type": "integer", "minimum": 1, "maximum": 100}}, "additionalProperties": False}}},
    {"type": "function", "function": {"name": "search_events", "description": "Search operator-facing activity events and alerts.", "parameters": {"type": "object", "properties": {"camera_ids": {"type": "array", "items": {"type": "string"}, "maxItems": 20}, "category": {"type": "string"}, "severity": {"type": "string", "enum": ["info", "warning", "critical"]}, "since_minutes": {"type": "integer", "minimum": 1, "maximum": 10080}, "limit": {"type": "integer", "minimum": 1, "maximum": 100}}, "additionalProperties": False}}},
    {"type": "function", "function": {"name": "get_activity_report", "description": "Return exact grouped counts of observations and events for a time window.", "parameters": {"type": "object", "properties": {"camera_ids": {"type": "array", "items": {"type": "string"}, "maxItems": 20}, "since_minutes": {"type": "integer", "minimum": 1, "maximum": 10080}}, "additionalProperties": False}}},
    {"type": "function", "function": {"name": "get_track_timeline", "description": "Get the chronological observation timeline for one track or confirmed cross-camera entity.", "parameters": {"type": "object", "properties": {"track_id": {"type": "string"}, "global_entity_id": {"type": "string"}, "limit": {"type": "integer", "minimum": 1, "maximum": 200}}, "anyOf": [{"required": ["track_id"]}, {"required": ["global_entity_id"]}], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "find_similar_appearance", "description": "Find observations whose tracked object looks similar to a reference observation.", "parameters": {"type": "object", "required": ["observation_id"], "properties": {"observation_id": {"type": "string"}, "camera_id": {"type": "string"}, "object_type": {"type": "string"}, "limit": {"type": "integer", "minimum": 1, "maximum": 50}}, "additionalProperties": False}}},
    {"type": "function", "function": {"name": "search_visual_description", "description": "Semantic search for a visual description across stored observation embeddings. Availability depends on the configured embedding provider.", "parameters": {"type": "object", "required": ["description"], "properties": {"description": {"type": "string"}, "camera_id": {"type": "string"}, "object_type": {"type": "string"}, "limit": {"type": "integer", "minimum": 1, "maximum": 50}}, "additionalProperties": False}}},
    {"type": "function", "function": {"name": "suggest_cross_camera_matches", "description": "Find likely continuations of a tracked observation in connected cameras using appearance and travel-time constraints.", "parameters": {"type": "object", "required": ["observation_id"], "properties": {"observation_id": {"type": "string"}, "limit": {"type": "integer", "minimum": 1, "maximum": 20}}, "additionalProperties": False}}},
    {"type": "function", "function": {"name": "list_alert_rules", "description": "List alert rules and whether they are enabled.", "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}},
    {"type": "function", "function": {"name": "list_ai_setups", "description": "List saved vision-model setups and their active/readiness state.", "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}},
    {"type": "function", "function": {"name": "get_cpu_configuration", "description": "Read CPU-tool settings for selected or all cameras.", "parameters": {"type": "object", "properties": {"camera_ids": {"type": "array", "items": {"type": "string"}, "maxItems": 20}}, "additionalProperties": False}}},
    {"type": "function", "function": {"name": "propose_alert_rule", "description": "Prepare an alert rule for operator confirmation. This does not change the system.", "parameters": {"type": "object", "required": ["target"], "properties": {"camera_id": {"type": "string"}, "source": {"type": "string", "enum": ["cpu", "vlm"]}, "target": {"type": "string"}, "severity": {"type": "string", "enum": ["info", "warning", "critical"]}}, "additionalProperties": False}}},
    {"type": "function", "function": {"name": "propose_update_alert_rule", "description": "Prepare enabling, disabling, or changing severity of an alert rule for confirmation.", "parameters": {"type": "object", "required": ["rule_id"], "properties": {"rule_id": {"type": "string"}, "enabled": {"type": "boolean"}, "severity": {"type": "string", "enum": ["info", "warning", "critical"]}}, "additionalProperties": False}}},
    {"type": "function", "function": {"name": "propose_delete_alert_rule", "description": "Prepare deleting an alert rule for confirmation.", "parameters": {"type": "object", "required": ["rule_id"], "properties": {"rule_id": {"type": "string"}}, "additionalProperties": False}}},
    {"type": "function", "function": {"name": "propose_cpu_configuration", "description": "Prepare a partial CPU-tool configuration change for confirmation. This does not change the system.", "parameters": {"type": "object", "required": ["camera_id", "settings"], "properties": {"camera_id": {"type": "string"}, "settings": {"type": "object", "properties": {"enabled": {"type": "boolean"}, "motion_alerts": {"type": "boolean"}, "quality_alerts": {"type": "boolean"}, "object_alerts": {"type": "boolean"}, "tracking_enabled": {"type": "boolean"}, "line_crossing_alerts": {"type": "boolean"}, "dwell_alerts": {"type": "boolean"}, "dwell_seconds": {"type": "integer", "minimum": 10, "maximum": 86400}, "object_confidence": {"type": "number", "minimum": 0.1, "maximum": 0.95}}, "additionalProperties": False}}, "additionalProperties": False}}},
    {"type": "function", "function": {"name": "propose_rtsp_camera", "description": "Prepare adding an RTSP camera for confirmation. This does not change the system.", "parameters": {"type": "object", "required": ["name", "rtsp_url"], "properties": {"name": {"type": "string"}, "rtsp_url": {"type": "string"}, "location": {"type": "string"}, "zone_tags": {"type": "array", "items": {"type": "string"}}, "description": {"type": "string"}}, "additionalProperties": False}}},
    {"type": "function", "function": {"name": "propose_update_rtsp_camera", "description": "Prepare changing an RTSP camera name, location, tags, description, or URL for confirmation.", "parameters": {"type": "object", "required": ["camera_id"], "properties": {"camera_id": {"type": "string"}, "name": {"type": "string"}, "rtsp_url": {"type": "string"}, "location": {"type": "string"}, "zone_tags": {"type": "array", "items": {"type": "string"}}, "description": {"type": "string"}}, "additionalProperties": False}}},
    {"type": "function", "function": {"name": "propose_delete_camera", "description": "Prepare deleting an RTSP camera for confirmation. This does not change the system.", "parameters": {"type": "object", "required": ["camera_id"], "properties": {"camera_id": {"type": "string"}}, "additionalProperties": False}}},
    {"type": "function", "function": {"name": "propose_activate_ai_setup", "description": "Prepare activating a saved vision-model setup for confirmation.", "parameters": {"type": "object", "required": ["profile_id"], "properties": {"profile_id": {"type": "string"}}, "additionalProperties": False}}},
    {"type": "function", "function": {"name": "inspect_live_cameras", "description": "Ask the vision model to inspect current frames. Use only when pixels must be examined, after identifying relevant cameras.", "parameters": {"type": "object", "required": ["camera_ids", "question"], "properties": {"camera_ids": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 4}, "question": {"type": "string", "maxLength": 1000}}, "additionalProperties": False}}},
]

TOOL_BY_NAME = {item["function"]["name"]: item for item in TOOLS}
TOOL_CATALOG = "\n".join(f"- {name}: {item['function']['description']}" for name, item in TOOL_BY_NAME.items())
LOAD_TOOLS = {"type": "function", "function": {"name": "load_tools", "description": "Load the schemas for tools needed to handle this request.", "parameters": {"type": "object", "required": ["names"], "properties": {"names": {"type": "array", "items": {"type": "string", "enum": list(TOOL_BY_NAME)}, "minItems": 1, "maxItems": 4}}, "additionalProperties": False}}}


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

    def _is_ollama(self):
        return "ollama" in self.base_url.lower() or any(port in self.base_url for port in (":11434", ":11435"))

    @staticmethod
    def _merge_tool_delta(target, delta):
        index = int(delta.get("index", 0))
        while len(target) <= index:
            target.append({"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
        call = target[index]
        if delta.get("id"): call["id"] = delta["id"]
        function = delta.get("function") or {}
        call["function"]["name"] += function.get("name") or ""
        arguments = function.get("arguments")
        call["function"]["arguments"] += json.dumps(arguments) if isinstance(arguments, dict) else arguments or ""

    def _complete(self, messages, tools=None, on_token=None):
        if not self.configured:
            raise AgentUnavailable("No Argus instruction model is configured.")
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        payload = {"model": self.model, "messages": messages, "tools": tools or [LOAD_TOOLS], "stream": on_token is not None}
        url = self._url()
        if self._is_ollama():
            base = self.base_url[:-3] if self.base_url.endswith("/v1") else self.base_url
            url = base + "/api/chat"
            payload.update({"think": False, "options": {"temperature": 0.1, "num_predict": 512}})
        else:
            payload.update({"tool_choice": "auto", "temperature": 0.1, "max_tokens": 512, "reasoning_effort": "none"})
        try:
            if on_token is None:
                response = httpx.post(url, headers=headers, json=payload, timeout=AGENT_TIMEOUT_SECONDS)
                response.raise_for_status()
                packet = response.json()
                message = packet["message"] if self._is_ollama() else packet["choices"][0]["message"]
            else:
                content, calls = [], []
                with httpx.stream("POST", url, headers=headers, json=payload, timeout=AGENT_TIMEOUT_SECONDS) as response:
                    response.raise_for_status()
                    for line in response.iter_lines():
                        if not line: continue
                        if self._is_ollama():
                            packet = json.loads(line); delta = packet.get("message") or {}
                        else:
                            if not line.startswith("data:"): continue
                            data = line[5:].strip()
                            if data == "[DONE]": break
                            packet = json.loads(data); choices = packet.get("choices") or []
                            if not choices: continue
                            delta = choices[0].get("delta") or {}
                        chunk = delta.get("content") or ""
                        if chunk:
                            content.append(chunk); on_token(chunk)
                        for call_delta in delta.get("tool_calls") or []:
                            self._merge_tool_delta(calls, call_delta)
                message = {"role": "assistant", "content": "".join(content), "tool_calls": calls}
            for call in message.get("tool_calls") or []:
                arguments = (call.get("function") or {}).get("arguments")
                if isinstance(arguments, dict):
                    call["function"]["arguments"] = json.dumps(arguments)
            return message
        except (httpx.HTTPError, KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise AgentUnavailable(f"Instruction model unavailable: {exc}") from None

    @staticmethod
    def _since(query, column, minutes):
        return query.filter(column >= datetime.now(timezone.utc) - timedelta(minutes=minutes)) if minutes else query

    def _run_tool(self, name, args, db, registry, cpu_monitor, vlm, session_id="unsaved"):
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
        if name == "get_activity_report":
            since = datetime.now(timezone.utc) - timedelta(minutes=args.get("since_minutes", 1440))
            observation_query = db.query(Observation.kind, func.count(Observation.id)).filter(Observation.observed_at >= since)
            event_query = db.query(Event.severity, func.count(Event.id)).filter(Event.created_at >= since)
            if requested:
                observation_query = observation_query.filter(Observation.camera_id.in_(requested))
                event_query = event_query.filter(Event.camera_id.in_(requested))
            observation_counts = dict(observation_query.group_by(Observation.kind).all())
            event_counts = dict(event_query.group_by(Event.severity).all())
            return {"since": since.isoformat(), "observation_counts": observation_counts, "event_counts": event_counts,
                    "total_observations": sum(observation_counts.values()), "total_events": sum(event_counts.values())}, [camera.id for camera in selected], None
        if name == "get_track_timeline":
            query = db.query(Observation)
            if args.get("global_entity_id"):
                query = query.filter(Observation.global_entity_id == args["global_entity_id"])
            elif args.get("track_id"):
                query = query.filter(Observation.track_id == args["track_id"])
            else:
                return {"error": "Provide track_id or global_entity_id."}, [], None
            rows = query.order_by(Observation.observed_at).limit(min(args.get("limit", 100), 200)).all()
            return {"count": len(rows), "timeline": [row.to_dict() for row in rows]}, list(dict.fromkeys(row.camera_id for row in rows)), None
        if name == "find_similar_appearance":
            result = find_similar(db, args["observation_id"], camera_id=args.get("camera_id"), object_type=args.get("object_type"), limit=args.get("limit", 20))
            if result is None: return {"error": "Reference observation has no visual embedding."}, [], None
            used = list(dict.fromkeys(item["camera_id"] for item in result.get("matches", [])))
            return result, used, None
        if name == "search_visual_description":
            try:
                result = find_by_text(db, args["description"], camera_id=args.get("camera_id"), object_type=args.get("object_type"), limit=args.get("limit", 20))
            except ValueError as exc:
                return {"error": str(exc)}, [], None
            used = list(dict.fromkeys(item["camera_id"] for item in result.get("matches", [])))
            return result, used, None
        if name == "suggest_cross_camera_matches":
            result = suggest_matches(db, args["observation_id"], args.get("limit", 10))
            if result is None: return {"error": "A tracked source observation is required."}, [], None
            used = [result["source"]["camera_id"]] + [item["target"]["camera_id"] for item in result.get("matches", [])]
            return result, list(dict.fromkeys(used)), None
        if name == "list_alert_rules":
            rows = db.query(AlertRule).order_by(AlertRule.created_at).all()
            return {"rules": [row.to_dict() for row in rows]}, list(dict.fromkeys(row.camera_id for row in rows if row.camera_id)), None
        if name == "list_ai_setups":
            return {"profiles": vlm.public_profiles(), "active": vlm.public_settings()}, [], None
        if name == "get_cpu_configuration":
            data = []
            for camera in selected:
                row = db.get(CpuConfig, camera.id)
                data.append({"camera_id": camera.id, "camera_name": camera.name, "settings": CpuSettings(**(row.settings if row else {})).model_dump()})
            return {"cameras": data}, [camera.id for camera in selected], None
        proposal_map = {
            "propose_alert_rule": ("create_alert_rule", f"Create {args.get('source', 'vlm')} alert for {args.get('target', '')}"),
            "propose_update_alert_rule": ("update_alert_rule", f"Change alert rule {args.get('rule_id', '')}"),
            "propose_delete_alert_rule": ("delete_alert_rule", f"Delete alert rule {args.get('rule_id', '')}"),
            "propose_cpu_configuration": ("configure_cpu", f"Change CPU tools for {by_id.get(args.get('camera_id')).name if by_id.get(args.get('camera_id')) else args.get('camera_id', '')}"),
            "propose_rtsp_camera": ("add_rtsp_camera", f"Add RTSP camera {args.get('name', '')}"),
            "propose_update_rtsp_camera": ("update_rtsp_camera", f"Change camera {by_id.get(args.get('camera_id')).name if by_id.get(args.get('camera_id')) else args.get('camera_id', '')}"),
            "propose_delete_camera": ("delete_camera", f"Delete camera {by_id.get(args.get('camera_id')).name if by_id.get(args.get('camera_id')) else args.get('camera_id', '')}"),
            "propose_activate_ai_setup": ("activate_ai_profile", f"Activate AI setup {args.get('profile_id', '')}"),
        }
        if name in proposal_map:
            action, summary = proposal_map[name]
            action_args = dict(args)
            if action == "add_rtsp_camera": action_args["rtsp_url"] = action_args.pop("rtsp_url", "")
            row = create_proposal(db, session_id, action, action_args, summary)
            return {"proposal": row.to_dict(), "requires_confirmation": True}, requested, None
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

    @staticmethod
    def _remember(context, tool, result, used):
        context = dict(context or {})
        if used:
            context["camera_ids"] = list(dict.fromkeys(used))[:20]
        if tool == "search_observations":
            context["observations"] = [{key: item.get(key) for key in ("id", "camera_id", "observed_at", "object_type", "track_id", "global_entity_id")} for item in result.get("observations", [])[:10]]
        elif tool == "search_events":
            context["events"] = [{key: item.get(key) for key in ("id", "camera_id", "created_at", "category", "severity")} for item in result.get("events", [])[:10]]
        elif tool == "get_track_timeline":
            context["track_timeline"] = [{key: item.get(key) for key in ("id", "camera_id", "observed_at", "track_id", "global_entity_id")} for item in result.get("timeline", [])[:20]]
        elif tool in ("find_similar_appearance", "search_visual_description"):
            context["visual_matches"] = [{key: item.get(key) for key in ("observation_id", "camera_id", "observed_at", "object_type", "similarity")} for item in result.get("matches", [])[:10]]
        elif tool == "suggest_cross_camera_matches":
            context["cross_camera_matches"] = [{"association_id": item.get("id"), "target_observation_id": item.get("target_observation_id"), "camera_id": (item.get("target") or {}).get("camera_id"), "similarity": item.get("similarity")} for item in result.get("matches", [])[:10]]
        elif result.get("proposal"):
            context["pending_action"] = {key: result["proposal"].get(key) for key in ("id", "action", "summary", "status")}
        context["last_tool"] = tool
        return context

    def answer(self, question, history, db, registry, cpu_monitor, vlm, on_status=None, on_token=None, session_id="unsaved", session_context=None):
        messages = [{"role": "system", "content": SYSTEM_PROMPT.format(now=datetime.now(timezone.utc).isoformat(), catalog=TOOL_CATALOG)}]
        context = dict(session_context or {})
        if context:
            messages.append({"role": "system", "content": "Structured context retained from this conversation. Resolve follow-ups against these stable IDs:\n" + json.dumps(context, default=str)[:12000]})
        messages.extend({"role": item["role"], "content": item["text"]} for item in history)
        messages.append({"role": "user", "content": question})
        trace, cameras_used, snapshot, pending_actions = [], [], None, []
        active_tools = [LOAD_TOOLS]
        for step in range(self.max_steps):
            message = self._complete(messages, active_tools, on_token)
            calls = message.get("tool_calls") or []
            if not calls:
                content = message.get("content")
                if not isinstance(content, str) or not content.strip():
                    raise AgentUnavailable("Instruction model returned no answer.")
                return {"answer": content.strip(), "cameras_used": list(dict.fromkeys(cameras_used)), "snapshot": snapshot, "tool_trace": trace, "pending_actions": pending_actions, "session_context": context, "steps": step + 1}
            messages.append(message)
            for call in calls:
                function = call.get("function") or {}
                name = function.get("name", "")
                try: args = json.loads(function.get("arguments") or "{}")
                except json.JSONDecodeError: args = {}
                if not isinstance(args, dict):
                    args = {}
                if on_status: on_status(f"Using {name.replace('_', ' ')}…")
                started = time.monotonic()
                status = "completed"
                try:
                    if name == "load_tools":
                        selected = [item for item in args.get("names", []) if item in TOOL_BY_NAME][:4]
                        active_tools = [LOAD_TOOLS, *(TOOL_BY_NAME[item] for item in selected)]
                        result, used, evidence = {"loaded": selected}, [], None
                    elif name not in {item["function"]["name"] for item in active_tools}:
                        result, used, evidence, status = {"error": "Tool is not loaded. Call load_tools first."}, [], None, "error"
                    else:
                        result, used, evidence = self._run_tool(name, args, db, registry, cpu_monitor, vlm, session_id)
                    if result.get("error"):
                        status = "error"
                except Exception as exc:  # keep one failed tool from destroying the conversation
                    log.exception("Agent tool %s failed", name)
                    result, used, evidence, status = {"error": str(exc)}, [], None, "error"
                db.add(AgentToolRun(session_id=session_id, step=step + 1, tool=name or "unknown", arguments=args,
                                    result=result, status=status, duration_ms=round((time.monotonic() - started) * 1000)))
                db.commit()
                trace.append({"tool": name, "arguments": args, "result": result})
                context = self._remember(context, name, result, used)
                if result.get("proposal"):
                    pending_actions.append(result["proposal"])
                cameras_used.extend(used)
                snapshot = snapshot or evidence
                messages.append({"role": "tool", "tool_call_id": call.get("id", f"step-{step}"), "name": name, "content": json.dumps(result, default=str)[:30000]})
        raise AgentUnavailable(f"Agent exceeded its {self.max_steps}-step limit.")


agent_runtime = AgentRuntime()
