"""Bounded, model-driven Argus agent with typed operational tools."""
import json
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlsplit

import httpx
import cv2
from sqlalchemy import func

from .config import AGENT_API_KEY, AGENT_BASE_URL, AGENT_FAILURE_COOLDOWN_SECONDS, AGENT_MAX_STEPS, AGENT_MODEL, AGENT_TIMEOUT_SECONDS, VIDEOS_DIR
from .agent_actions import create_proposal
from .cpu import CpuSettings
from .imaging import frame_to_pil, image_to_data_uri
from .models import AgentToolRun, AlertRule, Camera, CpuConfig, Event, Observation
from .entity_matching import suggest_matches
from .visual_search import find_by_text, find_similar

log = logging.getLogger("qwenvl.agent_runtime")

SYSTEM_PROMPT = """You are Argus, a CCTV operator. Speak naturally. Use tools for camera, scene, health, event, rule, and configuration facts. First call load_tools with up to four names from the catalog; load more later if needed. Never invent evidence or IDs. Include camera names and times when known. Changes create confirmation proposals. UTC: {now}.\nTools: {catalog}"""
OLLAMA_SYSTEM_PROMPT = """You are Argus, a CCTV operator. Speak naturally. For camera, scene, health, event, rule, and configuration facts, select an exact catalog tool. Never guess or ask for IDs when camera_ids is optional. Requests to create, change, activate, or delete must use a propose_* tool; list_* tools only read. 'List available recordings with duration or FPS' always means list_recordings. Current, live, now, or latest-frame inspection always means inspect_live_cameras, never inspect_recording. inspect_recording is only for an uploaded archive and requires one explicit camera plus numeric offsets. If a tool result follows, answer directly from it with camera names and times when known. Changes create confirmation proposals. UTC: {now}.\nTools: {catalog}"""


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
    {"type": "function", "function": {"name": "propose_alert_rule", "description": "Prepare an alert rule for operator confirmation. This does not change the system.", "parameters": {"type": "object", "required": ["target"], "properties": {"camera_id": {"type": "string", "description": "Exact camera ID; omit for all cameras."}, "source": {"type": "string", "enum": ["cpu", "vlm"], "description": "Use CPU for one detectable object class; VLM for a described visual condition."}, "target": {"type": "string", "description": "The object or visual condition to detect, such as 'person carrying a backpack'."}, "severity": {"type": "string", "enum": ["info", "warning", "critical"]}}, "additionalProperties": False}}},
    {"type": "function", "function": {"name": "propose_update_alert_rule", "description": "Prepare enabling, disabling, or changing severity of an alert rule for confirmation.", "parameters": {"type": "object", "required": ["rule_id"], "properties": {"rule_id": {"type": "string"}, "enabled": {"type": "boolean"}, "severity": {"type": "string", "enum": ["info", "warning", "critical"]}}, "additionalProperties": False}}},
    {"type": "function", "function": {"name": "propose_delete_alert_rule", "description": "Prepare deleting an alert rule for confirmation.", "parameters": {"type": "object", "required": ["rule_id"], "properties": {"rule_id": {"type": "string"}}, "additionalProperties": False}}},
    {"type": "function", "function": {"name": "propose_cpu_configuration", "description": "Prepare a partial CPU-tool configuration change for confirmation. This does not change the system.", "parameters": {"type": "object", "required": ["camera_id", "settings"], "properties": {"camera_id": {"type": "string"}, "settings": {"type": "object", "properties": {"enabled": {"type": "boolean"}, "motion_alerts": {"type": "boolean"}, "quality_alerts": {"type": "boolean"}, "object_alerts": {"type": "boolean"}, "tracking_enabled": {"type": "boolean"}, "line_crossing_alerts": {"type": "boolean"}, "dwell_alerts": {"type": "boolean"}, "dwell_seconds": {"type": "integer", "minimum": 10, "maximum": 86400}, "object_confidence": {"type": "number", "minimum": 0.1, "maximum": 0.95}}, "additionalProperties": False}}, "additionalProperties": False}}},
    {"type": "function", "function": {"name": "propose_rtsp_camera", "description": "Prepare adding an RTSP camera for confirmation. This does not change the system.", "parameters": {"type": "object", "required": ["name", "rtsp_url"], "properties": {"name": {"type": "string"}, "rtsp_url": {"type": "string"}, "location": {"type": "string"}, "zone_tags": {"type": "array", "items": {"type": "string"}}, "description": {"type": "string"}}, "additionalProperties": False}}},
    {"type": "function", "function": {"name": "propose_update_rtsp_camera", "description": "Prepare changing an RTSP camera name, location, tags, description, or URL for confirmation.", "parameters": {"type": "object", "required": ["camera_id"], "properties": {"camera_id": {"type": "string"}, "name": {"type": "string"}, "rtsp_url": {"type": "string"}, "location": {"type": "string"}, "zone_tags": {"type": "array", "items": {"type": "string"}}, "description": {"type": "string"}}, "additionalProperties": False}}},
    {"type": "function", "function": {"name": "propose_delete_camera", "description": "Prepare deleting an RTSP camera for confirmation. This does not change the system.", "parameters": {"type": "object", "required": ["camera_id"], "properties": {"camera_id": {"type": "string"}}, "additionalProperties": False}}},
    {"type": "function", "function": {"name": "propose_activate_ai_setup", "description": "Prepare activating a saved vision-model setup for confirmation.", "parameters": {"type": "object", "required": ["profile_id"], "properties": {"profile_id": {"type": "string"}}, "additionalProperties": False}}},
    {"type": "function", "function": {"name": "inspect_live_cameras", "description": "Ask the vision model to inspect current frames. Use only when pixels must be examined, after identifying relevant cameras.", "parameters": {"type": "object", "required": ["camera_ids", "question"], "properties": {"camera_ids": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 4}, "question": {"type": "string", "maxLength": 1000}}, "additionalProperties": False}}},
    {"type": "function", "function": {"name": "list_recordings", "description": "List all uploaded camera recordings with duration, frame rate, and availability.", "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}},
    {"type": "function", "function": {"name": "inspect_recording", "description": "Inspect explicit timestamps from one uploaded archive recording; never use for current or live frames.", "parameters": {"type": "object", "required": ["camera_id", "offset_seconds", "question"], "properties": {"camera_id": {"type": "string"}, "offset_seconds": {"type": "array", "items": {"type": "number", "minimum": 0}, "minItems": 1, "maxItems": 6}, "question": {"type": "string", "maxLength": 1000}}, "additionalProperties": False}}},
]

TOOL_BY_NAME = {item["function"]["name"]: item for item in TOOLS}


def _tool_hint(item):
    function = item["function"]
    parameters = function.get("parameters") or {}
    required = set(parameters.get("required") or [])
    arguments = ",".join(name if name in required else f"{name}?" for name in (parameters.get("properties") or {}))
    return f"{function['name']}({arguments}): {function.get('description', '')}"


TOOL_CATALOG = "; ".join(_tool_hint(item) for item in TOOLS)
LOAD_TOOLS = {"type": "function", "function": {"name": "load_tools", "description": "Load schemas by exact names from the system catalog.", "parameters": {"type": "object", "required": ["names"], "properties": {"names": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 4}}, "additionalProperties": False}}}
CALL_ARGUS_TOOL = {"type": "function", "function": {"name": "call_argus_tool", "description": "Call one exact tool from the system catalog.", "parameters": {"type": "object", "required": ["name", "arguments"], "properties": {"name": {"type": "string"}, "arguments": {"type": "object"}}, "additionalProperties": False}}}
OLLAMA_PLAN_FORMAT = {"type": "object", "properties": {
    "tool": {"type": "string", "enum": ["", *TOOL_BY_NAME]},
    "answer": {"type": "string", "maxLength": 1200},
}, "required": ["tool", "answer"], "additionalProperties": False}


class AgentUnavailable(RuntimeError):
    pass


class AgentRuntime:
    def __init__(self, base_url=AGENT_BASE_URL, model=AGENT_MODEL, api_key=AGENT_API_KEY, max_steps=AGENT_MAX_STEPS, path=None):
        self.path = Path(path or os.environ.get("ARGUS_AGENT_RUNTIME_FILE", Path(__file__).resolve().parents[1] / "data" / "agent.json"))
        if self.path.exists():
            saved = json.loads(self.path.read_text())
            base_url, model, api_key, max_steps = saved.get("base_url", base_url), saved.get("model", model), saved.get("api_key", api_key), saved.get("max_steps", max_steps)
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.max_steps = max_steps
        self._last_error = None
        self._last_failure_at = 0.0
        self._last_success_at = None

    @property
    def configured(self):
        return bool(self.base_url and self.model)

    def public_status(self):
        cooling_down = bool(self._last_failure_at and time.monotonic() - self._last_failure_at < AGENT_FAILURE_COOLDOWN_SECONDS)
        return {"configured": self.configured, "ready": self.configured and bool(self._last_success_at) and not cooling_down,
                "base_url": self.base_url, "model": self.model, "max_steps": self.max_steps,
                "has_api_key": bool(self.api_key), "last_error": self._last_error,
                "last_success_at": self._last_success_at, "cooling_down": cooling_down}

    def configure(self, *, base_url, model, api_key=None, max_steps=None, allow_remote=False):
        base_url, model = base_url.strip().rstrip("/"), model.strip()
        parsed = urlsplit(base_url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("Use an HTTP(S) API base URL without credentials, query, or fragment.")
        if not model:
            raise ValueError("Enter an instruction model name.")
        local = parsed.hostname in ("localhost", "127.0.0.1", "::1", "host.docker.internal")
        if not local and not allow_remote:
            raise ValueError("Confirm that camera metadata and event records may be sent to this server.")
        self.base_url, self.model = base_url, model
        if api_key is not None: self.api_key = api_key
        if max_steps is not None: self.max_steps = max(1, min(int(max_steps), 16))
        self._last_error, self._last_failure_at = None, 0.0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as output:
            json.dump({"base_url": self.base_url, "model": self.model, "api_key": self.api_key, "max_steps": self.max_steps}, output)
        temporary.replace(self.path)
        return self.public_status()

    def _url(self):
        base = self.base_url[:-3] if self.base_url.endswith("/v1") else self.base_url
        return base + "/v1/chat/completions"

    def _is_ollama(self):
        return "ollama" in self.base_url.lower() or any(port in self.base_url for port in (":11434", ":11435"))

    @staticmethod
    def _camera_bindings(messages, schema):
        properties = schema.get("properties") or {}
        if not ({"camera_id", "camera_ids"} & set(properties)):
            return {}
        question = next((str(message.get("content") or "") for message in reversed(messages)
                         if message.get("role") == "user"), "")
        directory = []
        for message in messages:
            content = str(message.get("content") or "")
            if message.get("role") == "system" and content.startswith("Camera directory"):
                try:
                    directory = json.loads(content.split("\n", 1)[1])
                except (IndexError, json.JSONDecodeError):
                    directory = []
        lowered = question.casefold()
        matched = []
        for camera in directory:
            camera_id, camera_name = str(camera.get("id") or ""), str(camera.get("name") or "")
            if (camera_id and camera_id.casefold() in lowered) or (camera_name and camera_name.casefold() in lowered):
                if camera_id not in matched:
                    matched.append(camera_id)
        if "camera_ids" in properties and matched:
            return {"camera_ids": matched}
        if "camera_id" in properties and len(matched) == 1:
            return {"camera_id": matched[0]}
        return {}

    def _ollama_extract_arguments(self, messages, tool):
        function = TOOL_BY_NAME[tool]["function"]
        schema = function.get("parameters") or {"type": "object", "properties": {}}
        if not schema.get("properties"):
            return {}
        camera_bindings = self._camera_bindings(messages, schema)
        non_camera_fields = set(schema.get("properties") or {}) - {"camera_id", "camera_ids"}
        if not non_camera_fields:
            return camera_bindings
        context = [message.get("content", "") for message in messages[-8:]
                   if message.get("role") == "user" or (message.get("role") == "system" and
                      message.get("content", "").startswith(("Structured context", "Camera directory")))]
        prompt = (f"Extract arguments for {tool} from the request. {function.get('description', '')} "
                  "Return only fields defined by the JSON schema. Use only information stated by the user or retained context. "
                  "Omit optional values that are absent. camera_id and camera_ids accept an exact camera ID from Structured context "
                  "or an exact camera name stated by the user; never invent references or put 'all cameras' or other scope phrases "
                  "in them. If all cameras are requested, "
                  "omit the optional camera field. For propose_alert_rule, target is only the object or visual condition to detect; "
                  "copy that condition from the request and never use camera scope as target. Example: 'a person carrying a backpack "
                  "on all cameras' has target 'person carrying a backpack' and no camera_id.")
        base = self.base_url[:-3] if self.base_url.endswith("/v1") else self.base_url
        response = httpx.post(base + "/api/chat", headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}, json={
            "model": self.model, "messages": [{"role": "system", "content": prompt}, {"role": "user", "content": "\n".join(context)}],
            "format": schema, "stream": False, "think": False,
            "options": {"temperature": 0, "num_predict": 96},
        }, timeout=AGENT_TIMEOUT_SECONDS)
        response.raise_for_status()
        arguments = json.loads((response.json().get("message") or {}).get("content") or "{}")
        if not isinstance(arguments, dict):
            return {}
        grounding_text = "\n".join(context).lower()
        fleet_scopes = {"all", "all cameras", "every camera", "every cameras", "any camera"}
        if "camera_id" in arguments:
            camera_ref = str(arguments["camera_id"]).strip().casefold()
            if camera_ref in fleet_scopes or camera_ref not in grounding_text:
                arguments.pop("camera_id")
        if "camera_ids" in arguments:
            grounded = [value for value in arguments["camera_ids"]
                        if isinstance(value, str) and value.strip().casefold() not in fleet_scopes
                        and value.lower() in grounding_text]
            if grounded:
                arguments["camera_ids"] = grounded
            else:
                arguments.pop("camera_ids")
        arguments.update(camera_bindings)
        return arguments

    @staticmethod
    def _ollama_messages(messages):
        native = []
        for original in messages:
            message = dict(original)
            if message.get("role") == "assistant" and message.get("tool_calls"):
                calls = []
                for original_call in message["tool_calls"]:
                    function = dict(original_call.get("function") or {})
                    arguments = function.get("arguments")
                    if isinstance(arguments, str):
                        try: function["arguments"] = json.loads(arguments)
                        except json.JSONDecodeError: function["arguments"] = {}
                    calls.append({"function": function})
                message = {"role": "assistant", "content": message.get("content") or "", "tool_calls": calls}
            elif message.get("role") == "tool":
                message = {"role": "tool", "content": message.get("content") or "", "tool_name": message.get("name") or ""}
            native.append(message)
        return native

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
        if self._last_failure_at and time.monotonic() - self._last_failure_at < AGENT_FAILURE_COOLDOWN_SECONDS:
            raise AgentUnavailable(self._last_error or "Instruction model is cooling down after a failure.")
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        json_plan = self._is_ollama() and len(tools or []) == 1 and (tools or [])[0]["function"]["name"] == "call_argus_tool"
        stream_response = (on_token is not None or self._is_ollama()) and not json_plan
        payload = {"model": self.model, "messages": messages, "tools": tools if tools is not None else [LOAD_TOOLS], "stream": stream_response}
        url = self._url()
        if self._is_ollama():
            base = self.base_url[:-3] if self.base_url.endswith("/v1") else self.base_url
            url = base + "/api/chat"
            payload["messages"] = self._ollama_messages(messages)
            payload.update({"think": False, "options": {"temperature": 0 if json_plan else 0.1, "num_predict": 96 if json_plan else 160}})
            if json_plan:
                payload.pop("tools", None)
                payload["format"] = OLLAMA_PLAN_FORMAT
        else:
            payload.update({"tool_choice": "auto", "temperature": 0.1, "max_tokens": 512, "reasoning_effort": "none"})
        try:
            if not stream_response:
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
                            content.append(chunk)
                            if on_token: on_token(chunk)
                        for call_delta in delta.get("tool_calls") or []:
                            self._merge_tool_delta(calls, call_delta)
                message = {"role": "assistant", "content": "".join(content), "tool_calls": calls}
            if json_plan:
                plan = json.loads(message.get("content") or "{}")
                tool = plan.get("tool") if plan.get("tool") in TOOL_BY_NAME else ""
                if tool:
                    arguments = self._ollama_extract_arguments(messages, tool)
                    message = {"role": "assistant", "content": "", "tool_calls": [{"id": f"local-{time.time_ns()}", "type": "function", "function": {"name": "call_argus_tool", "arguments": json.dumps({"name": tool, "arguments": arguments})}}]}
                else:
                    message = {"role": "assistant", "content": str(plan.get("answer") or "")}
            for call in message.get("tool_calls") or []:
                arguments = (call.get("function") or {}).get("arguments")
                if isinstance(arguments, dict):
                    call["function"]["arguments"] = json.dumps(arguments)
            self._last_error = None
            self._last_failure_at = 0.0
            self._last_success_at = datetime.now(timezone.utc).isoformat()
            return message
        except (httpx.HTTPError, KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            self._last_error = f"Instruction model unavailable: {exc}"
            self._last_failure_at = time.monotonic()
            raise AgentUnavailable(self._last_error) from None

    @staticmethod
    def _since(query, column, minutes):
        return query.filter(column >= datetime.now(timezone.utc) - timedelta(minutes=minutes)) if minutes else query

    @staticmethod
    def _model_result(name, result):
        if name == "get_camera_health":
            cameras = []
            for camera in result.get("cameras", []):
                cpu = camera.get("cpu_status") or {}
                flags = [key for key in ("motion", "low_light", "low_detail") if cpu.get(key)]
                cameras.append([camera.get("camera_id"), camera.get("name"), bool(camera.get("online")),
                                round(camera.get("fps") or 0, 1), bool(cpu.get("online")), flags,
                                cpu.get("tracked_counts") or {}, cpu.get("message")])
            return {"columns": ["id", "name", "online", "fps", "cpu_online", "flags", "objects", "message"], "cameras": cameras}
        return result

    @staticmethod
    def _operational_answer(name, result):
        if result.get("error"):
            return f"I could not complete that check: {result['error']}"
        if result.get("proposal"):
            return f"I prepared this change: {result['proposal'].get('summary', 'configuration update')}. Review and approve the confirmation card to apply it."
        if isinstance(result.get("answer"), str):
            return result["answer"]
        if name == "list_cameras":
            rows = result.get("cameras", [])
            lines = [f"{len(rows)} cameras are configured:"]
            lines.extend(f"- {row['name']} ({row.get('location') or 'No location'}) — {'online' if row.get('online') else 'offline'}" for row in rows)
            return "\n".join(lines)
        if name == "get_camera_health":
            rows = result.get("cameras", [])
            online = sum(bool(row.get("online")) for row in rows)
            lines = [f"{online} of {len(rows)} cameras are online."]
            for row in rows:
                cpu = row.get("cpu_status") or {}
                state = "online" if row.get("online") else "offline"
                detail = f"{round(row.get('fps') or 0, 1)} FPS" if row.get("online") else cpu.get("message") or "needs attention"
                lines.append(f"- {row['name']} — {state}; {detail}")
            return "\n".join(lines)
        if name == "get_activity_report":
            return (f"Since {result.get('since')}, Argus recorded {result.get('total_observations', 0)} observations "
                    f"and {result.get('total_events', 0)} events. Observation types: {result.get('observation_counts') or {}}. "
                    f"Event severities: {result.get('event_counts') or {}}.")
        if name == "list_alert_rules":
            rows = result.get("rules", [])
            if not rows: return "No alert rules are configured."
            return "Alert rules:\n" + "\n".join(f"- {row.get('target')} — {row.get('source')} / {row.get('severity')} / {'enabled' if row.get('enabled') else 'disabled'}" for row in rows)
        if name == "list_recordings":
            rows = result.get("recordings", [])
            if not rows: return "No uploaded camera recordings are available."
            return "Recordings:\n" + "\n".join(f"- {row['camera_name']} — {row.get('duration_seconds')} seconds at {row.get('fps')} FPS" if row.get("available") else f"- {row['camera_name']} — unavailable" for row in rows)
        return None

    @staticmethod
    def _resolve_camera_arguments(args, cameras):
        """Resolve model-provided camera names to stable IDs before tool execution."""
        normalized = dict(args or {})
        references = {}
        for camera in cameras:
            references.setdefault(str(camera.id).strip().casefold(), []).append(camera.id)
            references.setdefault(str(camera.name).strip().casefold(), []).append(camera.id)

        def resolve(value):
            matches = references.get(str(value).strip().casefold(), [])
            return matches[0] if len(set(matches)) == 1 else None

        if normalized.get("camera_id"):
            raw = normalized["camera_id"]
            resolved = resolve(raw)
            if resolved is None:
                return normalized, f"Camera '{raw}' was not found or its name is ambiguous."
            normalized["camera_id"] = resolved
        if normalized.get("camera_ids"):
            resolved_ids = []
            for raw in normalized["camera_ids"]:
                resolved = resolve(raw)
                if resolved is None:
                    return normalized, f"Camera '{raw}' was not found or its name is ambiguous."
                if resolved not in resolved_ids:
                    resolved_ids.append(resolved)
            normalized["camera_ids"] = resolved_ids
        return normalized, None

    def _run_tool(self, name, args, db, registry, cpu_monitor, vlm, session_id="unsaved"):
        cameras = db.query(Camera).order_by(Camera.created_at).all()
        args, camera_error = self._resolve_camera_arguments(args, cameras)
        if camera_error:
            return {"error": camera_error}, [], None
        definition = TOOL_BY_NAME.get(name)
        required = ((definition or {}).get("function", {}).get("parameters", {}).get("required") or [])
        missing = [field for field in required if args.get(field) in (None, "", [])]
        if missing:
            return {"error": f"Missing required tool arguments: {', '.join(missing)}."}, [], None
        by_id = {camera.id: camera for camera in cameras}
        requested = args.get("camera_ids") or []
        selected = [by_id[camera_id] for camera_id in requested] if requested else cameras
        if name == "list_cameras":
            data = [{"id": camera.id, "name": camera.name, "location": camera.location,
                     "zone_tags": camera.zone_tags, "source_type": camera.source_type,
                     "online": bool(registry.get(camera.id) and registry.get(camera.id).is_online())} for camera in cameras]
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
        if name == "list_recordings":
            recordings = []
            for camera in selected:
                if camera.source_type != "file":
                    continue
                path = (VIDEOS_DIR / camera.source_path).resolve()
                available = path.is_file() and path.parent == VIDEOS_DIR.resolve()
                fps = frames = 0.0
                if available:
                    capture = cv2.VideoCapture(str(path))
                    try:
                        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
                        frames = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                    finally:
                        capture.release()
                recordings.append({"camera_id": camera.id, "camera_name": camera.name, "available": available,
                                   "duration_seconds": round(frames / fps, 2) if fps > 0 else None,
                                   "fps": round(fps, 2) if fps > 0 else None})
            return {"recordings": recordings}, [item["camera_id"] for item in recordings], None
        if name == "inspect_recording":
            camera = by_id.get(args.get("camera_id"))
            if camera is None or camera.source_type != "file":
                return {"error": "Choose a camera with an uploaded recording."}, [], None
            if not vlm.ready:
                return {"error": vlm.error or "Vision model is not ready."}, [], None
            path = (VIDEOS_DIR / camera.source_path).resolve()
            if path.parent != VIDEOS_DIR.resolve() or not path.is_file():
                return {"error": "Recording file is unavailable."}, [], None
            capture = cv2.VideoCapture(str(path))
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
            frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            duration = frame_count / fps if fps > 0 else 0
            images, offsets = [], []
            try:
                for raw_offset in args.get("offset_seconds", [])[:6]:
                    offset = float(raw_offset)
                    if duration and offset > duration:
                        continue
                    capture.set(cv2.CAP_PROP_POS_MSEC, offset * 1000)
                    ok, frame = capture.read()
                    if ok and frame is not None:
                        images.append(frame_to_pil(frame)); offsets.append(round(offset, 2))
            finally:
                capture.release()
            if not images:
                return {"error": "No frames could be read at those recording timestamps.", "duration_seconds": round(duration, 2)}, [], None
            prompt = f"Frames are from {camera.name} at offsets {offsets} seconds, in order. Answer only from these recording frames. {args['question']}"
            answer = vlm.ask(images, prompt, max_new_tokens=512)
            return {"answer": answer, "camera_id": camera.id, "camera_name": camera.name,
                    "offset_seconds": offsets, "duration_seconds": round(duration, 2)}, [camera.id], image_to_data_uri(images[0])
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
        prompt = OLLAMA_SYSTEM_PROMPT if self._is_ollama() else SYSTEM_PROMPT
        messages = [{"role": "system", "content": prompt.format(now=datetime.now(timezone.utc).isoformat(), catalog=TOOL_CATALOG)}]
        if hasattr(db, "query"):
            cameras = db.query(Camera).order_by(Camera.created_at).all()
            directory = [{"id": camera.id, "name": camera.name, "location": camera.location} for camera in cameras]
            messages.append({"role": "system", "content": "Camera directory. Resolve camera names to these exact stable IDs:\n" + json.dumps(directory, default=str)[:6000]})
        context = dict(session_context or {})
        if context:
            messages.append({"role": "system", "content": "Structured context retained from this conversation. Resolve follow-ups against these stable IDs:\n" + json.dumps(context, default=str)[:12000]})
        messages.extend({"role": item["role"], "content": item["text"]} for item in history)
        messages.append({"role": "user", "content": question})
        trace, cameras_used, snapshot, pending_actions = [], [], None, []
        active_tools = [CALL_ARGUS_TOOL] if self._is_ollama() else [LOAD_TOOLS]
        for step in range(self.max_steps):
            message = self._complete(messages, active_tools, on_token)
            calls = message.get("tool_calls") or []
            if not calls:
                content = message.get("content")
                if not isinstance(content, str) or not content.strip():
                    raise AgentUnavailable("Instruction model returned no answer.")
                return {"answer": content.strip(), "cameras_used": list(dict.fromkeys(cameras_used)), "snapshot": snapshot, "tool_trace": trace, "pending_actions": pending_actions, "session_context": context, "steps": step + 1}
            messages.append(message)
            used_generic_dispatch = False
            operational_answer = None
            for call in calls:
                function = call.get("function") or {}
                wire_name = function.get("name", "")
                try: args = json.loads(function.get("arguments") or "{}")
                except json.JSONDecodeError: args = {}
                if not isinstance(args, dict):
                    args = {}
                name = wire_name
                if wire_name == "call_argus_tool":
                    used_generic_dispatch = True
                    name = args.get("name", "")
                    args = args.get("arguments") if isinstance(args.get("arguments"), dict) else {}
                if on_status: on_status(f"Using {name.replace('_', ' ')}…")
                started = time.monotonic()
                status = "completed"
                try:
                    if name == "load_tools":
                        selected = [item for item in args.get("names", []) if item in TOOL_BY_NAME][:4]
                        active_tools = [LOAD_TOOLS, *(TOOL_BY_NAME[item] for item in selected)]
                        result, used, evidence = {"loaded": selected}, [], None
                    elif wire_name != "call_argus_tool" and name not in {item["function"]["name"] for item in active_tools}:
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
                model_result = self._model_result(name, result)
                messages.append({"role": "tool", "tool_call_id": call.get("id", f"step-{step}"), "name": wire_name, "content": json.dumps(model_result, default=str)[:6000]})
                if used_generic_dispatch:
                    operational_answer = self._operational_answer(name, result)
            if used_generic_dispatch:
                if operational_answer:
                    if on_token: on_token(operational_answer)
                    return {"answer": operational_answer, "cameras_used": list(dict.fromkeys(cameras_used)), "snapshot": snapshot,
                            "tool_trace": trace, "pending_actions": pending_actions, "session_context": context, "steps": step + 1}
        raise AgentUnavailable(f"Agent exceeded its {self.max_steps}-step limit.")


agent_runtime = AgentRuntime()
