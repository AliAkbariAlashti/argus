"""Validation and idempotent execution for operator-approved agent actions."""
from datetime import datetime, timezone
from urllib.parse import urlsplit

from .cpu import CpuSettings
from .models import AgentAction, AlertRule, Camera, CpuConfig
from . import yolo


ACTION_NAMES = {"create_alert_rule", "update_alert_rule", "delete_alert_rule", "configure_cpu", "add_rtsp_camera", "update_rtsp_camera", "delete_camera", "activate_ai_profile"}


def create_proposal(db, session_id, action, arguments, summary):
    if action not in ACTION_NAMES:
        raise ValueError("Unsupported agent action.")
    row = AgentAction(session_id=session_id, action=action, arguments=arguments, summary=summary, status="pending")
    db.add(row); db.commit(); db.refresh(row)
    return row


def _valid_rtsp(value):
    try:
        parsed = urlsplit(value.strip()); parsed.port
        return parsed.scheme in ("rtsp", "rtsps") and bool(parsed.hostname) and not any(c.isspace() for c in value)
    except ValueError:
        return False


def decide_action(db, row, decision, registry, cpu_monitor, vlm=None):
    if row.status != "pending":
        return row  # idempotent retries return the original decision/result
    if decision == "reject":
        row.status = "rejected"; row.decided_at = datetime.now(timezone.utc); db.commit(); db.refresh(row)
        return row
    if decision != "approve":
        raise ValueError("Decision must be approve or reject.")
    args = row.arguments or {}
    try:
        if row.action == "create_alert_rule":
            camera_id = args.get("camera_id") or None
            if camera_id and db.get(Camera, camera_id) is None: raise ValueError("Camera no longer exists.")
            source, target = args.get("source", "vlm"), str(args.get("target", "")).strip()
            if source not in ("cpu", "vlm") or not target: raise ValueError("Invalid alert rule.")
            if source == "cpu" and target.lower() not in yolo.CLASSES: raise ValueError("Unsupported CPU object class.")
            target = target.lower() if source == "cpu" else target
            severity = args.get("severity", "info")
            if severity not in ("info", "warning", "critical"): raise ValueError("Invalid severity.")
            created = AlertRule(camera_id=camera_id, source=source, target=target, severity=severity)
            db.add(created); db.flush(); result = created.to_dict()
        elif row.action == "update_alert_rule":
            rule = db.get(AlertRule, args.get("rule_id"))
            if rule is None: raise ValueError("Alert rule no longer exists.")
            if "enabled" in args: rule.enabled = bool(args["enabled"])
            if "severity" in args:
                if rule.source != "cpu": raise ValueError("Only CPU rules have adjustable severity.")
                if args["severity"] not in ("info", "warning", "critical"): raise ValueError("Invalid severity.")
                rule.severity = args["severity"]
            db.flush(); result = rule.to_dict()
        elif row.action == "delete_alert_rule":
            rule = db.get(AlertRule, args.get("rule_id"))
            if rule is None: raise ValueError("Alert rule no longer exists.")
            result = {"deleted": rule.id, "target": rule.target}; db.delete(rule)
        elif row.action == "configure_cpu":
            camera_id = args.get("camera_id")
            if db.get(Camera, camera_id) is None: raise ValueError("Camera no longer exists.")
            current = db.get(CpuConfig, camera_id)
            settings = CpuSettings(**{**(current.settings if current else {}), **(args.get("settings") or {})}).model_dump()
            if current is None: current = CpuConfig(camera_id=camera_id); db.add(current)
            current.settings = settings; cpu_monitor.reset(camera_id); result = {"camera_id": camera_id, "settings": settings}
        elif row.action == "add_rtsp_camera":
            url = str(args.get("rtsp_url", ""))
            if not _valid_rtsp(url): raise ValueError("Invalid RTSP URL.")
            name = str(args.get("name", "")).strip()
            if not name: raise ValueError("Camera name is required.")
            created = Camera(name=name, location=str(args.get("location", "")), zone_tags=args.get("zone_tags") or [],
                             description=str(args.get("description", "")), source_type="rtsp", source_path=url)
            db.add(created); db.flush(); registry.add(created.id, created.source_path, "rtsp"); result = created.to_dict()
        elif row.action == "update_rtsp_camera":
            camera = db.get(Camera, args.get("camera_id"))
            if camera is None: raise ValueError("Camera no longer exists.")
            if camera.source_type != "rtsp": raise ValueError("The agent can only change RTSP camera sources.")
            source_changed = False
            for field in ("name", "location", "description"):
                if field in args: setattr(camera, field, str(args[field]).strip())
            if "zone_tags" in args: camera.zone_tags = [str(tag).strip() for tag in args["zone_tags"] if str(tag).strip()]
            if "rtsp_url" in args:
                if not _valid_rtsp(str(args["rtsp_url"])): raise ValueError("Invalid RTSP URL.")
                source_changed = camera.source_path != args["rtsp_url"]; camera.source_path = args["rtsp_url"]
            db.flush()
            if source_changed:
                registry.remove(camera.id); registry.add(camera.id, camera.source_path, "rtsp")
            result = camera.to_dict()
        elif row.action == "delete_camera":
            camera = db.get(Camera, args.get("camera_id"))
            if camera is None: raise ValueError("Camera no longer exists.")
            if camera.source_type != "rtsp": raise ValueError("The agent can only delete RTSP cameras.")
            registry.remove(camera.id); db.query(CpuConfig).filter(CpuConfig.camera_id == camera.id).delete()
            cpu_monitor.reset(camera.id); result = {"deleted": camera.id, "name": camera.name}; db.delete(camera)
        elif row.action == "activate_ai_profile":
            if vlm is None: raise ValueError("Vision runtime is unavailable.")
            result = vlm.activate(str(args.get("profile_id", "")))
        else:
            raise ValueError("Unsupported agent action.")
        row.status = "approved"; row.result = result; row.decided_at = datetime.now(timezone.utc); db.commit(); db.refresh(row)
    except Exception as exc:
        db.rollback(); row = db.get(AgentAction, row.id); row.status = "failed"; row.result = {"error": str(exc)}
        row.decided_at = datetime.now(timezone.utc); db.commit(); db.refresh(row)
    return row
