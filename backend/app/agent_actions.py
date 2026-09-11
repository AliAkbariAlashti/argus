"""Validation and idempotent execution for operator-approved agent actions."""
from datetime import datetime, timezone
from urllib.parse import urlsplit

from .cpu import CpuSettings
from .models import AgentAction, AlertRule, Camera, CpuConfig
from . import yolo


ACTION_NAMES = {"create_alert_rule", "configure_cpu", "add_rtsp_camera", "delete_camera"}


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


def decide_action(db, row, decision, registry, cpu_monitor):
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
        elif row.action == "delete_camera":
            camera = db.get(Camera, args.get("camera_id"))
            if camera is None: raise ValueError("Camera no longer exists.")
            if camera.source_type != "rtsp": raise ValueError("The agent can only delete RTSP cameras.")
            registry.remove(camera.id); db.query(CpuConfig).filter(CpuConfig.camera_id == camera.id).delete()
            cpu_monitor.reset(camera.id); result = {"deleted": camera.id, "name": camera.name}; db.delete(camera)
        else:
            raise ValueError("Unsupported agent action.")
        row.status = "approved"; row.result = result; row.decided_at = datetime.now(timezone.utc); db.commit(); db.refresh(row)
    except Exception as exc:
        db.rollback(); row = db.get(AgentAction, row.id); row.status = "failed"; row.result = {"error": str(exc)}
        row.decided_at = datetime.now(timezone.utc); db.commit(); db.refresh(row)
    return row
