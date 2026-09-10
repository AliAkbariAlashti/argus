"""Small CPU object tracker and tripwire logic.

The tracker deliberately uses only detections already produced by YOLO. It is
not identity recognition: an ID is an in-memory label that survives nearby
detections for a short gap so the UI can show dwell time and line crossings.
"""
from collections import defaultdict
from datetime import datetime, timezone
from math import hypot
import uuid


def _center(box):
    x, y, width, height = box
    return x + width / 2, y + height / 2


def _iou(left, right):
    lx, ly, lw, lh = left
    rx, ry, rw, rh = right
    x1, y1 = max(lx, rx), max(ly, ry)
    x2, y2 = min(lx + lw, rx + rw), min(ly + lh, ry + rh)
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = lw * lh + rw * rh - intersection
    return intersection / union if union else 0.0


def _side(point, line):
    """Signed side of a point relative to the configured A -> B line."""
    x1, y1 = line["x1"], line["y1"]
    x2, y2 = line["x2"], line["y2"]
    return (x2 - x1) * (point[1] - y1) - (y2 - y1) * (point[0] - x1)


def _side_sign(value, deadband=0.015):
    if abs(value) <= deadband:
        return 0
    return 1 if value > 0 else -1


class ObjectTracker:
    """Greedy nearest-neighbour tracker for normalized YOLO boxes."""

    def __init__(self, max_missed_ticks=2, max_center_distance=0.18):
        self.max_missed_ticks = max_missed_ticks
        self.max_center_distance = max_center_distance
        self._tracks = defaultdict(dict)
        self._next_id = defaultdict(int)
        self._crossings = defaultdict(lambda: {"A_to_B": 0, "B_to_A": 0, "total": 0})

    def reset(self, camera_id):
        self._tracks.pop(camera_id, None)
        self._crossings.pop(camera_id, None)

    def counts(self, camera_id):
        return dict(self._crossings[camera_id])

    def _new_track(self, camera_id, detection, now, line):
        self._next_id[camera_id] += 1
        track_id = self._next_id[camera_id]
        box = detection["box"]
        side = _side_sign(_side(_center(box), line)) if line else 0
        self._tracks[camera_id][track_id] = {
            "track_id": track_id,
            "track_uid": uuid.uuid4().hex,
            "class": detection["class"],
            "box": box,
            "confidence": detection.get("confidence"),
            "center": _center(box),
            "first_seen": now,
            "last_seen": now,
            "missed": 0,
            "side": side,
            "dwell_alerted": False,
        }
        return self._tracks[camera_id][track_id]

    def _view(self, camera_id, now):
        return [
            {
                "track_id": track["track_id"],
                "track_uid": track["track_uid"],
                "class": track["class"],
                "box": track["box"],
                "confidence": track["confidence"],
                "dwell_seconds": round(max(0, now - track["first_seen"]), 1),
                "active": track["missed"] == 0,
                "last_seen_seconds": round(max(0, now - track["last_seen"]), 1),
                "first_seen": datetime.fromtimestamp(track["first_seen"], timezone.utc).isoformat(),
            }
            for track in sorted(self._tracks[camera_id].values(), key=lambda item: item["track_id"])
        ]

    def update(self, camera_id, detections, now, line=None, dwell_seconds=None):
        tracks = self._tracks[camera_id]
        unmatched_tracks = set(tracks)
        crossings = []
        dwell_events = []

        # Match strongest detections first. A class match plus nearby center or
        # overlapping box is enough for a short sampled-video interval.
        for detection_index in sorted(
            range(len(detections)),
            key=lambda index: detections[index].get("confidence", 0),
            reverse=True,
        ):
            detection = detections[detection_index]
            center = _center(detection["box"])
            candidates = []
            for track_id in unmatched_tracks:
                track = tracks[track_id]
                if track["class"] != detection["class"]:
                    continue
                distance = hypot(center[0] - track["center"][0], center[1] - track["center"][1])
                overlap = _iou(detection["box"], track["box"])
                if distance <= self.max_center_distance or overlap >= 0.05:
                    candidates.append((distance - overlap * 0.25, track_id))
            if candidates:
                _, track_id = min(candidates)
                track = tracks[track_id]
                unmatched_tracks.remove(track_id)
            else:
                track = self._new_track(camera_id, detection, now, line)
                track_id = track["track_id"]

            previous_side = track["side"]
            new_side = _side_sign(_side(center, line)) if line else 0
            if line and previous_side and new_side and previous_side != new_side:
                # Side A is the positive side of the oriented line and side B
                # is the negative side. For a vertical top-to-bottom line,
                # this reads naturally as left-to-right: A_to_B.
                direction = "A_to_B" if previous_side > new_side else "B_to_A"
                self._crossings[camera_id][direction] += 1
                self._crossings[camera_id]["total"] += 1
                crossings.append({
                    "track_id": track_id,
                    "track_uid": track["track_uid"],
                    "class": detection["class"],
                    "direction": direction,
                })
            if new_side:
                track["side"] = new_side

            track.update({
                "box": detection["box"],
                "confidence": detection.get("confidence"),
                "center": center,
                "last_seen": now,
                "missed": 0,
            })
            detection["track_id"] = track_id
            detection["track_uid"] = track["track_uid"]
            detection["dwell_seconds"] = round(max(0, now - track["first_seen"]), 1)
            if dwell_seconds is not None and detection["dwell_seconds"] >= dwell_seconds and not track["dwell_alerted"]:
                track["dwell_alerted"] = True
                dwell_events.append({
                    "track_id": track_id,
                    "track_uid": track["track_uid"],
                    "class": detection["class"],
                    "dwell_seconds": detection["dwell_seconds"],
                })

        for track_id in list(unmatched_tracks):
            tracks[track_id]["missed"] += 1
            if tracks[track_id]["missed"] > self.max_missed_ticks:
                tracks.pop(track_id, None)

        view = self._view(camera_id, now)
        counts = {}
        for track in view:
            if not track["active"]:
                continue
            counts[track["class"]] = counts.get(track["class"], 0) + 1
        return view, counts, crossings, dwell_events
