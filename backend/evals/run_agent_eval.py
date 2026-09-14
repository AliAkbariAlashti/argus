#!/usr/bin/env python3
"""Run reproducible black-box checks against a live Argus agent."""
import argparse
import json
import socket
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def request(base, path, method="GET", payload=None, timeout=180):
    body = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(base + path, data=body, method=method, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def stream_answer(base, payload, timeout=300, total_timeout=None):
    body = json.dumps(payload).encode()
    req = urllib.request.Request(base + "/api/chat/stream", data=body, method="POST", headers={"Content-Type": "application/json"})
    event = None
    deadline = time.monotonic() + (total_timeout or timeout)
    with urllib.request.urlopen(req, timeout=timeout) as response:
        for raw in response:
            if time.monotonic() > deadline:
                raise TimeoutError(f"Agent exceeded {total_timeout or timeout}s wall-clock limit.")
            line = raw.decode().rstrip("\r\n")
            if line.startswith("event: "):
                event = line[7:]
            elif line.startswith("data: "):
                data = json.loads(line[6:])
                if event == "answer":
                    return data
                if event == "error":
                    raise RuntimeError(data.get("detail") or "Agent stream failed.")
    raise RuntimeError("Agent stream ended without an answer.")


def assess(name, answer, expected, elapsed):
    tools = [item.get("tool") for item in answer.get("tool_trace", [])]
    failures = []
    if answer.get("scope") != "agent": failures.append(f"scope={answer.get('scope')!r}")
    for tool in expected.get("expected_tools", []):
        if tool not in tools: failures.append(f"missing tool {tool}")
    choices = expected.get("expected_any_tools", [])
    if choices and not any(tool in tools for tool in choices): failures.append(f"missing any tool from {choices}")
    if elapsed > expected.get("max_seconds", 90): failures.append(f"latency {elapsed}s")
    actions = answer.get("pending_actions") or []
    if expected.get("expects_confirmation") and not actions: failures.append("missing confirmation proposal")
    if expected.get("expected_action") and not any(item.get("action") == expected["expected_action"] for item in actions):
        failures.append(f"missing action {expected['expected_action']}")
    for field in expected.get("required_action_arguments", []):
        if not any((item.get("arguments") or {}).get(field) not in (None, "", []) for item in actions):
            failures.append(f"missing action argument {field}")
    for field, value in expected.get("action_argument_contains", {}).items():
        if not any(value.lower() in str((item.get("arguments") or {}).get(field, "")).lower() for item in actions):
            failures.append(f"action argument {field} does not contain {value}")
    answer_text = answer.get("answer", "").strip()
    if not answer_text: failures.append("empty answer")
    for value in expected.get("answer_contains", []):
        if value.lower() not in answer_text.lower(): failures.append(f"answer does not contain {value}")
    for value in expected.get("answer_excludes", []):
        if value.lower() in answer_text.lower(): failures.append(f"answer unexpectedly contains {value}")
    return {"name": name, "passed": not failures, "elapsed_seconds": elapsed,
            "tools": tools, "failures": failures, "answer": answer.get("answer")}


def run(base, scenarios):
    results = []
    status = request(base, "/api/agent/status")
    if not status.get("configured"):
        raise RuntimeError("Argus instruction model is not configured.")
    for scenario in scenarios:
        print(f"Running {scenario['name']}…", file=sys.stderr, flush=True)
        session = request(base, "/api/chat/sessions", "POST", {})
        try:
            turns = scenario.get("turns") or [scenario]
            for index, turn in enumerate(turns, 1):
                started = time.monotonic()
                limit = turn.get("max_seconds", scenario.get("max_seconds", 90))
                answer = stream_answer(base, {"question": turn["question"], "session_id": session["id"]}, total_timeout=limit + 30)
                elapsed = round(time.monotonic() - started, 2)
                name = scenario["name"] if len(turns) == 1 else f"{scenario['name']} / turn {index}"
                results.append(assess(name, answer, {**scenario, **turn}, elapsed))
        except (urllib.error.URLError, TimeoutError, socket.timeout, RuntimeError) as exc:
            results.append({"name": scenario["name"], "passed": False, "failures": [str(exc)]})
        finally:
            try: request(base, f"/api/chat/sessions/{session['id']}", "DELETE")
            except Exception: pass
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8093")
    parser.add_argument("--scenarios", type=Path, default=Path(__file__).with_name("agent_scenarios.json"))
    args = parser.parse_args()
    results = run(args.url.rstrip("/"), json.loads(args.scenarios.read_text()))
    print(json.dumps(results, indent=2))
    raise SystemExit(0 if all(item["passed"] for item in results) else 1)


if __name__ == "__main__":
    main()
