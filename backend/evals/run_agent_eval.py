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


def stream_answer(base, payload, timeout=300):
    body = json.dumps(payload).encode()
    req = urllib.request.Request(base + "/api/chat/stream", data=body, method="POST", headers={"Content-Type": "application/json"})
    event = None
    with urllib.request.urlopen(req, timeout=timeout) as response:
        for raw in response:
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


def run(base, scenarios):
    results = []
    status = request(base, "/api/agent/status")
    if not status.get("configured"):
        raise RuntimeError("Argus instruction model is not configured.")
    for scenario in scenarios:
        print(f"Running {scenario['name']}…", file=sys.stderr, flush=True)
        session = request(base, "/api/chat/sessions", "POST", {})
        started = time.monotonic()
        try:
            answer = stream_answer(base, {"question": scenario["question"], "session_id": session["id"]})
            elapsed = round(time.monotonic() - started, 2)
            tools = [item.get("tool") for item in answer.get("tool_trace", [])]
            failures = []
            if answer.get("scope") != "agent": failures.append(f"scope={answer.get('scope')!r}")
            for expected in scenario.get("expected_tools", []):
                if expected not in tools: failures.append(f"missing tool {expected}")
            if elapsed > scenario.get("max_seconds", 90): failures.append(f"latency {elapsed}s")
            if scenario.get("expects_confirmation") and not answer.get("pending_actions"): failures.append("missing confirmation proposal")
            if not answer.get("answer", "").strip(): failures.append("empty answer")
            results.append({"name": scenario["name"], "passed": not failures, "elapsed_seconds": elapsed,
                            "tools": tools, "failures": failures, "answer": answer.get("answer")})
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
