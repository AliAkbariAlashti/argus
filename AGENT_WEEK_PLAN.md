# Argus Agent delivery plan

Delivery checkpoint: September 18, 2026.

## Product result

Argus Agent is a conversational camera operator backed by a local instruction
LLM. It can plan multi-step work, call typed Argus tools, inspect live and
historical visual evidence, ask the VLM for selective verification, maintain
context inside independent chat sessions, and safely perform approved changes.

## Work sequence

### 1. Agent model runtime

- Add a separate local instruction-LLM configuration; do not reuse the VLM role.
- Support Ollama and generic OpenAI-compatible chat endpoints.
- Add readiness and connection tests that verify structured tool-call output.
- Keep deterministic and camera features available when the agent model is down.

### 2. Typed tool registry

- Read tools: camera inventory and health, live-frame inspection, events,
  observations, reports, visual search, track timelines and evidence retrieval.
- Action tools: alert-rule management, CPU configuration and camera management.
- Validate every argument with schemas; never execute model-generated SQL.
- Return compact structured results with stable IDs for follow-up turns.

### 3. Bounded reasoning loop

- Give the instruction LLM tool descriptions and current session context.
- Execute one tool call at a time and return its result to the model.
- Allow multi-step planning with limits on iterations, duration, rows, frames and
  VLM calls.
- Stream tool activity and the final answer to the client.
- Preserve deterministic fast paths only as tools and reliability fallbacks.

### 4. Safe operations

- Execute read-only tools immediately.
- Persist proposed mutations before execution.
- Show concrete confirmation cards for rule, camera and CPU changes.
- Record tool calls, arguments, results, errors and operator decisions in an
  audit log.
- Make retries idempotent.

### 5. Conversation state

- Keep independent persistent chat sessions and titles.
- Store selected cameras, time ranges, entities, evidence and pending actions as
  structured session state.
- Support follow-ups such as “the second one,” “before that,” and “make a rule
  for this.”

### 6. Evaluation and hardening

- Build scenarios for live inspection, historical search, cross-camera tracing,
  health diagnosis, reports and configuration actions.
- Check tool selection, factual correctness, evidence attribution, mutation
  safety, latency and recovery from unavailable models/tools.
- Run with multiple cameras and long conversations.
- Tune prompts and limits from failures before considering fine-tuning.

## Acceptance scenarios

1. “Check every entrance now and tell me what needs attention.”
2. “Find the person who entered Reception and show where they may have gone.”
3. “Summarize yesterday’s important activity with evidence.”
4. “Why is the Lobby camera offline?”
5. “Create a warning when a vehicle stays at Loading for five minutes.”
6. “Enable tracking on entrance cameras,” followed by review and confirmation.
7. “Show the second candidate,” retaining the previous result context.

The delivery is complete when these scenarios use a real instruction LLM tool
loop, expose their tool progress, cite stored evidence, and protect mutations
with confirmation and audit records.
