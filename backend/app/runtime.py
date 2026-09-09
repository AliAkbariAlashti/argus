"""Hardware-independent vision runtime; camera viewing never requires model packages."""
import json
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import httpx
from PIL import Image

from .imaging import image_to_data_uri
from .vlm import VisionLanguageModel


class VisionRuntime:
    def __init__(self, path=None):
        self.path = Path(path or os.environ.get("ARGUS_RUNTIME_FILE", Path(__file__).resolve().parents[2] / "data" / "runtime.json"))
        self._settings = {"provider": "none", "base_url": "", "model": "", "api_key": "", "allow_remote": False, "monitor_enabled": False}
        self._verified = False
        self._error = None
        self._lock = threading.RLock()
        self._embedded = VisionLanguageModel()
        self.interactive = threading.Event()
        if self.path.exists():
            self._settings.update(json.loads(self.path.read_text()))

    @property
    def ready(self):
        if self._settings["provider"] == "embedded":
            return self._embedded.ready
        return self._settings["provider"] == "compatible" and self._verified

    @property
    def error(self):
        return self._error or (self._embedded.error if self._settings["provider"] == "embedded" else None)

    @property
    def configured(self):
        return self._settings["provider"] != "none"

    @property
    def monitor_enabled(self):
        return self._settings.get("monitor_enabled", False)

    def public_settings(self):
        # Settings are replaced atomically. Status must not wait behind inference.
        settings = self._settings
        return {**{k: v for k, v in settings.items() if k != "api_key"},
                "has_api_key": bool(settings.get("api_key")),
                "ready": self.ready, "error": self.error, "configured": self.configured}

    def configure(self, settings):
        settings = dict(settings)
        if settings["provider"] == "compatible":
            url = urlsplit(settings["base_url"])
            if url.scheme not in ("http", "https") or not url.hostname or url.username or url.password or url.query or url.fragment:
                raise ValueError("Use an HTTP(S) API base URL without credentials, query, or fragment.")
            if not settings["model"].strip():
                raise ValueError("Enter the exact name of a vision-capable model.")
            # Explicit consent applies to all servers outside this machine,
            # including a LAN server. Never silently switch to a hosted model.
            local = url.hostname in ("localhost", "127.0.0.1", "::1")
            if not local and not settings.get("allow_remote"):
                raise ValueError("Confirm that frames may be sent to the configured server.")
        with self._lock:
            if settings.get("api_key") is None:
                same_server = settings.get("base_url") == self._settings.get("base_url")
                settings["api_key"] = self._settings.get("api_key", "") if same_server else ""
            settings["base_url"] = settings["base_url"].rstrip("/")
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w") as out:
                json.dump(settings, out)
            os.chmod(tmp, 0o600)
            tmp.replace(self.path)
            self._settings = settings
            self._verified = False
            self._error = None
        return self.public_settings()

    def load(self):
        # No downloads, GPU allocation, or hosted calls without setup.
        if self._settings["provider"] == "embedded":
            self._embedded.load()

    def test(self):
        with self._lock:
            self._verified = False
            if self._settings["provider"] == "none":
                raise ValueError("Choose an AI connection first.")
            started = time.monotonic()
            if self._settings["provider"] == "embedded":
                self._embedded.load()
                response = self._embedded.ask(Image.new("RGB", (64, 64), "red"), "What color is this image?", max_new_tokens=24)
            else:
                response = self._remote_ask(Image.new("RGB", (64, 64), "red"), "What color is this image?", max_new_tokens=24)
            if "red" not in response.lower():
                raise ValueError("The server responded but did not identify the red test image. Check that the selected model supports vision.")
            self._verified = True
            self._error = None
            return {"ready": True, "seconds": round(time.monotonic() - started, 1), "message": "Vision test passed. Live chat is ready."}

    def ask(self, image, question, max_new_tokens=512, history=None, fps=None, on_token=None):
        with self._lock:
            if not self.ready:
                raise RuntimeError("Set up and test an AI connection in AI setup before asking visual questions.")
            if self._settings["provider"] == "embedded":
                return self._embedded.ask(image, question, max_new_tokens, history, fps, on_token)
            try:
                return self._remote_ask(image, question, max_new_tokens, history, fps, on_token)
            except RuntimeError as exc:
                self._verified = False
                self._error = str(exc)
                raise

    @staticmethod
    def _is_ollama_url(base_url):
        parsed = urlsplit(base_url)
        return parsed.port == 11434 or parsed.hostname == "host.docker.internal"

    def _is_moondream(self):
        return self._settings.get("model", "").strip().lower().split(":", 1)[0] == "moondream"

    @staticmethod
    def _request_url(base_url):
        parsed = urlsplit(base_url)
        # The browser can use localhost for a local Ollama install, but a
        # container's localhost is the container itself. compose.mvp.yml adds
        # this Docker host alias on Linux; keep the saved URL unchanged so the
        # same setup also works when Argus runs directly on the host.
        if os.path.exists("/.dockerenv") and parsed.hostname in ("localhost", "127.0.0.1", "::1"):
            host = "host.docker.internal"
            netloc = host
            port = parsed.port
            relay_port = os.environ.get("ARGUS_OLLAMA_RELAY_PORT")
            if port == 11434 and relay_port:
                port = int(relay_port)
            if port:
                netloc += f":{port}"
            parsed = parsed._replace(netloc=netloc)
        return urlunsplit(parsed).rstrip("/")

    def _ollama_url(self, base_url):
        return self._request_url(base_url).removesuffix("/v1") + "/api/generate"

    def _ollama_prompt(self, question, history):
        if self._is_moondream():
            # Ollama's Moondream template supplies its own Question/Answer
            # wrapper. Sending Argus's long grounded prompt verbatim nests a
            # second Question marker and frequently makes the model emit EOS.
            return question.rsplit("Question:", 1)[-1].strip()
        turns = []
        for turn in history or []:
            role = turn.get("role")
            text = (turn.get("text") or "").strip()
            if role in ("user", "assistant") and text:
                turns.append(f"{role.title()}: {text}")
        return "\n".join(turns + [question])

    def _ollama_ask(self, image, question, max_new_tokens, history=None, fps=None, on_token=None):
        frames = image if isinstance(image, list) else [image]
        if self._is_moondream() and len(frames) > 1:
            # Moondream's 2K context is consumed almost entirely by one image
            # at this thumbnail size. The latest frame is the useful current
            # state; sending the whole motion buffer causes Ollama HTTP 400.
            frames = [frames[-1]]
            fps = None
        if fps and len(frames) > 1:
            question = f"These images are chronological samples, approximately {1 / fps:g} seconds apart. " + question
        images = []
        for frame in frames:
            data_uri = image_to_data_uri(frame)
            if not data_uri or "," not in data_uri:
                raise RuntimeError("Could not prepare the image for Ollama.")
            images.append(data_uri.split(",", 1)[1])
        payload = {
            "model": self._settings["model"],
            "prompt": self._ollama_prompt(question, history[-2:] if self._is_moondream() and history else history),
            "images": images,
            "options": {"num_gpu": 0, "num_predict": max_new_tokens},
            "stream": on_token is not None,
        }
        headers = {}
        if self._settings.get("api_key"):
            headers["Authorization"] = "Bearer " + self._settings["api_key"]
        try:
            with httpx.Client(timeout=httpx.Timeout(180, connect=10), follow_redirects=False) as client:
                if on_token is None:
                    res = client.post(self._ollama_url(self._settings["base_url"]), headers=headers, json=payload)
                    res.raise_for_status()
                    answer = res.json()["response"]
                else:
                    chunks = []
                    completed = False
                    with client.stream("POST", self._ollama_url(self._settings["base_url"]), headers=headers, json=payload) as res:
                        res.raise_for_status()
                        for line in res.iter_lines():
                            if not line:
                                continue
                            packet = json.loads(line)
                            if packet.get("error"):
                                raise RuntimeError("The AI server reported a generation error.")
                            chunk = packet.get("response") or ""
                            if chunk:
                                chunks.append(chunk)
                                on_token(chunk)
                            if packet.get("done"):
                                completed = True
                                break
                    if not completed:
                        raise RuntimeError("The AI server disconnected before finishing. Please retry.")
                    answer = "".join(chunks)
                if not isinstance(answer, str) or not answer.strip():
                    raise RuntimeError("The AI server returned an empty answer. Check that the Ollama model can run vision inference on CPU.")
                return answer.strip()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 400 and "exceeds the available context size" in exc.response.text:
                raise RuntimeError("The Ollama model context is too small for the selected frames. Moondream uses the latest frame automatically; retry the question.") from None
            raise RuntimeError(f"AI server returned HTTP {exc.response.status_code}. Check the Ollama model and API address in AI setup.") from None
        except httpx.RequestError:
            raise RuntimeError("Cannot reach the AI server, or it timed out. Check AI setup and retry.") from None
        except (KeyError, TypeError, json.JSONDecodeError):
            raise RuntimeError("The Ollama server returned an unsupported response. Check the selected model.") from None

    def _remote_ask(self, image, question, max_new_tokens=512, history=None, fps=None, on_token=None):
        if self._is_ollama_url(self._settings["base_url"]):
            return self._ollama_ask(image, question, max_new_tokens, history, fps, on_token)
        settings = self._settings
        frames = image if isinstance(image, list) else [image]
        messages = [{"role": t["role"], "content": t["text"]} for t in (history or []) if t.get("role") in ("user", "assistant")]
        if fps and len(frames) > 1:
            question = f"These images are chronological samples, approximately {1 / fps:g} seconds apart. " + question
        content = [{"type": "image_url", "image_url": {"url": image_to_data_uri(frame)}} for frame in frames]
        content.append({"type": "text", "text": question})
        messages.append({"role": "user", "content": content})
        headers = {}
        if settings.get("api_key"):
            headers["Authorization"] = "Bearer " + settings["api_key"]
        payload = {"model": settings["model"], "messages": messages, "max_tokens": max_new_tokens, "stream": on_token is not None}
        url = self._request_url(settings["base_url"]) + "/chat/completions"
        try:
            with httpx.Client(timeout=httpx.Timeout(120, connect=10), follow_redirects=False) as client:
                if on_token is None:
                    res = client.post(url, headers=headers, json=payload)
                    res.raise_for_status()
                    answer = res.json()["choices"][0]["message"]["content"]
                else:
                    chunks = []
                    completed = False
                    with client.stream("POST", url, headers=headers, json=payload) as res:
                        res.raise_for_status()
                        for line in res.iter_lines():
                            if not line.startswith("data:"):
                                continue
                            data = line[5:].strip()
                            if data == "[DONE]":
                                completed = True
                                break
                            packet = json.loads(data)
                            if packet.get("error"):
                                raise RuntimeError("The AI server reported a generation error.")
                            choices = packet.get("choices", [])
                            if not choices:
                                continue
                            delta = choices[0].get("delta", {}).get("content") or ""
                            if delta:
                                chunks.append(delta)
                                on_token(delta)
                            if choices[0].get("finish_reason"):
                                completed = True
                    if not completed:
                        raise RuntimeError("The AI server disconnected before finishing. Please retry.")
                    answer = "".join(chunks)
                if not isinstance(answer, str) or not answer.strip():
                    raise RuntimeError("The AI server returned an empty answer. Check its vision model.")
                return answer.strip()
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(f"AI server returned HTTP {exc.response.status_code}. Check the model name, credentials, and API address in AI setup.") from None
        except httpx.RequestError:
            raise RuntimeError("Cannot reach the AI server, or it timed out. Check AI setup and retry.") from None
        except (KeyError, IndexError, TypeError, json.JSONDecodeError):
            raise RuntimeError("The server returned an unsupported response. Use a compatible vision chat API.") from None


def hardware_profile():
    gpu = None
    if shutil.which("nvidia-smi"):
        try:
            result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True, timeout=3)
            if result.returncode == 0:
                gpu = result.stdout.strip().splitlines()[0]
        except (OSError, subprocess.TimeoutExpired, IndexError):
            pass
    return {"cpu_count": os.cpu_count(), "nvidia_gpu": gpu,
            "recommendation": "Connect a local vision server or a hosted vision API. Camera viewing works without AI." if not gpu else "GPU detected. Use a local vision server or the embedded Qwen runtime after installing GPU dependencies."}


vlm = VisionRuntime()
