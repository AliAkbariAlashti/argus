import threading
import logging

from PIL import Image

from .config import MODEL_ID, PREWARM_MODEL, VLM_MIN_PIXELS, VLM_MAX_PIXELS

ImageOrList = Image.Image | list[Image.Image]

log = logging.getLogger("qwenvl.vlm")


class VisionLanguageModel:
    """Thin wrapper around Qwen2.5-VL-7B-Instruct. Loaded once, reused for
    every request. A single lock serializes inference since there's one GPU."""

    def __init__(self):
        self._lock = threading.Lock()
        self._model = None
        self._processor = None
        self._loaded = False
        self._load_error = None

    def load(self):
        if self._loaded or self._load_error:
            return
        try:
            import torch
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

            log.info("Loading %s ...", MODEL_ID)
            self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                # Explicit, not left to the library default: transformers
                # resolves flash_attention_2 > sdpa > eager automatically,
                # and flash-attn isn't installable as a prebuilt wheel for
                # this box's cu130/py3.12 combo. Pins the already-active
                # choice so a future transformers upgrade can't silently
                # drop it to eager.
                attn_implementation="sdpa",
            )
            self._processor = AutoProcessor.from_pretrained(
                MODEL_ID,
                min_pixels=VLM_MIN_PIXELS,
                max_pixels=VLM_MAX_PIXELS,
            )
            self._loaded = True
            log.info("Model loaded.")
        except Exception as exc:  # noqa: BLE001
            self._load_error = str(exc)
            log.exception("Failed to load model")
            return

        if PREWARM_MODEL:
            self._prewarm()

    def _prewarm(self):
        """One tiny inference so CUDA kernels/allocator are warm before the
        first real question — otherwise that question eats the setup cost."""
        try:
            log.info("Pre-warming model ...")
            blank = Image.new("RGB", (64, 64), (16, 16, 16))
            self.ask(blank, "Reply with the single word: ready.", max_new_tokens=8)
            log.info("Pre-warm complete.")
        except Exception:  # noqa: BLE001
            # Never let a pre-warm failure take down a model that loaded fine.
            log.exception("Pre-warm failed (continuing anyway)")

    @property
    def ready(self) -> bool:
        return self._loaded

    @property
    def error(self):
        return self._load_error

    def ask(
        self,
        image: ImageOrList,
        question: str,
        max_new_tokens: int = 512,
        history: list[dict] | None = None,
        fps: float | None = None,
    ) -> str:
        """`history` is prior turns as [{"role": "user"|"assistant", "text": ...}].
        Only the current turn carries images — replaying old frames would grow
        VRAM use with every message for little benefit.

        `fps` marks `image` as an ordered temporal sequence from one camera
        (recent frames, oldest first) rather than an unordered set of stills.
        When set, frames are sent as Qwen2.5-VL's native "video" content type
        instead of separate "image" entries: that's what routes them through
        the model's time-aware M-RoPE position encoding, so it reasons about
        motion from the architecture built for it instead of inferring frame
        order from a sentence in the prompt. Multi-camera snapshots (fleet
        questions) are a different moment per camera, not a sequence — those
        stay as plain images."""
        if not self._loaded:
            raise RuntimeError(self._load_error or "Model is not loaded yet")

        images = image if isinstance(image, list) else [image]
        as_video = fps is not None and len(images) > 1

        messages = []
        for turn in history or []:
            role = turn.get("role")
            text = (turn.get("text") or "").strip()
            if role in ("user", "assistant") and text:
                messages.append(
                    {"role": role, "content": [{"type": "text", "text": text}]}
                )

        if as_video:
            # qwen_vl_utils resizes video frames against its own internal
            # defaults, separate from the processor's min_pixels/max_pixels
            # (those only govern the plain-"image" path) — without pinning
            # them here explicitly, video frames can end up larger than the
            # budget the rest of the app assumes.
            visual_content = [
                {
                    "type": "video",
                    "video": images,
                    "fps": fps,
                    "min_pixels": VLM_MIN_PIXELS,
                    "max_pixels": VLM_MAX_PIXELS,
                }
            ]
        else:
            visual_content = [{"type": "image", "image": img} for img in images]

        messages.append(
            {
                "role": "user",
                "content": visual_content + [{"type": "text", "text": question}],
            }
        )

        with self._lock:
            text = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            if as_video:
                from qwen_vl_utils import process_vision_info

                image_inputs, video_inputs, video_kwargs = process_vision_info(
                    messages, return_video_kwargs=True
                )
                # qwen_vl_utils returns some entries (e.g. "fps") as one
                # value per video, since it supports multiple videos in one
                # call — this transformers version's processor validates
                # them as bare scalars instead. We only ever send one video
                # per call here, so unwrapping is always correct.
                video_kwargs = {
                    k: (v[0] if isinstance(v, list) and len(v) == 1 else v)
                    for k, v in video_kwargs.items()
                }
                inputs = self._processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                    **video_kwargs,
                )
            else:
                inputs = self._processor(
                    text=[text],
                    images=images,
                    padding=True,
                    return_tensors="pt",
                )
            inputs = inputs.to(self._model.device)

            try:
                generated = self._model.generate(**inputs, max_new_tokens=max_new_tokens)
                trimmed = [
                    out[len(inp):] for inp, out in zip(inputs.input_ids, generated)
                ]
                result = self._processor.batch_decode(
                    trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
            finally:
                # This box idles with well under 1GB of VRAM headroom, so
                # PyTorch's caching allocator holding reserved-but-unused
                # blocks between requests (visible as "reserved by PyTorch
                # but unallocated" in an OOM message) is enough on its own to
                # push the next request over the edge. Runs on failure too,
                # so one OOM doesn't leave the process worse off for the
                # next request.
                import torch

                torch.cuda.empty_cache()
        return result[0].strip()


vlm = VisionLanguageModel()
