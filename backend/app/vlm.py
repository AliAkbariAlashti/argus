import threading
import logging

from PIL import Image

ImageOrList = Image.Image | list[Image.Image]

from .config import MODEL_ID

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
            )
            self._processor = AutoProcessor.from_pretrained(MODEL_ID)
            self._loaded = True
            log.info("Model loaded.")
        except Exception as exc:  # noqa: BLE001
            self._load_error = str(exc)
            log.exception("Failed to load model")

    @property
    def ready(self) -> bool:
        return self._loaded

    @property
    def error(self):
        return self._load_error

    def ask(self, image: ImageOrList, question: str, max_new_tokens: int = 512) -> str:
        if not self._loaded:
            raise RuntimeError(self._load_error or "Model is not loaded yet")

        images = image if isinstance(image, list) else [image]

        messages = [
            {
                "role": "user",
                "content": (
                    [{"type": "image", "image": img} for img in images]
                    + [{"type": "text", "text": question}]
                ),
            }
        ]

        with self._lock:
            text = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self._processor(
                text=[text],
                images=images,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self._model.device)

            generated = self._model.generate(**inputs, max_new_tokens=max_new_tokens)
            trimmed = [
                out[len(inp):] for inp, out in zip(inputs.input_ids, generated)
            ]
            result = self._processor.batch_decode(
                trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
        return result[0].strip()


vlm = VisionLanguageModel()
