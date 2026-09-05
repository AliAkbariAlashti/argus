Yes — understood. We completely separate the **AI hardware/infrastructure environment** from the product/app.

## What is ready on the machine

### Hardware

* **GPU:** NVIDIA RTX 4000 Ada Generation
* **VRAM:** ~20 GB
* **GPU Driver:** 580.173.02
* **CUDA:** 13.0
* **OS:** Ubuntu Linux

### Main AI Model

**Qwen2.5-VL-7B-Instruct**

```text
Qwen/Qwen2.5-VL-7B-Instruct
```

* 7B parameter Vision-Language Model
* Image understanding
* Video understanding capability
* Text + visual input
* Text generation output
* Successfully loaded and tested on the RTX 4000 Ada
* Running with **BF16**
* Loaded with `device_map="auto"`

### AI Runtime

* **Python:** 3.12
* **PyTorch:** 2.12.1+cu130
* **Transformers:** 5.8.0
* **Accelerate:** 1.14.0

### Vision / Media Libraries

* `qwen-vl-utils` 0.0.14
* `decord` 0.6.0
* `av` 18.1.0
* `Pillow` 12.2.0
* `opencv-python-headless` 5.0.0.93

### Other LLMs available

Through Ollama:

* **DeepSeek-R1 14B**
* **Qwen3 14B**

These are separate from the Qwen2.5-VL vision model.

### Environment status

The important part is that the GPU stack is already working:

```text
NVIDIA GPU
    ↓
CUDA 13.0
    ↓
PyTorch 2.12.1+cu130
    ↓
Transformers
    ↓
Qwen2.5-VL-7B-Instruct
```

And we have already confirmed that **Qwen2.5-VL can load and perform inference successfully on this hardware**.

So, from the infrastructure/AI side, the machine is essentially **ready to be used as a local VLM inference machine**.
