import base64
import gc
import hashlib
import io
import os
import secrets
import time
from pathlib import Path
from typing import Any

import requests
import runpod
import torch
from PIL import Image

SERVICE = "kid-studio-flux-sdxl-worker"
WORKER_BUILD = "flux-sdxl-v1"

MODELS = {
    "flux-schnell": {
        "repo": os.getenv(
            "FLUX_MODEL_ID",
            "black-forest-labs/FLUX.1-schnell",
        ),
        "family": "flux",
        "license": "apache-2.0",
    },
    "sdxl": {
        "repo": os.getenv(
            "SDXL_MODEL_ID",
            "stabilityai/stable-diffusion-xl-base-1.0",
        ),
        "family": "sdxl",
        "license": "openrail++",
    },
}

CACHE_ROOT = Path(os.getenv("HF_HOME", "/runpod-volume/huggingface"))
TMP_ROOT = Path(os.getenv("TMPDIR", "/runpod-volume/tmp"))
MAX_INPUT_IMAGE_BYTES = 16 * 1024 * 1024

_loaded_key: str | None = None
_loaded_pipe: Any = None
_loaded_img2img: Any = None


def _ensure_storage() -> None:
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    TMP_ROOT.mkdir(parents=True, exist_ok=True)


def _storage() -> dict[str, Any]:
    _ensure_storage()
    try:
        stat = os.statvfs("/runpod-volume")
        return {
            "root": "/runpod-volume",
            "hf_home": str(CACHE_ROOT),
            "free_bytes": stat.f_bavail * stat.f_frsize,
            "total_bytes": stat.f_blocks * stat.f_frsize,
        }
    except OSError:
        return {
            "root": None,
            "hf_home": str(CACHE_ROOT),
            "free_bytes": None,
            "total_bytes": None,
        }


def _gpu() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"available": False}
    props = torch.cuda.get_device_properties(0)
    return {
        "available": True,
        "name": props.name,
        "vram_bytes": props.total_memory,
        "cuda": torch.version.cuda,
        "torch": torch.__version__,
        "bf16": bool(torch.cuda.is_bf16_supported()),
    }


def _health() -> dict[str, Any]:
    return {
        "ok": True,
        "service": SERVICE,
        "worker_build": WORKER_BUILD,
        "models": {
            name: {
                "repo": data["repo"],
                "license": data["license"],
            }
            for name, data in MODELS.items()
        },
        "loaded_model": _loaded_key,
        "gpu": _gpu(),
        "storage": _storage(),
    }


def _release_model() -> None:
    global _loaded_key, _loaded_pipe, _loaded_img2img
    _loaded_img2img = None
    _loaded_pipe = None
    _loaded_key = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _load_model(model_key: str) -> Any:
    global _loaded_key, _loaded_pipe, _loaded_img2img

    if model_key not in MODELS:
        raise ValueError(
            "Unsupported model. Use flux-schnell or sdxl."
        )
    if _loaded_key == model_key and _loaded_pipe is not None:
        return _loaded_pipe

    _release_model()
    _ensure_storage()

    from diffusers import FluxPipeline, StableDiffusionXLPipeline

    spec = MODELS[model_key]
    if spec["family"] == "flux":
        dtype = (
            torch.bfloat16
            if torch.cuda.is_available()
            and torch.cuda.is_bf16_supported()
            else torch.float16
        )
        pipe = FluxPipeline.from_pretrained(
            spec["repo"],
            torch_dtype=dtype,
            cache_dir=str(CACHE_ROOT),
        )
    else:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            spec["repo"],
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
            cache_dir=str(CACHE_ROOT),
        )

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required.")

    pipe.enable_model_cpu_offload()
    pipe.enable_attention_slicing()

    _loaded_key = model_key
    _loaded_pipe = pipe
    _loaded_img2img = None
    return pipe


def _int_value(value: Any, name: str, minimum: int, maximum: int, default: int) -> int:
    if value is None:
        return default
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer.") from exc
    if number < minimum or number > maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}.")
    return number


def _float_value(value: Any, name: str, minimum: float, maximum: float, default: float) -> float:
    if value is None:
        return default
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric.") from exc
    if number < minimum or number > maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}.")
    return number


def _dimension(value: Any, name: str, default: int) -> int:
    number = _int_value(value, name, 512, 1536, default)
    if number % 8:
        raise ValueError(f"{name} must be divisible by 8.")
    return number


def _decode_image(value: Any) -> Image.Image | None:
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        raise ValueError("reference_image must be a string.")

    if value.startswith("data:image/"):
        marker = ";base64,"
        if marker not in value:
            raise ValueError("reference_image data URL must use base64 encoding.")
        raw = base64.b64decode(value.split(marker, 1)[1], validate=True)
    elif value.startswith("https://"):
        response = requests.get(value, timeout=30)
        response.raise_for_status()
        raw = response.content
    else:
        raw = base64.b64decode(value, validate=True)

    if not raw or len(raw) > MAX_INPUT_IMAGE_BYTES:
        raise ValueError("reference_image must be between 1 byte and 16 MB.")
    image = Image.open(io.BytesIO(raw))
    image.load()
    return image.convert("RGB")


def _quality_settings(model_key: str, quality: str) -> tuple[int, float]:
    if quality not in {"cheap", "balanced", "high"}:
        raise ValueError("quality must be cheap, balanced, or high.")
    if model_key == "flux-schnell":
        return (4, 0.0)
    return {
        "cheap": (18, 5.0),
        "balanced": (28, 6.0),
        "high": (40, 7.0),
    }[quality]


def _generate(data: dict[str, Any]) -> dict[str, Any]:
    global _loaded_img2img

    prompt = str(data.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("prompt is required.")
    if len(prompt) > 5000:
        raise ValueError("prompt cannot exceed 5000 characters.")

    model_key = str(data.get("model") or "flux-schnell").strip().lower()
    quality = str(data.get("quality") or "cheap").strip().lower()
    width = _dimension(data.get("width"), "width", 1024)
    height = _dimension(data.get("height"), "height", 1024)
    seed = _int_value(
        data.get("seed"), "seed", 0, 2**31 - 1, secrets.randbelow(2**31)
    )
    reference = _decode_image(data.get("reference_image"))
    strength = _float_value(data.get("strength"), "strength", 0.05, 0.95, 0.35)
    steps, guidance = _quality_settings(model_key, quality)
    steps = _int_value(
        data.get("num_inference_steps"), "num_inference_steps", 1, 50, steps
    )
    guidance = _float_value(
        data.get("guidance_scale"), "guidance_scale", 0.0, 20.0, guidance
    )

    pipe = _load_model(model_key)
    active_pipe = pipe
    reference_applied = reference is not None

    if reference is not None:
        from diffusers import AutoPipelineForImage2Image

        if _loaded_img2img is None:
            _loaded_img2img = AutoPipelineForImage2Image.from_pipe(pipe)
        active_pipe = _loaded_img2img
        reference = reference.resize((width, height), Image.Resampling.LANCZOS)

    generator = torch.Generator(device="cpu").manual_seed(seed)
    arguments: dict[str, Any] = {
        "prompt": prompt,
        "width": width,
        "height": height,
        "num_inference_steps": steps,
        "guidance_scale": guidance,
        "generator": generator,
        "output_type": "pil",
    }
    if model_key == "sdxl":
        negative = str(data.get("negative_prompt") or "").strip()
        if negative:
            arguments["negative_prompt"] = negative
    if reference is not None:
        arguments["image"] = reference
        arguments["strength"] = strength

    started = time.perf_counter()
    result = active_pipe(**arguments)
    elapsed_ms = round((time.perf_counter() - started) * 1000)

    if not result.images:
        raise RuntimeError("The image model returned no image.")

    image = result.images[0]
    output = io.BytesIO()
    image.save(output, format="PNG", optimize=True)
    raw = output.getvalue()

    return {
        "ok": True,
        "service": SERVICE,
        "worker_build": WORKER_BUILD,
        "model": MODELS[model_key]["repo"],
        "model_key": model_key,
        "model_license": MODELS[model_key]["license"],
        "quality": quality,
        "width": image.width,
        "height": image.height,
        "seed": seed,
        "num_inference_steps": steps,
        "guidance_scale": guidance,
        "reference_conditioning_applied": reference_applied,
        "reference_input_count": 1 if reference_applied else 0,
        "inference_ms": elapsed_ms,
        "image_sha256": hashlib.sha256(raw).hexdigest(),
        "mime_type": "image/png",
        "image_base64": base64.b64encode(raw).decode("ascii"),
    }


def handler(job: dict[str, Any]) -> dict[str, Any]:
    data = job.get("input")
    if not isinstance(data, dict):
        return {
            "ok": False,
            "error": "input must be a JSON object.",
            "worker_build": WORKER_BUILD,
        }

    operation = str(data.get("operation") or "generate").strip().lower()

    try:
        if operation in {"health", "preflight"}:
            return _health()
        if operation == "unload":
            _release_model()
            return {**_health(), "unloaded": True}
        if operation == "warmup":
            model_key = str(data.get("model") or "flux-schnell").strip().lower()
            started = time.perf_counter()
            _load_model(model_key)
            return {
                **_health(),
                "warmed_model": model_key,
                "load_ms": round((time.perf_counter() - started) * 1000),
            }
        if operation != "generate":
            raise ValueError(
                "operation must be health, preflight, warmup, unload, or generate."
            )
        return _generate(data)
    except Exception as exc:
        return {
            "ok": False,
            "service": SERVICE,
            "worker_build": WORKER_BUILD,
            "error": str(exc),
            "error_type": type(exc).__name__,
        }


if __name__ == "__main__":
    _ensure_storage()
    runpod.serverless.start({"handler": handler})
