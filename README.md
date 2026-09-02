# Kid Studio FLUX/SDXL RunPod Worker

RunPod Serverless image worker for Kid Studio.

## Models

- **FLUX.1 Schnell** — default inexpensive text-to-image model; Apache-2.0 weights.
- **Stable Diffusion XL 1.0** — alternate and reference-conditioned image generation.

FLUX 'dev' models are deliberately excluded because their default model license is not appropriate for unrestricted commercial YouTube production.

## RunPod deployment

Deploy this repository through **Serverless → Deploy → Deploy from a GitHub repository**.

- Branch: `main`
- Dockerfile path: `/Dockerfile`
- Endpoint type: Queue
- GPU: 24 GB recommended
- Minimum workers: 0
- Maximum workers: 1
- Network volume mount: `/runpod-volume`

The first `warmup` or `generate` job downloads model weights to the shared network volume. Later workers reuse that cache.

## Operations

Health: `{"input":{"operation":"health"}}`

Warmup: `{"input":{"operation":"warmup","model":"flux-schnell"}}`

The generate operation accepts model, prompt, quality, width, height, seed, optional negative_prompt, and an optional reference_image as base64, data URL, or HTTPS URL.

## Output and metering

The worker returns a PNG in `image_base64`, SHA-256 digest, model, quality, seed, inference time, reference-conditioning status and worker build. RunPod adds queue delay and total execution time for actual GPU-cost accounting.
