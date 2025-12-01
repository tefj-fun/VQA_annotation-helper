# VQA Annotation Helper (LLaVA GUI)

Simple desktop helper to guide image annotation with a vision-language model. It loads a local dataset (e.g., iFixit JSONL + images), suggests what to highlight in each step, and can propose reviewable masks (placeholder logic included).

## Features
- LLaVA 1.5 VQA (7B) via `transformers` with CUDA support.
- Load metadata (`metadata.jsonl`) and browse guide/step captions; auto-link images to their step.
- Prompt includes step context; output shows concise bullets (with class labels).
- Scrollable output; simple mask proposal UI (dummy boxes now—swap with real detector/segmenter later).
- CLI runner (`scripts/run_llava_vqa.py`) for quick one-off VQA on an image.

## Requirements
- Python 3.10+ (recommend a fresh conda env, e.g., `blip`).
- NVIDIA GPU and CUDA-capable PyTorch (e.g., cu121 build).
- Packages: torch (CUDA build), torchvision, transformers, pillow, datasets, peft, accelerate, bitsandbytes.

## One-time setup (conda)
```bash
conda create -n blip python=3.10 -y
conda activate blip
# install CUDA torch; use cu121 or cu118 to match your driver
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets peft accelerate pillow bitsandbytes
```
Verify GPU:
```bash
python - <<'PY'
import torch
print("cuda available:", torch.cuda.is_available())
print("torch version:", torch.__version__)
print("cuda version:", torch.version.cuda)
PY
```

## Run the GUI
```bash
conda activate blip
cd D:\Joseph\FUN\VQA_annotation-helper
python scripts\llava_gui.py
```
Workflow:
1) Click "Load Metadata" and select your `metadata.jsonl` (default expected at `D:\Joseph\FUN\ifixit\datasets\ifixit_blip_en\metadata.jsonl` or wherever your JSONL lives).
2) Pick a guide and step (or open an image; it will auto-link if found in JSONL).
3) Click "Run VQA" to get highlight suggestions (bullets with class labels/positions). Output shows only the assistant text.
4) (Optional) Click "Propose masks" to draw placeholder boxes from the suggestions; accept/reject to simulate review. Replace the placeholder logic with a real backend when ready.

## CLI VQA
```bash
conda activate blip
cd D:\Joseph\FUN\VQA_annotation-helper
python scripts\run_llava_vqa.py ^
  --model-id llava-hf/llava-1.5-7b-hf ^
  --image D:\path\to\image.jpg ^
  --question "Suggest what to highlight (bullets with counts/positions)." ^
  --use-fast --max-new-tokens 64
```

## Mask Backend Stub
Options:
- `scripts/mask_backend_stub.py`: placeholder FastAPI service returning random boxes.
- `scripts/mask_backend.py`: FastAPI service using transformers zero-shot detection (Grounding DINO via `IDEA-Research/grounding-dino-base`) and optional SAM masks (set `SAM_CHECKPOINT` and `SAM_MODEL_TYPE`).

Run stub (dev):
```bash
python scripts\mask_backend_stub.py
# will serve at http://localhost:8000/propose_masks
```

Run mask backend (needs transformers, torch, optional segment-anything):
```bash
# set SAM_CHECKPOINT if you want masks; otherwise only boxes are returned
python scripts\mask_backend.py
# POST /propose_masks with {image_path, labels, box_threshold, text_threshold, top_k, return_masks}
```

## Notes
- First model run will download weights to HF cache (`C:\Users\Supplier\.cache\huggingface\hub\...`).
- Ensure `torch.cuda.is_available()` is True before running; otherwise LLaVA will be very slow on CPU.
- Keep datasets out of the repo; point the app to your local JSONL/images.
