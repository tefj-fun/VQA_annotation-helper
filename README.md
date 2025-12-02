# VQA Annotation Helper (LLaVA GUI)

Desktop helper to guide image annotation with a vision-language model. Loads a local dataset (e.g., iFixit JSONL + images), suggests what to highlight per step, and can propose reviewable masks via a backend (Grounding DINO + SAM) or a stub.

## Features
- LLaVA 1.5 VQA (7B) via `transformers`, CUDA-capable.
- Load `metadata.jsonl`, browse guide/step captions; auto-link images to steps.
- Step-aware prompts; concise bullets with class labels/positions.
- Two UIs:
  - Tkinter GUI: `scripts/llava_gui.py`
  - PyQt GUI: `scripts/llava_gui_qt.py` with mask list/filter, show/hide rejected overlays, truncated labels.
- Mask proposals call a backend (`MASK_BACKEND_URL`) that can be Grounding DINO + SAM; dummy boxes on backend failure.
- CLI runner: `scripts/run_llava_vqa.py` for one-off VQA.

## Requirements
- Python 3.10+ (fresh conda env recommended, e.g., `blip`).
- NVIDIA GPU and CUDA PyTorch (e.g., cu121 build).
- Packages: torch (CUDA build), torchvision, transformers, pillow, datasets, peft, accelerate, bitsandbytes, pyqt5.

## One-time setup (conda)
```bash
conda create -n blip python=3.10 -y
conda activate blip
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets peft accelerate pillow bitsandbytes pyqt5
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

## Run the Tk GUI
```bash
conda activate blip
cd D:\Joseph\FUN\VQA_annotation-helper
python scripts\llava_gui.py
```
Workflow:
1) Load `metadata.jsonl` (default iFixit path ok).
2) Pick guide/step (or open image); auto-link if found.
3) Run VQA for highlight bullets.
4) Optional masks: proposes boxes; Accept/Reject; set `MASK_BACKEND_URL` for real backend.

## Run the PyQt GUI (recommended)
```bash
conda activate blip
cd D:\Joseph\FUN\VQA_annotation-helper
python scripts\llava_gui_qt.py
```
Workflow:
- Load metadata, pick guide/step (or open image). Caption and image auto-link.
- Run VQA; answers are line-wrapped.
- Propose masks:
  - Backend at `MASK_BACKEND_URL` (default `http://localhost:8000/propose_masks`).
  - Score threshold filter.
  - Click list row to highlight overlay; Accept/Reject set color/status.
  - Toggle “Show rejected masks” to hide/show rejected rows and overlays.
  - Labels in list are truncated to keep UI compact.

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

## Mask Backend
- `scripts/mask_backend_stub.py`: random boxes (dev).
- `scripts/mask_backend.py`: Grounding DINO (`IDEA-Research/grounding-dino-base`) + optional SAM (set `SAM_CHECKPOINT`, `SAM_MODEL_TYPE`). Returns polygons when SAM is set, else boxes.
Run:
```bash
# stub
python scripts\mask_backend_stub.py
# real backend
python scripts\mask_backend.py
# POST /propose_masks with {image_path, labels, box_threshold, text_threshold, top_k, return_masks}
```

## Notes
- First run downloads HF weights to cache.
- Ensure `torch.cuda.is_available()` is True; CPU is very slow.
- Keep datasets out of the repo; point the app to your local JSONL/images.
