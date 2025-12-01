# VQA Annotation Helper (LLaVA GUI)

Simple desktop helper to guide image annotation with a vision-language model. It loads a local dataset (e.g., iFixit JSONL + images), suggests what to highlight in each step, and can propose reviewable masks (placeholder logic included).

## Features
- LLaVA 1.5 VQA (7B) via `transformers` with CUDA support.
- Load metadata (`metadata.jsonl`) and browse guide/step captions; auto-link images to their step.
- Prompt includes step context; output shows concise bullets (with class labels).
- Scrollable output; simple mask proposal UI (dummy boxes now—swap with real detector/segmenter later).
- CLI runner (`scripts/run_llava_vqa.py`) for quick one-off VQA on an image.

## Requirements
- Python 3.10+ in the `blip` conda env (torch with CUDA, transformers, pillow, datasets, peft, accelerate, bitsandbytes, torchvision).
- NVIDIA GPU recommended; torch must be a CUDA build (e.g., cu121).

## Run the GUI
```bash
conda activate blip
cd D:\Joseph\FUN\BLIP
python scripts\llava_gui.py
```
Workflow:
1) Click “Load Metadata” and select your `metadata.jsonl` (default expected at `D:\Joseph\FUN\ifixit\datasets\ifixit_blip_en\metadata.jsonl`).
2) Pick a guide and step (or open an image; it will auto-link if found in JSONL).
3) Click “Run VQA” to get highlight suggestions. Output shows only the assistant text.
4) (Optional) Click “Propose masks” to draw placeholder boxes from the suggestions; accept/reject to simulate review. Replace the placeholder logic with a real backend when ready.

## CLI VQA
```bash
conda activate blip
cd D:\Joseph\FUN\BLIP
python scripts\run_llava_vqa.py ^
  --model-id llava-hf/llava-1.5-7b-hf ^
  --image D:\path\to\image.jpg ^
  --question "Suggest what to highlight (bullets with counts/positions)." ^
  --use-fast --max-new-tokens 64
```

## Mask Backend Stub
A placeholder FastAPI service is in `scripts/mask_backend_stub.py`. It returns random boxes; replace with Grounding DINO + SAM/SAM2 to generate real masks per VQA label.
Run stub (dev):
```bash
python scripts\mask_backend_stub.py
# will serve at http://localhost:8000/propose_masks
```

## Notes
- First model run will download weights to HF cache (`C:\Users\Supplier\.cache\huggingface\hub\...`).
- Ensure `torch.cuda.is_available()` is True before running; otherwise LLaVA will be very slow on CPU.
