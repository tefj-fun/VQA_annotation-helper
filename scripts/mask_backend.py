#!/usr/bin/env python3
"""
Mask proposal backend using zero-shot detection (Grounding DINO via transformers)
and optional SAM/SAM2 masks.

POST /propose_masks
Payload:
{
  "image_path": "D:/path/to/image.jpg",
  "labels": ["keyboard ribbon connector (class: connector)", "T8 screw heads (count: 4)"],
  "box_threshold": 0.25,
  "text_threshold": 0.25,
  "top_k": 3,
  "return_masks": true
}

Response:
[
  {
    "label": "keyboard ribbon connector (class: connector)",
    "score": 0.82,
    "bbox": [x1, y1, x2, y2],
    "polygon": [[x, y], ...]  # empty if masks disabled/unavailable
  },
  ...
]

Dependencies:
- transformers >= 4.37, torch, torchvision, pillow
- Optional masks: segment-anything (SAM) and a SAM checkpoint (set SAM_CHECKPOINT)
"""

from __future__ import annotations

import os
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
from transformers import pipeline

try:
    from segment_anything import SamPredictor, sam_model_registry

    SAM_AVAILABLE = True
except Exception:
    SAM_AVAILABLE = False

DEFAULT_MODEL_ID = os.environ.get("GROUNDING_MODEL_ID", "IDEA-Research/grounding-dino-base")
SAM_CHECKPOINT = os.environ.get("SAM_CHECKPOINT")  # e.g., path to sam_vit_h_4b8939.pth
SAM_MODEL_TYPE = os.environ.get("SAM_MODEL_TYPE", "vit_h")

app = FastAPI()


class ProposeRequest(BaseModel):
    image_path: str
    labels: List[str]
    box_threshold: float = 0.25
    text_threshold: float = 0.25
    top_k: int = 3
    return_masks: bool = False


class MaskProposal(BaseModel):
    label: str
    score: float
    bbox: List[float]
    polygon: Optional[List[List[float]]] = None


def load_detector():
    det = pipeline(
        "zero-shot-object-detection",
        model=DEFAULT_MODEL_ID,
        device_map="auto",
        torch_dtype="auto",
    )
    return det


def load_sam():
    if not SAM_AVAILABLE or not SAM_CHECKPOINT:
        return None
    sam = sam_model_registry.get(SAM_MODEL_TYPE)(checkpoint=SAM_CHECKPOINT)
    predictor = SamPredictor(sam)
    return predictor


detector = load_detector()
sam_predictor = load_sam()


def run_detection(img: Image.Image, labels: List[str], box_threshold: float, text_threshold: float, top_k: int):
    results = []
    for lbl in labels:
        dets = detector(
            img,
            candidate_labels=[lbl],
            threshold=box_threshold,
        )
        # pipeline returns list of dicts with 'score' and 'box' keys
        sorted_dets = sorted(dets, key=lambda d: d.get("score", 0), reverse=True)[:top_k]
        for d in sorted_dets:
            box = d.get("box", {})
            x1, y1, x2, y2 = box.get("xmin"), box.get("ymin"), box.get("xmax"), box.get("ymax")
            if None in (x1, y1, x2, y2):
                continue
            results.append(
                {
                    "label": lbl,
                    "score": float(d.get("score", 0)),
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                }
            )
    return results


def box_to_polygon(predictor: SamPredictor, img: Image.Image, bbox: List[float]) -> Optional[List[List[float]]]:
    try:
        predictor.set_image(np.array(img))
        box_np = np.array(bbox, dtype=np.float32)
        masks, scores, _ = predictor.predict(box=box_np, point_coords=None, point_labels=None, multimask_output=False)
        mask = masks[0]
        # Extract polygon from mask
        coords = np.argwhere(mask)
        if coords.size == 0:
            return None
        # Use bounding contour
        ys, xs = coords[:, 0], coords[:, 1]
        poly = [
            [float(xs.min()), float(ys.min())],
            [float(xs.max()), float(ys.min())],
            [float(xs.max()), float(ys.max())],
            [float(xs.min()), float(ys.max())],
        ]
        return poly
    except Exception:
        return None


@app.post("/propose_masks", response_model=List[MaskProposal])
def propose_masks(req: ProposeRequest):
    img_path = req.image_path
    if not os.path.exists(img_path):
        raise HTTPException(status_code=400, detail=f"Image not found: {img_path}")
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to open image: {exc}")

    dets = run_detection(
        img=img,
        labels=req.labels,
        box_threshold=req.box_threshold,
        text_threshold=req.text_threshold,
        top_k=req.top_k,
    )
    proposals: List[MaskProposal] = []
    for d in dets:
        poly = None
        if req.return_masks and sam_predictor:
            poly = box_to_polygon(sam_predictor, img, d["bbox"])
        proposals.append(
            MaskProposal(
                label=d["label"],
                score=d["score"],
                bbox=d["bbox"],
                polygon=poly,
            )
        )
    return proposals


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
