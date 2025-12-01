#!/usr/bin/env python3
"""
Stub backend for mask proposals.
Expected to replace dummy boxes in the GUI with real detections/masks (e.g., Grounding DINO + SAM2).

API (FastAPI):
- POST /propose_masks
  {
    "image_path": "D:/.../image.jpg",
    "labels": ["keyboard ribbon connector (class: connector)", "T8 screw heads (count: 4)"]
  }
  Returns: [{"label": str, "score": float, "bbox": [x1,y1,x2,y2], "polygon": [[x,y],...]}...]

This is a placeholder; implement actual detection/masking with Grounding DINO + SAM2.
"""

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import random

app = FastAPI()


class ProposeRequest(BaseModel):
    image_path: str
    labels: List[str]


class MaskProposal(BaseModel):
    label: str
    score: float
    bbox: List[float]
    polygon: Optional[List[List[float]]] = None


@app.post("/propose_masks", response_model=List[MaskProposal])
def propose_masks(req: ProposeRequest):
    # TODO: replace with Grounding DINO + SAM2 inference
    proposals: List[MaskProposal] = []
    for lbl in req.labels:
        # dummy random box
        x1 = random.randint(10, 100)
        y1 = random.randint(10, 100)
        w = random.randint(50, 150)
        h = random.randint(50, 150)
        proposals.append(
            MaskProposal(
                label=lbl,
                score=round(random.uniform(0.5, 0.95), 2),
                bbox=[x1, y1, x1 + w, y1 + h],
                polygon=None,
            )
        )
    return proposals


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
