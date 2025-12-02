#!/usr/bin/env python3
"""
Simple Tkinter GUI to run LLaVA VQA on uploaded images.
- Choose an image (or pick from iFixit metadata)
- Enter a question
- Adjust max new tokens
Includes a metadata browser for guides/steps from metadata.jsonl.
"""

import json
import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from collections import defaultdict
import pathlib
from typing import Any, Dict, List

import requests
from PIL import Image, ImageTk
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor


MODEL_ID = "llava-hf/llava-1.5-7b-hf"
DEFAULT_METADATA = r"D:\Joseph\FUN\ifixit\datasets\ifixit_blip_en\metadata.jsonl"
MASK_BACKEND_URL = os.environ.get("MASK_BACKEND_URL", "http://localhost:8000/propose_masks")


class LlavaGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("LLaVA VQA")
        self.geometry("1200x800")

        self.model = None
        self.processor = None
        self.device = "cpu"
        self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.image_path = None
        self.photo = None
        self.metadata = []
        self.guide_index = defaultdict(lambda: defaultdict(list))
        self.data_dir = None
        self.current_entry = None

        # Controls
        top = tk.Frame(self)
        top.pack(fill="x", padx=10, pady=5)

        tk.Label(top, text="Question:").pack(side="left")
        self.question_var = tk.StringVar(
            value="Suggest what to circle (imperative bullets, include counts and positions)."
        )
        tk.Entry(top, textvariable=self.question_var, width=50).pack(side="left", padx=5)

        tk.Label(top, text="Max tokens:").pack(side="left")
        self.max_tokens_var = tk.IntVar(value=64)
        tk.Spinbox(top, from_=8, to=512, textvariable=self.max_tokens_var, width=5).pack(
            side="left", padx=5
        )

        tk.Button(top, text="Load Metadata", command=self.load_metadata).pack(side="left", padx=5)
        tk.Button(top, text="Open Image", command=self.open_image).pack(side="left", padx=5)
        tk.Button(top, text="Run VQA", command=self.run_vqa).pack(side="left", padx=5)

        # Guide/step selectors
        nav = tk.Frame(self)
        nav.pack(fill="x", padx=10, pady=5)
        tk.Label(nav, text="Guide:").pack(side="left")
        self.guide_var = tk.StringVar()
        self.guide_combo = ttk.Combobox(nav, textvariable=self.guide_var, width=40, state="readonly")
        self.guide_combo.bind("<<ComboboxSelected>>", self.on_guide_selected)
        self.guide_combo.pack(side="left", padx=5)

        tk.Label(nav, text="Step:").pack(side="left")
        self.step_var = tk.StringVar()
        self.step_combo = ttk.Combobox(nav, textvariable=self.step_var, width=10, state="readonly")
        self.step_combo.bind("<<ComboboxSelected>>", self.on_step_selected)
        self.step_combo.pack(side="left", padx=5)

        # Image preview
        # Image + canvas (for simple mask overlays)
        self.canvas = tk.Canvas(self, bg="gray", relief="groove")
        self.canvas.pack(padx=10, pady=10, fill="both", expand=True)
        self.canvas_image_id = None
        self.photo = None
        self.display_size = (800, 600)
        self.orig_size = (1, 1)
        self.masks = []
        self.selected_mask_idx = None

        # Step info / guidance
        self.step_info_label = tk.Label(
            self,
            text="Guide/Step: n/a\nCaption: n/a",
            wraplength=1100,
            justify="left",
            anchor="w",
        )
        self.step_info_label.pack(padx=10, pady=5, fill="x")

        # Output (scrollable)
        output_frame = tk.Frame(self)
        output_frame.pack(fill="both", padx=10, pady=5)
        tk.Label(output_frame, text="Output to user:").pack(anchor="w")
        self.answer_text = tk.Text(output_frame, height=5, wrap="word")
        scroll = tk.Scrollbar(output_frame, command=self.answer_text.yview)
        self.answer_text.configure(yscrollcommand=scroll.set)
        self.answer_text.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

        # Mask controls
        mask_frame = tk.Frame(self)
        mask_frame.pack(fill="x", padx=10, pady=5)
        tk.Button(mask_frame, text="Propose masks", command=self.propose_masks).pack(
            side="left", padx=5
        )
        tk.Button(mask_frame, text="Accept", command=self.accept_mask).pack(
            side="left", padx=5
        )
        tk.Button(mask_frame, text="Reject", command=self.reject_mask).pack(
            side="left", padx=5
        )
        tk.Label(mask_frame, text="Score >=").pack(side="left")
        self.score_threshold_var = tk.DoubleVar(value=0.30)
        tk.Entry(mask_frame, textvariable=self.score_threshold_var, width=6).pack(
            side="left", padx=3
        )
        self.mask_info_label = tk.Label(mask_frame, text="Mask prompts: n/a")
        self.mask_info_label.pack(fill="x", padx=5, pady=2)
        self.mask_list = tk.Listbox(mask_frame, height=5)
        self.mask_list.bind("<<ListboxSelect>>", self.on_mask_select)
        self.mask_list.pack(fill="x", expand=True, padx=10)

        # Log area
        self.log_text = tk.Text(self, height=10, wrap="word")
        self.log_text.pack(fill="both", expand=True, padx=10, pady=10)

        # Load model in background
        self.append_log(f"Loading model '{MODEL_ID}' ...")
        threading.Thread(target=self.load_model, daemon=True).start()

    def append_log(self, msg: str):
        self.log_text.insert("end", msg + "\n")
        self.log_text.see("end")

    def load_model(self):
        try:
            use_cuda = torch.cuda.is_available()
            self.device = "cuda" if use_cuda else "cpu"
            self.dtype = torch.float16 if use_cuda else torch.float32
            if not use_cuda:
                self.append_log("[WARN] CUDA not available; running on CPU will be slow.")

            self.processor = AutoProcessor.from_pretrained(MODEL_ID, use_fast=True)
            self.model = AutoModelForImageTextToText.from_pretrained(
                MODEL_ID,
                dtype=self.dtype,
                device_map="auto" if use_cuda else {"": self.device},
            )
            self.append_log("Model loaded.")
        except Exception as exc:
            self.append_log(f"[ERROR] Failed to load model: {exc}")
            messagebox.showerror("Error", f"Failed to load model:\n{exc}")

    def open_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.jpg;*.jpeg;*.png;*.webp;*.bmp")]
        )
        if not path:
            return
        self.show_image(path)
        self.link_image_to_metadata(path)

    def show_image(self, path: str):
        try:
            img = Image.open(path).convert("RGB")
            thumb = img.copy()
            thumb.thumbnail((1100, 700))
            self.photo = ImageTk.PhotoImage(thumb)
            self.orig_size = img.size
            self.display_size = (thumb.width, thumb.height)
            self.canvas.config(width=thumb.width, height=thumb.height)
            self.canvas.delete("all")
            self.canvas_image_id = self.canvas.create_image(
                0, 0, anchor="nw", image=self.photo
            )
            self.masks = []
            self.mask_list.delete(0, "end")
            self.image_path = path
            self.append_log(f"Loaded image: {path}")
        except Exception as exc:
            messagebox.showerror("Error", f"Failed to open image: {exc}")

    def load_metadata(self):
        path = filedialog.askopenfilename(
            title="Select metadata.jsonl",
            filetypes=[("JSONL", "*.jsonl"), ("All files", "*.*")],
            initialdir=str(pathlib.Path(DEFAULT_METADATA).parent),
        )
        if not path:
            return
        try:
            meta_path = pathlib.Path(path)
            self.data_dir = meta_path.parent
            self.metadata = []
            self.guide_index.clear()
            with open(meta_path, "r", encoding="utf-8") as infile:
                for line in infile:
                    row = json.loads(line)
                    gid = str(row.get("guide_id"))
                    sid = str(row.get("step_id"))
                    self.metadata.append(row)
                    self.guide_index[gid][sid].append(row)
            guides_sorted = sorted(self.guide_index.keys(), key=lambda x: int(x) if x.isdigit() else x)
            self.guide_combo["values"] = guides_sorted
            if guides_sorted:
                self.guide_combo.current(0)
                self.on_guide_selected()
            self.append_log(f"Loaded metadata: {len(self.metadata)} rows from {path}")
        except Exception as exc:
            messagebox.showerror("Error", f"Failed to load metadata: {exc}")

    def on_guide_selected(self, event=None):
        gid = self.guide_var.get()
        if not gid or gid not in self.guide_index:
            return
        steps = sorted(self.guide_index[gid].keys(), key=lambda x: int(x) if x.isdigit() else x)
        self.step_combo["values"] = steps
        if steps:
            self.step_combo.current(0)
            self.on_step_selected()

    def on_step_selected(self, event=None):
        gid = self.guide_var.get()
        sid = self.step_var.get()
        if not gid or not sid:
            return
        entries = self.guide_index.get(gid, {}).get(sid, [])
        if not entries:
            return
        entry = entries[0]  # first image for this step
        self.current_entry = entry
        self.step_info_label.config(
            text=(
                f"Guide: {entry.get('guide_title','N/A')} (ID {gid}) | Step ID: {sid}\n"
                f"Caption: {entry.get('caption','')}"
            )
        )
        self.current_entry = entry
        if self.data_dir:
            img_path = (self.data_dir / entry["image"]).resolve()
            if img_path.exists():
                self.show_image(str(img_path))
            else:
                self.append_log(f"[WARN] Image not found: {img_path}")

    def link_image_to_metadata(self, path: str):
        """Attempt to link a chosen image to metadata (guide/step) and update selectors."""
        if not self.metadata:
            # Try auto-load metadata from default or from image parent
            candidate = pathlib.Path(DEFAULT_METADATA)
            img_path = pathlib.Path(path).resolve()
            # If image is under .../images/..., set data_dir to parent of images
            for parent in img_path.parents:
                if parent.name.lower() == "images":
                    candidate = parent.parent / "metadata.jsonl"
                    break
            if candidate.exists():
                try:
                    self.data_dir = candidate.parent
                    self.metadata = []
                    self.guide_index.clear()
                    with open(candidate, "r", encoding="utf-8") as infile:
                        for line in infile:
                            row = json.loads(line)
                            gid = str(row.get("guide_id"))
                            sid = str(row.get("step_id"))
                            self.metadata.append(row)
                            self.guide_index[gid][sid].append(row)
                    guides_sorted = sorted(
                        self.guide_index.keys(),
                        key=lambda x: int(x) if x.isdigit() else x,
                    )
                    self.guide_combo["values"] = guides_sorted
                    self.append_log(f"[INFO] Auto-loaded metadata from {candidate}")
                except Exception as exc:
                    self.append_log(f"[WARN] Auto-load metadata failed: {exc}")
        if not self.metadata or not self.data_dir:
            return
        try:
            p = pathlib.Path(path).resolve()
            rel = p.relative_to(self.data_dir)
            rel_str = str(rel).replace("\\", "/")
        except Exception:
            rel_str = p.name

        matches = [
            row
            for row in self.metadata
            if row.get("image") == rel_str
            or os.path.basename(row.get("image", "")) == os.path.basename(path)
        ]
        if not matches:
            self.append_log(f"[INFO] No metadata match for image {path}")
            return
        row = matches[0]
        self.current_entry = row
        gid = str(row.get("guide_id"))
        sid = str(row.get("step_id"))
        if gid in self.guide_index:
            self.guide_combo.set(gid)
            self.on_guide_selected()
            if sid in self.guide_index[gid]:
                self.step_combo.set(sid)
                self.on_step_selected()
            self.append_log(f"[INFO] Linked image to guide {gid}, step {sid}")
        self.step_info_label.config(
            text=(
                f"Guide: {row.get('guide_title','N/A')} (ID {gid}) | Step ID: {sid}\n"
                f"Caption: {row.get('caption','')}"
            )
        )

    def run_vqa(self):
        if self.model is None or self.processor is None:
            messagebox.showinfo("Please wait", "Model is still loading.")
            return
        if not self.image_path:
            messagebox.showinfo("No image", "Please open an image first.")
            return
        question = self.question_var.get().strip()
        if not question:
            messagebox.showinfo("No question", "Please enter a question.")
            return
        max_tokens = max(8, int(self.max_tokens_var.get()))
        threading.Thread(
            target=self._run_vqa_worker, args=(self.image_path, question, max_tokens), daemon=True
        ).start()

    def _run_vqa_worker(self, image_path: str, question: str, max_tokens: int):
        try:
            img = Image.open(image_path).convert("RGB")
            prompt_text = (
                self.build_annotation_prompt(self.current_entry)
                if self.current_entry
                else question
            )
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ]
            prompt = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            self.append_log(f"Running VQA (max_new_tokens={max_tokens}) ...")
            inputs = self.processor(text=prompt, images=img, return_tensors="pt").to(
                self.model.device
            )
            output = self.model.generate(**inputs, max_new_tokens=max_tokens)
            raw_answer = self.processor.decode(output[0], skip_special_tokens=True)
            if "ASSISTANT:" in raw_answer:
                answer = raw_answer.split("ASSISTANT:", 1)[1].strip()
            else:
                answer = raw_answer.strip()
            # Ensure UI updates happen on the main thread
            self.after(
                0,
                lambda: (
                    self.answer_text.delete("1.0", "end"),
                    self.answer_text.insert("end", answer),
                ),
            )
        except Exception as exc:
            self.after(0, lambda: self.append_log(f"[ERROR] {exc}"))
            self.after(
                0, lambda: messagebox.showerror("Error", f"Failed to run VQA:\n{exc}")
            )

    def build_annotation_prompt(self, entry: dict) -> str:
        if not entry:
            return ""
        guide_title = entry.get("guide_title", "N/A")
        guide_id = entry.get("guide_id", "N/A")
        step_id = entry.get("step_id", "N/A")
        caption = entry.get("caption", "")
        language = entry.get("language", "N/A")
        return (
            "You are an annotation assistant for repair guides. Given the step caption and the image, "
            "propose what the user should highlight/label.\n\n"
            f"Context:\n"
            f"- Guide title: {guide_title}\n"
            f"- Guide ID: {guide_id}, Step ID: {step_id}\n"
            f"- Step caption: {caption}\n"
            f"- Language: {language}\n\n"
            "Instructions:\n"
            "- Return 1-3 bullets starting with 'Highlight ...' that are relevant to this step.\n"
            "- Include a short class/label in parentheses (e.g., 'Highlight keyboard ribbon connector (class: connector), center-right near logic board').\n"
            "- Use short, actionable bullets: what it is + where it is in the image.\n"
            "- Only refer to items that are relevant to the step caption; ignore unrelated objects. If unsure something is visible, say 'uncertain'.\n"
            "- If multiple instances are visible, state the expected count (e.g., 'Highlight screw heads (count: 4) at top edge').\n"
            "- Keep it concise (<= 25 words per bullet).\n"
            "- If uncertain, say 'uncertain' rather than hallucinating.\n"
            "- Output only bullets, no extra text."
        )

    # Mask proposal with backend (fallback to dummy boxes if backend fails)
    def propose_masks(self):
        """Request masks from backend per VQA label; fallback to dummy boxes on failure."""
        self.clear_masks()
        lines = [
            ln.strip()
            for ln in self.answer_text.get("1.0", "end").splitlines()
            if ln.strip()
        ]
        if not lines:
            self.append_log("[INFO] No answer text to propose masks from.")
            return
        labels = [ln.lstrip("*- ").strip() for ln in lines]
        self.mask_info_label.config(text=f"Mask prompts: {labels}")
        self.selected_mask_idx = None
        w, h = self.display_size
        try:
            threshold = float(self.score_threshold_var.get())
        except Exception:
            threshold = 0.0
        proposals: List[Dict[str, Any]] = []
        try:
            if self.image_path is None:
                raise RuntimeError("No image loaded for mask proposal")
            payload = {
                "image_path": self.image_path,
                "labels": labels,
                "box_threshold": 0.25,
                "text_threshold": 0.25,
                "top_k": 3,
                "return_masks": True,
            }
            resp = requests.post(MASK_BACKEND_URL, json=payload, timeout=30)
            resp.raise_for_status()
            proposals = [
                p for p in resp.json() if p.get("score", 0.0) >= threshold
            ]
            self.append_log(
                f"[INFO] Masks from backend: {len(proposals)} proposals (score>={threshold})"
            )
        except Exception as exc:
            self.append_log(f"[WARN] Backend mask proposal failed: {exc}. Using dummy boxes.")
            for idx, lbl in enumerate(labels):
                box_w = max(50, w // 4)
                box_h = max(50, h // 4)
                x1 = (10 + idx * 40) % max(1, w - box_w)
                y1 = (10 + idx * 40) % max(1, h - box_h)
                proposals.append(
                    {
                        "label": lbl,
                        "score": 0.5,
                        "bbox": [x1, y1, x1 + box_w, y1 + box_h],
                        "polygon": None,
                    }
                )
        sx = self.display_size[0] / max(1, self.orig_size[0])
        sy = self.display_size[1] / max(1, self.orig_size[1])
        for idx, prop in enumerate(proposals):
            lbl = prop.get("label", f"item-{idx}")
            score = prop.get("score", 0.0)
            poly = prop.get("polygon")
            rect_id = None
            text_id = None
            base_color = prop.get("color", "yellow")
            if poly:
                scaled_poly = [(x * sx, y * sy) for x, y in poly]
                flat = [coord for xy in scaled_poly for coord in xy]
                rect_id = self.canvas.create_polygon(
                    *flat, outline=base_color, fill="", width=2
                )
                xs = [xy[0] for xy in scaled_poly]
                ys = [xy[1] for xy in scaled_poly]
                anchor_x = min(xs)
                anchor_y = min(ys)
            else:
                x1, y1, x2, y2 = prop.get("bbox", [10, 10, 60, 60])
                x1, y1, x2, y2 = x1 * sx, y1 * sy, x2 * sx, y2 * sy
                rect_id = self.canvas.create_rectangle(
                    x1, y1, x2, y2, outline=base_color, width=2
                )
                anchor_x, anchor_y = x1, y1
            text_id = self.canvas.create_text(
                anchor_x + 5,
                anchor_y + 10,
                anchor="nw",
                fill=base_color,
                text=f"{idx}: pending (score {score:.2f})",
            )
            self.masks.append(
                {
                    "label": lbl,
                    "score": score,
                    "status": "pending",
                    "rect_id": rect_id,
                    "text_id": text_id,
                    "color": base_color,
                }

            )
            self.mask_list.insert("end", f"{idx}: {lbl} [pending] score={score:.2f}")
    def accept_mask(self):
        sel = self.mask_list.curselection()
        if not sel:
            return
        idx = sel[0]
        self.update_mask_status(idx, "accepted", color="green")

    def reject_mask(self):
        sel = self.mask_list.curselection()
        if not sel:
            return
        idx = sel[0]
        self.update_mask_status(idx, "rejected", color="red")

    def update_mask_status(self, idx: int, status: str, color: str):
        if idx < 0 or idx >= len(self.masks):
            return
        m = self.masks[idx]
        m["status"] = status
        m["color"] = color
        self.selected_mask_idx = idx
        self.canvas.itemconfig(m["rect_id"], outline=color)
        self.canvas.itemconfig(m["text_id"], fill=color, text=f"{idx}: {status}")
        self.mask_list.delete(idx)
        self.mask_list.insert(idx, f"{idx}: {m['label']} [{status}]")

    def clear_masks(self):
        for m in self.masks:
            try:
                self.canvas.delete(m.get("rect_id"))
                self.canvas.delete(m.get("text_id"))
            except Exception:
                pass
        self.masks = []
        self.mask_list.delete(0, "end")
        self.selected_mask_idx = None

    def on_mask_select(self, event=None):
        sel = self.mask_list.curselection()
        if not sel:
            return
        idx = sel[0]
        if self.selected_mask_idx is not None and 0 <= self.selected_mask_idx < len(
            self.masks
        ):
            prev = self.masks[self.selected_mask_idx]
            self.canvas.itemconfig(prev["rect_id"], outline=prev.get("color", "yellow"))
        self.selected_mask_idx = idx
        m = self.masks[idx]
        self.canvas.itemconfig(m["rect_id"], outline="cyan")



if __name__ == "__main__":
    app = LlavaGUI()
    app.mainloop()

