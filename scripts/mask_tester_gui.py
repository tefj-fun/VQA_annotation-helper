#!/usr/bin/env python3
"""
Simple GUI to test the mask backend (/propose_masks).

- Load an image
- Enter labels (one per line or comma-separated)
- Call backend and visualize returned boxes/polygons
"""

import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
from typing import Any, Dict, List

import requests
from PIL import Image, ImageTk

MASK_BACKEND_URL = os.environ.get("MASK_BACKEND_URL", "http://localhost:8000/propose_masks")


class MaskTesterGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Mask Backend Tester")
        self.geometry("1200x800")

        self.image_path = None
        self.orig_size = (1, 1)
        self.display_size = (800, 600)
        self.photo = None
        self.canvas_image_id = None
        self.masks = []

        # Controls
        top = tk.Frame(self)
        top.pack(fill="x", padx=10, pady=5)

        tk.Button(top, text="Open Image", command=self.open_image).pack(side="left", padx=5)
        tk.Label(top, text="Labels (one per line or comma-separated):").pack(side="left")
        self.labels_text = tk.Text(top, height=3, width=60)
        self.labels_text.pack(side="left", padx=5, pady=5)
        tk.Button(top, text="Propose masks", command=self.propose_masks).pack(side="left", padx=5)

        # Canvas
        self.canvas = tk.Canvas(self, bg="gray", relief="groove")
        self.canvas.pack(padx=10, pady=10, fill="both", expand=True)

        # Log area
        self.log_text = tk.Text(self, height=10, wrap="word")
        self.log_text.pack(fill="both", expand=True, padx=10, pady=10)

    def append_log(self, msg: str):
        self.log_text.insert("end", msg + "\n")
        self.log_text.see("end")

    def open_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.jpg;*.jpeg;*.png;*.webp;*.bmp")]
        )
        if not path:
            return
        try:
            img = Image.open(path).convert("RGB")
        except Exception as exc:
            messagebox.showerror("Error", f"Failed to open image: {exc}")
            return
        self.image_path = path
        self.orig_size = img.size
        thumb = img.copy()
        thumb.thumbnail((1100, 700))
        self.display_size = thumb.size
        self.photo = ImageTk.PhotoImage(thumb)
        self.canvas.config(width=thumb.width, height=thumb.height)
        self.canvas.delete("all")
        self.canvas_image_id = self.canvas.create_image(
            0, 0, anchor="nw", image=self.photo
        )
        self.masks = []
        self.append_log(f"Loaded image: {path}")

    def propose_masks(self):
        if not self.image_path:
            messagebox.showinfo("No image", "Please open an image first.")
            return
        labels_raw = self.labels_text.get("1.0", "end").strip()
        if not labels_raw:
            messagebox.showinfo("No labels", "Enter at least one label.")
            return
        labels = []
        for part in labels_raw.replace(",", "\n").splitlines():
            p = part.strip()
            if p:
                labels.append(p)
        if not labels:
            messagebox.showinfo("No labels", "Enter at least one label.")
            return
        threading.Thread(target=self._propose_worker, args=(labels,), daemon=True).start()

    def _propose_worker(self, labels: List[str]):
        payload = {
            "image_path": self.image_path,
            "labels": labels,
            "box_threshold": 0.25,
            "text_threshold": 0.25,
            "top_k": 3,
            "return_masks": True,
        }
        try:
            self.append_log(f"[INFO] POST {MASK_BACKEND_URL} with labels: {labels}")
            resp = requests.post(MASK_BACKEND_URL, json=payload, timeout=30)
            resp.raise_for_status()
            proposals = resp.json()
            self.after(0, lambda: self.draw_masks(proposals))
        except Exception as exc:
            self.append_log(f"[ERROR] Backend call failed: {exc}")

    def draw_masks(self, proposals: List[Dict[str, Any]]):
        self.canvas.delete("mask")
        self.masks = []
        if not proposals:
            self.append_log("[INFO] No proposals returned.")
            return
        sx = self.display_size[0] / self.orig_size[0]
        sy = self.display_size[1] / self.orig_size[1]
        for idx, prop in enumerate(proposals):
            lbl = prop.get("label", f"item-{idx}")
            score = prop.get("score", 0.0)
            poly = prop.get("polygon")
            if poly:
                scaled = []
                for x, y in poly:
                    scaled.append((x * sx, y * sy))
                flat = [coord for xy in scaled for coord in xy]
                self.canvas.create_polygon(
                    *flat, outline="yellow", fill="", width=2, tags="mask"
                )
                xs = [p[0] for p in scaled]
                ys = [p[1] for p in scaled]
                ax, ay = min(xs), min(ys)
            else:
                x1, y1, x2, y2 = prop.get("bbox", [10, 10, 60, 60])
                x1, y1, x2, y2 = x1 * sx, y1 * sy, x2 * sx, y2 * sy
                self.canvas.create_rectangle(
                    x1, y1, x2, y2, outline="yellow", width=2, tags="mask"
                )
                ax, ay = x1, y1
            self.canvas.create_text(
                ax + 5,
                ay + 10,
                anchor="nw",
                fill="yellow",
                text=f"{idx}: {lbl} (score {score:.2f})",
                tags="mask",
            )
            self.masks.append(prop)
        self.append_log(f"[INFO] Drew {len(proposals)} proposals.")


if __name__ == "__main__":
    app = MaskTesterGUI()
    app.mainloop()
