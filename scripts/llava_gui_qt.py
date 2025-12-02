#!/usr/bin/env python3
"""
PyQt5 GUI for LLaVA VQA + mask proposals.
Features:
- Load image or pick from metadata.jsonl (iFixit).
- Run VQA with a question/max tokens.
- Propose masks via backend (MASK_BACKEND_URL) with score threshold.
- Click mask list to highlight; Accept/Reject to keep color/state.
"""

import json
import io
import os
import pathlib
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional

import requests
from PIL import Image
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from PyQt5 import QtCore, QtGui, QtWidgets


MODEL_ID = "llava-hf/llava-1.5-7b-hf"
DEFAULT_METADATA = r"D:\Joseph\FUN\ifixit\datasets\ifixit_blip_en\metadata.jsonl"
MASK_BACKEND_URL = os.environ.get("MASK_BACKEND_URL", "http://localhost:8000/propose_masks")


class Worker(QtCore.QThread):
    finished = QtCore.pyqtSignal(object)
    error = QtCore.pyqtSignal(str)

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

    def run(self):
        try:
            res = self.fn(*self.args, **self.kwargs)
            self.finished.emit(res)
        except Exception as exc:  # pragma: no cover - safety path
            self.error.emit(str(exc))


class LlavaQt(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LLaVA VQA (PyQt)")
        self.resize(1300, 900)

        self.model = None
        self.processor = None
        self.device = "cpu"
        self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        self.metadata: List[dict] = []
        self.guide_index = defaultdict(lambda: defaultdict(list))
        self.current_entry: Optional[dict] = None
        self.path_index: Dict[str, dict] = {}
        self.data_dir = None
        self.image_list: List[str] = []
        self.image_pos: int = -1

        self.image_path = None
        self.orig_size = (1, 1)
        self.display_size = (1, 1)
        self.pixmap_item = None

        self.masks: List[Dict[str, Any]] = []
        self.selected_mask_idx: Optional[int] = None
        self.list_index_map: List[int] = []

        self._build_ui()
        self._load_model_background()

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_h = QtWidgets.QHBoxLayout(central)

        # Left side: image + log
        left_frame = QtWidgets.QWidget()
        left_v = QtWidgets.QVBoxLayout(left_frame)
        self.scene = QtWidgets.QGraphicsScene()
        self.view = QtWidgets.QGraphicsView(self.scene)
        self.view.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.SmoothPixmapTransform)
        left_v.addWidget(self.view, stretch=4)
        left_v.addWidget(QtWidgets.QLabel("Log:"))
        self.log_text = QtWidgets.QPlainTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(150)
        left_v.addWidget(self.log_text, stretch=1)
        main_h.addWidget(left_frame, stretch=3)

        # Right side in a scroll area
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        right_container = QtWidgets.QWidget()
        right_v = QtWidgets.QVBoxLayout(right_container)

        # Top controls
        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel("Question:"))
        self.question_edit = QtWidgets.QLineEdit("Suggest what to highlight (counts + positions).")
        top.addWidget(self.question_edit, stretch=2)
        top.addWidget(QtWidgets.QLabel("Max tokens:"))
        self.max_tokens_spin = QtWidgets.QSpinBox()
        self.max_tokens_spin.setRange(8, 512)
        self.max_tokens_spin.setValue(64)
        top.addWidget(self.max_tokens_spin)
        right_v.addLayout(top)

        btn_row = QtWidgets.QHBoxLayout()
        self.load_meta_btn = QtWidgets.QPushButton("Load metadata")
        self.load_meta_btn.clicked.connect(self.load_metadata)
        btn_row.addWidget(self.load_meta_btn)
        self.open_img_btn = QtWidgets.QPushButton("Open image")
        self.open_img_btn.clicked.connect(self.open_image)
        btn_row.addWidget(self.open_img_btn)
        self.prev_img_btn = QtWidgets.QPushButton("Prev")
        self.prev_img_btn.clicked.connect(self.prev_image)
        self.next_img_btn = QtWidgets.QPushButton("Next")
        self.next_img_btn.clicked.connect(self.next_image)
        btn_row.addWidget(self.prev_img_btn)
        btn_row.addWidget(self.next_img_btn)
        self.run_vqa_btn = QtWidgets.QPushButton("Run VQA")
        self.run_vqa_btn.clicked.connect(self.run_vqa)
        btn_row.addWidget(self.run_vqa_btn)
        btn_row.addStretch()
        right_v.addLayout(btn_row)

        # Guide/step selectors
        nav = QtWidgets.QHBoxLayout()
        nav.addWidget(QtWidgets.QLabel("Guide:"))
        self.guide_combo = QtWidgets.QComboBox()
        self.guide_combo.currentTextChanged.connect(self.on_guide_selected)
        nav.addWidget(self.guide_combo, stretch=2)
        nav.addWidget(QtWidgets.QLabel("Step:"))
        self.step_combo = QtWidgets.QComboBox()
        self.step_combo.currentTextChanged.connect(self.on_step_selected)
        nav.addWidget(self.step_combo)
        right_v.addLayout(nav)

        # Step info
        self.step_info = QtWidgets.QLabel("Guide/Step: n/a\nCaption: n/a")
        self.step_info.setWordWrap(True)
        right_v.addWidget(self.step_info)

        # Output / answer
        right_v.addWidget(QtWidgets.QLabel("Answer:"))
        self.answer_text = QtWidgets.QPlainTextEdit()
        self.answer_text.setMinimumHeight(80)
        self.answer_text.setLineWrapMode(QtWidgets.QPlainTextEdit.WidgetWidth)
        right_v.addWidget(self.answer_text)

        # Mask controls
        mask_row = QtWidgets.QHBoxLayout()
        self.propose_btn = QtWidgets.QPushButton("Propose masks")
        self.propose_btn.clicked.connect(self.propose_masks)
        mask_row.addWidget(self.propose_btn)
        self.accept_btn = QtWidgets.QPushButton("Accept")
        self.accept_btn.clicked.connect(self.accept_mask)
        mask_row.addWidget(self.accept_btn)
        self.reject_btn = QtWidgets.QPushButton("Reject")
        self.reject_btn.clicked.connect(self.reject_mask)
        mask_row.addWidget(self.reject_btn)
        mask_row.addWidget(QtWidgets.QLabel("Score >="))
        self.score_edit = QtWidgets.QDoubleSpinBox()
        self.score_edit.setRange(0.0, 1.0)
        self.score_edit.setSingleStep(0.05)
        self.score_edit.setValue(0.30)
        mask_row.addWidget(self.score_edit)
        right_v.addLayout(mask_row)
        self.mask_info = QtWidgets.QLabel("Mask prompts: n/a")
        self.mask_info.setWordWrap(True)
        right_v.addWidget(self.mask_info)

        self.mask_list = QtWidgets.QListWidget()
        self.mask_list.currentRowChanged.connect(self.on_mask_select)
        self.mask_list.setMinimumHeight(120)
        self.mask_list.setWordWrap(True)
        self.mask_list.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        right_v.addWidget(self.mask_list)

        right_v.addStretch()
        right_container.setFixedWidth(500)
        scroll.setWidget(right_container)
        scroll.setFixedWidth(520)
        scroll.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustToContents)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        main_h.addWidget(scroll, stretch=2)

        # Rejected mask toggle
        toggle_row = QtWidgets.QHBoxLayout()
        self.show_rejected = QtWidgets.QCheckBox("Show rejected masks")
        self.show_rejected.setChecked(False)
        self.show_rejected.stateChanged.connect(self.refresh_mask_list)
        toggle_row.addWidget(self.show_rejected)
        right_v.insertLayout(right_v.count() - 1, toggle_row)

    # Logging helper
    def append_log(self, msg: str):
        self.log_text.appendPlainText(msg)
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())

    # Model loading
    def _load_model_background(self):
        self.append_log(f"Loading model '{MODEL_ID}' ...")
        worker = Worker(self._load_model)
        worker.finished.connect(lambda _: self.append_log("Model loaded."))
        worker.error.connect(lambda e: self.append_log(f"[ERROR] {e}"))
        worker.start()
        self.model_worker = worker

    def _load_model(self):
        use_cuda = torch.cuda.is_available()
        self.device = "cuda" if use_cuda else "cpu"
        self.dtype = torch.float16 if use_cuda else torch.float32
        self.processor = AutoProcessor.from_pretrained(MODEL_ID, use_fast=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID,
            dtype=self.dtype,
            device_map="auto" if use_cuda else {"": self.device},
        )

    # Metadata
    def load_metadata(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select metadata.jsonl", str(pathlib.Path(DEFAULT_METADATA).parent), "JSONL (*.jsonl)"
        )
        if not path:
            return
        self.metadata.clear()
        self.guide_index.clear()
        self.path_index.clear()
        self.data_dir = pathlib.Path(path).parent
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                gid = str(obj.get("guide_id", ""))
                step_id = obj.get("step_id")
                self.metadata.append(obj)
                self.guide_index[gid][step_id].append(obj)
                img_rel = obj.get("image_relpath") or obj.get("image", "")
                if img_rel:
                    abs_path = (self.data_dir / img_rel).resolve()
                    self.path_index[str(abs_path)] = obj
        self.guide_combo.clear()
        self.guide_combo.addItems(sorted(self.guide_index.keys()))
        # Build image list from metadata for navigation
        self.image_list = []
        for obj in self.metadata:
            img_rel = obj.get("image_relpath") or obj.get("image", "")
            if img_rel:
                abs_path = str((self.data_dir / img_rel).resolve())
                self.image_list.append(abs_path)
        self.image_pos = -1
        self.append_log(f"Loaded metadata: {len(self.metadata)} rows")

    def on_guide_selected(self, gid: str):
        if not gid:
            return
        steps = sorted(self.guide_index[gid].keys())
        self.step_combo.clear()
        self.step_combo.addItems([str(s) for s in steps])

    def on_step_selected(self, step: str):
        if not step:
            return
        gid = self.guide_combo.currentText()
        entries = self.guide_index.get(gid, {}).get(int(step), [])
        if not entries:
            return
        entry = entries[0]
        self.current_entry = entry
        caption = entry.get("caption", "")
        self.step_info.setText(f"Guide: {gid} Step: {step}\nCaption: {caption}")
        # Link image if present
        img_rel = entry.get("image_relpath") or entry.get("image", "")
        img_path = str(self.data_dir / img_rel) if img_rel else ""
        if img_path and os.path.exists(img_path):
            self.set_image_path(img_path)

    # Image handling
    def open_image(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select image",
            "",
            "Images (*.jpg *.jpeg *.png *.webp *.bmp)",
        )
        if not path:
            return
        resolved = str(pathlib.Path(path).resolve())
        entry = self.path_index.get(resolved)
        if entry:
            self.current_entry = entry
            gid = str(entry.get("guide_id", ""))
            step = str(entry.get("step_id", ""))
            caption = entry.get("caption", "")
            # sync combos without spamming signals
            if gid and self.guide_combo.count() > 0:
                idx = self.guide_combo.findText(gid)
                if idx >= 0:
                    self.guide_combo.blockSignals(True)
                    self.guide_combo.setCurrentIndex(idx)
                    self.guide_combo.blockSignals(False)
                    steps = sorted(self.guide_index[gid].keys())
                    self.step_combo.blockSignals(True)
                    self.step_combo.clear()
                    self.step_combo.addItems([str(s) for s in steps])
                    step_idx = self.step_combo.findText(str(step))
                    if step_idx >= 0:
                        self.step_combo.setCurrentIndex(step_idx)
                    self.step_combo.blockSignals(False)
            self.step_info.setText(f"Guide: {gid} Step: {step}\nCaption: {caption}")
        else:
            self.current_entry = None
        self.set_image_path(path)

    def set_image_path(self, path: str):
        """Set current image path, update position in list, and show image."""
        resolved = str(pathlib.Path(path).resolve())
        if resolved in self.image_list:
            self.image_pos = self.image_list.index(resolved)
        self.show_image(resolved)

    def show_image(self, path: str):
        img = Image.open(path).convert("RGB")
        self.orig_size = img.size
        thumb = img.copy()
        thumb.thumbnail((1100, 750))
        self.display_size = (thumb.width, thumb.height)
        buf = io.BytesIO()
        thumb.save(buf, format="PNG")
        buf.seek(0)
        pix = QtGui.QPixmap()
        pix.loadFromData(buf.getvalue(), "PNG")
        self.clear_masks()
        self.scene.clear()
        self.pixmap_item = self.scene.addPixmap(pix)
        self.scene.setSceneRect(QtCore.QRectF(pix.rect()))
        self.image_path = path
        if self.current_entry:
            caption = self.current_entry.get("caption", "")
            gid = self.current_entry.get("guide_id", "n/a")
            step = self.current_entry.get("step_id", "n/a")
            self.step_info.setText(f"Guide: {gid} Step: {step}\nCaption: {caption}")
        else:
            fname = os.path.basename(path)
            self.step_info.setText(f"Guide: n/a Step: n/a\nCaption: (manual image)\nFile: {fname}")
        self.append_log(f"Loaded image: {path}")

    def prev_image(self):
        if not self.image_list:
            return
        if self.image_pos <= 0:
            self.image_pos = 0
        else:
            self.image_pos -= 1
        self._load_by_index()

    def next_image(self):
        if not self.image_list:
            return
        if self.image_pos < 0:
            self.image_pos = 0
        elif self.image_pos >= len(self.image_list) - 1:
            self.image_pos = len(self.image_list) - 1
        else:
            self.image_pos += 1
        self._load_by_index()

    def _load_by_index(self):
        if self.image_pos < 0 or self.image_pos >= len(self.image_list):
            return
        path = self.image_list[self.image_pos]
        entry = self.path_index.get(path)
        if entry:
            self.current_entry = entry
            gid = str(entry.get("guide_id", ""))
            step = str(entry.get("step_id", ""))
            caption = entry.get("caption", "")
            # sync combos quietly
            if gid and self.guide_combo.count() > 0:
                idx = self.guide_combo.findText(gid)
                if idx >= 0:
                    self.guide_combo.blockSignals(True)
                    self.guide_combo.setCurrentIndex(idx)
                    self.guide_combo.blockSignals(False)
                    steps = sorted(self.guide_index[gid].keys())
                    self.step_combo.blockSignals(True)
                    self.step_combo.clear()
                    self.step_combo.addItems([str(s) for s in steps])
                    step_idx = self.step_combo.findText(str(step))
                    if step_idx >= 0:
                        self.step_combo.setCurrentIndex(step_idx)
                    self.step_combo.blockSignals(False)
            self.step_info.setText(f"Guide: {gid} Step: {step}\nCaption: {caption}")
        else:
            self.current_entry = None
        self.show_image(path)

    # VQA
    def run_vqa(self):
        if not self.model or not self.processor:
            self.append_log("[WARN] Model not ready yet.")
            return
        if not self.image_path:
            self.append_log("[WARN] No image loaded.")
            return
        question = self.question_edit.text().strip()
        if not question:
            self.append_log("[WARN] Question is empty.")
            return
        max_tokens = int(self.max_tokens_spin.value())
        self.append_log(f"Running VQA (max_new_tokens={max_tokens}) ...")
        worker = Worker(self._run_vqa_worker, question, max_tokens)
        worker.finished.connect(self._on_vqa_done)
        worker.error.connect(lambda e: self.append_log(f"[ERROR] {e}"))
        worker.start()
        self.vqa_worker = worker

    def _run_vqa_worker(self, question: str, max_tokens: int):
        img = Image.open(self.image_path).convert("RGB")
        messages = [
            {"role": "system", "content": "You are a helpful vision assistant."},
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]},
        ]
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=prompt,
            images=img,
            return_tensors="pt",
        ).to(self.model.device)
        output = self.model.generate(**inputs, max_new_tokens=max_tokens)
        raw = self.processor.decode(output[0], skip_special_tokens=True)
        if "ASSISTANT:" in raw:
            answer = raw.split("ASSISTANT:", 1)[1].strip()
        else:
            answer = raw.strip()
        return answer

    def _on_vqa_done(self, answer: str):
        self.answer_text.setPlainText(answer)
        self.append_log("[INFO] VQA finished.")

    # Masks
    def propose_masks(self):
        self.clear_masks()
        self.list_index_map = []
        lines = [ln.strip() for ln in self.answer_text.toPlainText().splitlines() if ln.strip()]
        if not lines:
            self.append_log("[INFO] No answer text to propose masks from.")
            return
        labels = [ln.lstrip("*- ").strip() for ln in lines]
        self.mask_info.setText(f"Mask prompts: {labels}")
        try:
            threshold = float(self.score_edit.value())
        except Exception:
            threshold = 0.0
        if not self.image_path:
            self.append_log("[WARN] No image loaded for mask proposal.")
            return
        payload = {
            "image_path": self.image_path,
            "labels": labels,
            "box_threshold": 0.25,
            "text_threshold": 0.25,
            "top_k": 3,
            "return_masks": True,
        }
        try:
            resp = requests.post(MASK_BACKEND_URL, json=payload, timeout=30)
            resp.raise_for_status()
            proposals = [p for p in resp.json() if p.get("score", 0.0) >= threshold]
            self.append_log(f"[INFO] Masks from backend: {len(proposals)} proposals (score>={threshold})")
        except Exception as exc:  # pragma: no cover - network path
            self.append_log(f"[WARN] Backend mask proposal failed: {exc}. Using dummy boxes.")
            w, h = self.display_size
            proposals = []
            for idx, lbl in enumerate(labels):
                box_w = max(50, w // 4)
                box_h = max(50, h // 4)
                x1 = (10 + idx * 40) % max(1, w - box_w)
                y1 = (10 + idx * 40) % max(1, h - box_h)
                proposals.append(
                    {"label": lbl, "score": 0.5, "bbox": [x1, y1, x1 + box_w, y1 + box_h], "polygon": None}
                )
        sx = self.display_size[0] / max(1, self.orig_size[0])
        sy = self.display_size[1] / max(1, self.orig_size[1])
        for idx, prop in enumerate(proposals):
            lbl = prop.get("label", f"item-{idx}")
            score = prop.get("score", 0.0)
            display_lbl = lbl if len(lbl) <= 80 else lbl[:80] + "…"
            poly = prop.get("polygon")
            base_color = prop.get("color", "yellow")
            if poly:
                scaled = [QtCore.QPointF(x * sx, y * sy) for x, y in poly]
                poly_item = self.scene.addPolygon(QtGui.QPolygonF(scaled), QtGui.QPen(QtGui.QColor(base_color), 2))
                rect_item = poly_item
                anchor_x = min(p.x() for p in scaled)
                anchor_y = min(p.y() for p in scaled)
            else:
                x1, y1, x2, y2 = prop.get("bbox", [10, 10, 60, 60])
                x1, y1, x2, y2 = x1 * sx, y1 * sy, x2 * sx, y2 * sy
                rect_item = self.scene.addRect(x1, y1, x2 - x1, y2 - y1, QtGui.QPen(QtGui.QColor(base_color), 2))
                anchor_x, anchor_y = x1, y1
            text_item = self.scene.addText(f"{idx}: pending (score {score:.2f})")
            text_item.setDefaultTextColor(QtGui.QColor(base_color))
            text_item.setPos(anchor_x + 5, anchor_y + 5)
            self.masks.append(
                {
                    "label": lbl,
                    "score": score,
                    "status": "pending",
                    "item": rect_item,
                    "text_item": text_item,
                    "color": base_color,
                }
            )
            self.mask_list.addItem(f"{idx}: {display_lbl} [pending] score={score:.2f}")
        self.refresh_mask_list()

    def on_mask_select(self, idx: int):
        if idx < 0 or idx >= len(self.list_index_map):
            return
        mask_idx = self.list_index_map[idx]
        if self.selected_mask_idx is not None and 0 <= self.selected_mask_idx < len(self.masks):
            prev = self.masks[self.selected_mask_idx]
            prev["item"].setPen(QtGui.QPen(QtGui.QColor(prev.get("color", "yellow")), 2))
        self.selected_mask_idx = mask_idx
        m = self.masks[mask_idx]
        m["item"].setPen(QtGui.QPen(QtGui.QColor("cyan"), 2))

    def accept_mask(self):
        row = self.mask_list.currentRow()
        if row < 0 or row >= len(self.list_index_map):
            return
        idx = self.list_index_map[row]
        self.update_mask_status(idx, "accepted", "green")

    def reject_mask(self):
        row = self.mask_list.currentRow()
        if row < 0 or row >= len(self.list_index_map):
            return
        idx = self.list_index_map[row]
        self.update_mask_status(idx, "rejected", "red")

    def update_mask_status(self, idx: int, status: str, color: str):
        if idx < 0 or idx >= len(self.masks):
            return
        m = self.masks[idx]
        m["status"] = status
        m["color"] = color
        self.selected_mask_idx = idx
        m["item"].setPen(QtGui.QPen(QtGui.QColor(color), 2))
        m["text_item"].setDefaultTextColor(QtGui.QColor(color))
        m["text_item"].setPlainText(f"{idx}: {status}")
        display_lbl = m["label"] if len(m["label"]) <= 80 else m["label"][:80] + "…"
        # Refresh list/visibility based on toggle
        self.refresh_mask_list()
        self._refresh_mask_visibility()

    def clear_masks(self):
        for m in self.masks:
            try:
                item = m.get("item")
                try:
                    scene_ref = item.scene() if item is not None else None
                except Exception:
                    scene_ref = None
                if item is not None and scene_ref is self.scene:
                    self.scene.removeItem(item)
                text_item = m.get("text_item")
                try:
                    text_scene = text_item.scene() if text_item is not None else None
                except Exception:
                    text_scene = None
                if text_item is not None and text_scene is self.scene:
                    self.scene.removeItem(text_item)
            except Exception:
                pass
        self.masks = []
        self.mask_list.clear()
        self.selected_mask_idx = None
        self.list_index_map = []
        self.mask_list.clearSelection()

    def refresh_mask_list(self):
        """Repopulate the listbox based on show_rejected toggle."""
        self.mask_list.clear()
        self.list_index_map = []
        for idx, m in enumerate(self.masks):
            if m.get("status") == "rejected" and not self.show_rejected.isChecked():
                continue
            display_lbl = m["label"] if len(m["label"]) <= 80 else m["label"][:80] + "…"
            status = m.get("status", "pending")
            score = m.get("score", 0.0)
            self.mask_list.addItem(f"{idx}: {display_lbl} [{status}] score={score:.2f}")
            self.list_index_map.append(idx)
        self._refresh_mask_visibility()
        self.mask_list.clearSelection()
        self.selected_mask_idx = None

    def _refresh_mask_visibility(self):
        """Show/hide overlays for rejected masks based on toggle."""
        show_rej = self.show_rejected.isChecked()
        for m in self.masks:
            vis = m.get("status") != "rejected" or show_rej
            item = m.get("item")
            if item is not None:
                item.setVisible(vis)
            t_item = m.get("text_item")
            if t_item is not None:
                t_item.setVisible(vis)
        # If current selection became hidden, clear it
        if self.selected_mask_idx is not None:
            sel_mask = self.masks[self.selected_mask_idx]
            if sel_mask.get("status") == "rejected" and not show_rej:
                self.selected_mask_idx = None
                self.mask_list.clearSelection()


def main():
    app = QtWidgets.QApplication(sys.argv)
    gui = LlavaQt()
    gui.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
