from __future__ import annotations

import base64
import io
import json
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import cv2
import numpy as np


@dataclass
class OCRResult:
    backend: str
    text: str
    confidence: float = 0.0
    details: str = ""


class TemplateSymbolRecognizer:
    """Fast offline fallback recognizer for digits and uppercase A-Z."""

    def __init__(self):
        self.templates = self._build_templates()

    def _build_templates(self):
        templates = {}
        chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        fonts = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_COMPLEX, cv2.FONT_HERSHEY_TRIPLEX]
        for ch in chars:
            variants = []
            for font in fonts:
                for scale in (1.35, 1.65, 1.95, 2.25):
                    for thickness in (3, 4, 5):
                        img = np.zeros((112, 112), dtype=np.uint8)
                        (tw, th), _ = cv2.getTextSize(ch, font, scale, thickness)
                        x = max(1, (112 - tw) // 2)
                        y = max(th + 2, (112 + th) // 2)
                        cv2.putText(img, ch, (x, y), font, scale, 255, thickness, cv2.LINE_AA)
                        variants.append(self.normalize(img))
            templates[ch] = variants
        return templates

    def normalize(self, img: np.ndarray) -> np.ndarray:
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        pts = cv2.findNonZero(img)
        if pts is None:
            return np.zeros((64, 64), dtype=np.float32)
        x, y, w, h = cv2.boundingRect(pts)
        crop = img[y : y + h, x : x + w]
        side = max(w, h) + 22
        square = np.zeros((side, side), dtype=np.uint8)
        ox, oy = (side - w) // 2, (side - h) // 2
        square[oy : oy + h, ox : ox + w] = crop
        norm = cv2.resize(square, (64, 64), interpolation=cv2.INTER_AREA)
        return (norm > 70).astype(np.float32)

    def recognize_symbol(self, ink_binary: np.ndarray) -> OCRResult:
        x = self.normalize(ink_binary)
        best_ch, best_score = "?", 1e9
        for ch, variants in self.templates.items():
            score = min(float(np.mean((x - t) ** 2)) for t in variants)
            if score < best_score:
                best_ch, best_score = ch, score
        confidence = max(0.0, min(1.0, 1.0 - best_score * 3.0))
        return OCRResult("template", best_ch, confidence, f"template_mse={best_score:.4f}")

    def recognize_components(self, ink_binary: np.ndarray) -> OCRResult:
        kernel = np.ones((5, 5), np.uint8)
        merged = cv2.dilate((ink_binary > 0).astype(np.uint8) * 255, kernel, iterations=1)
        n, _labels, stats, _ = cv2.connectedComponentsWithStats((merged > 0).astype(np.uint8), 8)
        boxes = []
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if area > 25 and w > 4 and h > 8:
                boxes.append((int(x), int(y), int(w), int(h)))
        boxes.sort(key=lambda b: b[0])
        if not boxes:
            return OCRResult("template-components", "", 0.0, "no components")
        chars: List[str] = []
        confs: List[float] = []
        prev_end = None
        for x, y, w, h in boxes[:32]:
            if prev_end is not None and x - prev_end > max(18, w * 0.8):
                chars.append(" ")
            pad = 10
            crop = ink_binary[max(0, y - pad) : min(ink_binary.shape[0], y + h + pad), max(0, x - pad) : min(ink_binary.shape[1], x + w + pad)]
            res = self.recognize_symbol(crop)
            chars.append(res.text)
            confs.append(res.confidence)
            prev_end = x + w
        return OCRResult("template-components", "".join(chars), float(np.mean(confs)) if confs else 0.0, f"components={len(boxes)}")


class AdvancedOCREngine:
    """
    Multi-backend OCR engine for the GestureOS whiteboard.

    Backends, in order:
    1. EasyOCR if installed. Good for handwritten-ish block text.
    2. Tesseract via pytesseract if installed and Tesseract executable is available.
    3. OpenAI Vision if OPENAI_API_KEY is set. Excellent for messy handwritten notes.
    4. OCR.space online API if OCR_SPACE_API_KEY is set.
    5. OpenCV template fallback for digits and uppercase symbols.
    """

    def __init__(self):
        self.template = TemplateSymbolRecognizer()
        self.easyocr_reader = None
        self.easyocr_error = ""
        self.tesseract_error = ""
        self._try_easyocr()
        self._try_tesseract()

    def _try_easyocr(self) -> None:
        try:
            import easyocr  # type: ignore

            # gpu=False avoids CUDA setup issues on most desktops.
            self.easyocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        except Exception as exc:  # optional dependency
            self.easyocr_reader = None
            self.easyocr_error = str(exc)

    def _try_tesseract(self) -> None:
        try:
            import pytesseract  # type: ignore

            _ = pytesseract.get_tesseract_version()
        except Exception as exc:
            self.tesseract_error = str(exc)

    def preprocess(self, ink_binary: np.ndarray, scale: int = 3) -> np.ndarray:
        """Prepare whiteboard ink for OCR."""
        img = (ink_binary > 0).astype(np.uint8) * 255
        if cv2.countNonZero(img) == 0:
            return img
        pts = cv2.findNonZero(img)
        x, y, w, h = cv2.boundingRect(pts)
        pad = max(30, int(max(w, h) * 0.18))
        crop = img[max(0, y - pad) : min(img.shape[0], y + h + pad), max(0, x - pad) : min(img.shape[1], x + w + pad)]
        # black text on white background for OCR engines
        crop = 255 - crop
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        crop = cv2.GaussianBlur(crop, (3, 3), 0)
        _, crop = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return crop

    def recognize_easyocr(self, ink_binary: np.ndarray) -> Optional[OCRResult]:
        if self.easyocr_reader is None:
            return None
        img = self.preprocess(ink_binary, scale=2)
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        try:
            results = self.easyocr_reader.readtext(rgb, detail=1, paragraph=False, allowlist="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
        except Exception as exc:
            return OCRResult("easyocr", "", 0.0, f"error={exc}")
        if not results:
            return OCRResult("easyocr", "", 0.0, "no text")
        parts = []
        confs = []
        for _box, text, conf in results:
            text = str(text).strip()
            if text:
                parts.append(text)
                confs.append(float(conf))
        return OCRResult("easyocr", " ".join(parts), float(np.mean(confs)) if confs else 0.0, f"items={len(results)}")

    def recognize_tesseract(self, ink_binary: np.ndarray) -> Optional[OCRResult]:
        try:
            import pytesseract  # type: ignore
        except Exception:
            return None
        img = self.preprocess(ink_binary, scale=3)
        configs = [
            "--oem 3 --psm 10 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
            "--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
            "--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz ",
            "--oem 3 --psm 6",
        ]
        best = OCRResult("tesseract", "", 0.0, "")
        for cfg in configs:
            try:
                text = pytesseract.image_to_string(img, config=cfg).strip()
                data = pytesseract.image_to_data(img, config=cfg, output_type=pytesseract.Output.DICT)
                confs = []
                for c in data.get("conf", []):
                    try:
                        v = float(c)
                        if v >= 0:
                            confs.append(v / 100.0)
                    except Exception:
                        pass
                conf = float(np.mean(confs)) if confs else (0.45 if text else 0.0)
                if text and conf >= best.confidence:
                    best = OCRResult("tesseract", text, conf, f"config={cfg}")
            except Exception as exc:
                best.details = f"error={exc}"
        return best


    def recognize_openai_vision(self, ink_binary: np.ndarray) -> Optional[OCRResult]:
        """Powerful optional handwritten-note recognition using OpenAI Vision."""
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            return None
        model = os.getenv("OPENAI_VISION_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
        img = self.preprocess(ink_binary, scale=2)
        ok, png = cv2.imencode(".png", img)
        if not ok:
            return OCRResult("openai-vision", "", 0.0, "png encode failed")
        b64 = base64.b64encode(png.tobytes()).decode("ascii")
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Read the handwritten content in this whiteboard image. Return only the recognized text. If it is a single digit/letter, return only that symbol. If uncertain, give the best guess without explanation.",
                        },
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}},
                    ],
                }
            ],
            "max_tokens": 80,
            "temperature": 0,
        }
        try:
            req = Request(
                "https://api.openai.com/v1/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
            )
            with urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8", errors="replace"))
            text = data["choices"][0]["message"]["content"].strip()
            return OCRResult("openai-vision", text, 0.92 if text else 0.0, f"model={model}")
        except Exception as exc:
            return OCRResult("openai-vision", "", 0.0, f"error={exc}")

    def recognize_ocr_space(self, ink_binary: np.ndarray) -> Optional[OCRResult]:
        api_key = os.getenv("OCR_SPACE_API_KEY", "").strip()
        if not api_key:
            return None
        img = self.preprocess(ink_binary, scale=2)
        ok, png = cv2.imencode(".png", img)
        if not ok:
            return OCRResult("ocr.space", "", 0.0, "png encode failed")
        b64 = base64.b64encode(png.tobytes()).decode("ascii")
        payload = urlencode({
            "apikey": api_key,
            "base64Image": "data:image/png;base64," + b64,
            "language": "eng",
            "OCREngine": "2",
            "scale": "true",
            "isOverlayRequired": "false",
        }).encode("utf-8")
        try:
            req = Request("https://api.ocr.space/parse/image", data=payload, headers={"Content-Type": "application/x-www-form-urlencoded"})
            with urlopen(req, timeout=20) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
            data = json.loads(raw)
            texts = []
            for item in data.get("ParsedResults", []) or []:
                parsed = str(item.get("ParsedText", "")).strip()
                if parsed:
                    texts.append(parsed)
            return OCRResult("ocr.space", " ".join(texts).strip(), 0.75 if texts else 0.0, "online")
        except Exception as exc:
            return OCRResult("ocr.space", "", 0.0, f"error={exc}")

    def recognize_all(self, ink_binary: np.ndarray) -> List[OCRResult]:
        results: List[OCRResult] = []
        if cv2.countNonZero(ink_binary) < 40:
            return [OCRResult("none", "Nothing drawn", 0.0, "empty canvas")]

        easy = self.recognize_easyocr(ink_binary)
        if easy is not None:
            results.append(easy)

        tess = self.recognize_tesseract(ink_binary)
        if tess is not None:
            results.append(tess)

        openai_result = self.recognize_openai_vision(ink_binary)
        if openai_result is not None:
            results.append(openai_result)

        online = self.recognize_ocr_space(ink_binary)
        if online is not None:
            results.append(online)

        results.append(self.template.recognize_symbol(ink_binary))
        results.append(self.template.recognize_components(ink_binary))
        return results

    def best(self, ink_binary: np.ndarray) -> Tuple[OCRResult, List[OCRResult]]:
        results = self.recognize_all(ink_binary)
        # Prefer real OCR text if it has non-empty output, then fall back to template.
        candidates = [r for r in results if r.text and r.text != "?" and r.text != "Nothing drawn"]
        if not candidates:
            return results[0], results
        backend_weight = {"openai-vision": 0.35, "easyocr": 0.18, "tesseract": 0.12, "ocr.space": 0.10, "template": 0.0, "template-components": -0.02}
        best = max(candidates, key=lambda r: r.confidence + backend_weight.get(r.backend, 0.0) + min(len(r.text), 8) * 0.005)
        return best, results
