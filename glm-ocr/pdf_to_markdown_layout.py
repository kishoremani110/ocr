#!/usr/bin/env python3
"""
PDF to Markdown converter using PP-DocLayoutV3 layout analysis + GLM-OCR via Ollama.

Pipeline:
    PDF -> page images -> PP-DocLayoutV3 (detect regions) -> crop regions -> GLM-OCR (Ollama) -> markdown

Environment variables (loaded from .env):
    GLM_OCR_HOST  - Ollama server URL (default: http://localhost:11434)
    GLM_OCR_MODEL - Model name (default: glm-ocr)

Usage:
    python pdf_to_markdown_layout.py input.pdf
    python pdf_to_markdown_layout.py input.pdf --output output.md
    python pdf_to_markdown_layout.py input.pdf --dpi 200

Requirements:
    pip install ollama pdfplumber python-dotenv torch transformers
"""

import argparse
import base64
import io
import os
import ssl
import sys
import threading
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# SSL workaround for corporate proxies (needed for HuggingFace model download)
ssl._create_default_https_context = ssl._create_unverified_context
os.environ["HF_HUB_DISABLE_SSL_VERIFY"] = "1"
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""
os.environ["HTTPX_SSL_VERIFY"] = "0"

# Monkey-patch httpx to disable SSL verification globally
import httpx
_original_httpx_client_init = httpx.Client.__init__

def _patched_httpx_client_init(self, *args, **kwargs):
    kwargs["verify"] = False
    _original_httpx_client_init(self, *args, **kwargs)

httpx.Client.__init__ = _patched_httpx_client_init

try:
    import pdfplumber
except ImportError:
    print("Error: pip install pdfplumber"); sys.exit(1)

try:
    import ollama
except ImportError:
    print("Error: pip install ollama"); sys.exit(1)

try:
    import torch
    from transformers import PPDocLayoutV3ForObjectDetection, PPDocLayoutV3ImageProcessor
except ImportError:
    print("Error: pip install torch transformers"); sys.exit(1)

from PIL import Image


# ── Configuration ─────────────────────────────────────────────────────────────

DEFAULT_MODEL = os.environ.get("GLM_OCR_MODEL", "glm-ocr")
DEFAULT_DPI   = 150
DEFAULT_HOST  = os.environ.get("GLM_OCR_HOST", "http://localhost:11434")
if not DEFAULT_HOST.startswith("http"):
    DEFAULT_HOST = f"http://{DEFAULT_HOST}:{os.environ.get('GLM_OCR_PORT', '11434')}"

LAYOUT_MODEL_ID = "PaddlePaddle/PP-DocLayoutV3_safetensors"
LAYOUT_THRESHOLD = 0.5

OCR_PROMPT = (
    "You are an OCR assistant. Extract all text from this document page and "
    "return it as clean, well-structured Markdown.\n\n"
    "CRITICAL TABLE RULES:\n"
    "- Count the exact number of columns from the table header.\n"
    "- Every row MUST have the same number of pipe (|) separators as the header.\n"
    "- If a cell is empty/blank, write it as '| |' (pipe space pipe).\n"
    "- NEVER skip or collapse empty cells.\n"
    "- Example: '| Data | | Value | |' has 4 columns, 2 of which are empty.\n\n"
    "- If a cell appears misaligned or spans awkardly, still extract its content into "
    "  the correct column position.\n"
    "- Extract all visible text in every cell, even if the cell borders look broken. \n\n"
    "SUBSCRIPT AND SUPERSCRIPT RULES:\n"
    "- Use HTML tags for superscripts: x<sup>2</sup>, 10<sup>3</sup>\n"
    "- Use HTML tags for subscripts: H<sub>2</sub>O, CO<sub>2</sub>\n"
    "- NEVER use LaTeX notation like ^{} or _{}\n\n"
    "Preserve headings, bullet lists, numbered lists, and emphasis where visible. "
    "Do not add commentary — output only the Markdown content."
)


# ── Elapsed-time ticker ────────────────────────────────────────────────────────

class Ticker:
    FRAMES = ["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"]

    def __init__(self, label: str):
        self.label   = label
        self._stop   = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        t0  = time.time()
        idx = 0
        while not self._stop.is_set():
            elapsed = int(time.time() - t0)
            frame   = self.FRAMES[idx % len(self.FRAMES)]
            line    = f"\r  {frame}  {self.label} {elapsed}s"
            sys.stdout.write(line)
            sys.stdout.flush()
            time.sleep(0.1)
            idx += 1

    def start(self) -> "Ticker":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self._thread.join()
        sys.stdout.write("\r" + " " * 60 + "\r")
        sys.stdout.flush()


# ── Layout Detection ──────────────────────────────────────────────────────────

class LayoutDetector:
    def __init__(self, model_id: str = LAYOUT_MODEL_ID, device: str = "cpu"):
        self.device = device
        print(f"  Loading layout model: {model_id} (device: {device})")
        self.processor = PPDocLayoutV3ImageProcessor.from_pretrained(model_id)
        self.model = PPDocLayoutV3ForObjectDetection.from_pretrained(model_id)
        self.model.eval()
        self.model.to(device)
        self.id2label = self.model.config.id2label

    def detect(self, image: Image.Image, threshold: float = LAYOUT_THRESHOLD) -> list[dict]:
        img_rgb = image.convert("RGB") if image.mode != "RGB" else image
        inputs = self.processor(images=[img_rgb], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = torch.tensor([img_rgb.size[::-1]], device=self.device)
        results = self.processor.post_process_object_detection(
            outputs, threshold=threshold, target_sizes=target_sizes
        )[0]

        regions = []
        for score, label_id, box in zip(
            results["scores"].tolist(),
            results["labels"].tolist(),
            results["boxes"].tolist(),
        ):
            x_min, y_min, x_max, y_max = [int(v) for v in box]
            regions.append({
                "label": self.id2label.get(label_id, f"class_{label_id}"),
                "score": score,
                "bbox": (x_min, y_min, x_max, y_max),
            })

        # Sort by reading order: top-to-bottom, left-to-right
        regions.sort(key=lambda r: (r["bbox"][1], r["bbox"][0]))
        return regions


# ── PDF rasterisation ─────────────────────────────────────────────────────────

def rasterise_pdf(pdf_path: str, dpi: int) -> list[Image.Image]:
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            img = page.to_image(resolution=dpi).original
            print(f"  Page {i}: {img.size[0]}×{img.size[1]}px")
            pages.append(img)
    return pages


# ── OCR via Ollama with streaming ─────────────────────────────────────────────

def image_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def ocr_region(img: Image.Image, host: str, model: str, label: str) -> str:
    b64    = image_to_base64(img)
    client = ollama.Client(host=host, timeout=400)

    ticker = Ticker(f"OCR [{label}] ...").start()
    first  = True
    parts  = []

    try:
        stream = client.chat(
            model=model,
            messages=[{"role": "user", "content": OCR_PROMPT, "images": [b64]}],
            stream=True,
        )
        for chunk in stream:
            token = chunk["message"]["content"]
            if first:
                ticker.stop()
                first = False
            sys.stdout.write(token)
            sys.stdout.flush()
            parts.append(token)

    except Exception as exc:
        ticker.stop()
        print(f"\n  [!] OCR error for [{label}]: {exc}")
        return f"<!-- OCR failed for region [{label}]: {exc} -->"

    if first:
        ticker.stop()
    print()
    return "".join(parts).strip()


# ── Orchestration ─────────────────────────────────────────────────────────────

def convert(pdf_path: str, output_path: str, host: str, model: str, dpi: int) -> None:
    pdf_path = os.path.abspath(pdf_path)
    if not os.path.isfile(pdf_path):
        print(f"Error: File not found — {pdf_path}"); sys.exit(1)

    print(f"PDF     : {pdf_path}")
    print(f"Output  : {output_path}")
    print(f"Host    : {host}")
    print(f"Model   : {model}")
    print(f"DPI     : {dpi}\n")

    # Rasterise PDF
    ticker = Ticker("Rasterising PDF ...").start()
    pages = rasterise_pdf(pdf_path, dpi)
    ticker.stop()
    total = len(pages)
    print(f"  Rasterised {total} page(s)\n")

    # Load layout detector
    ticker = Ticker("Loading layout model ...").start()
    detector = LayoutDetector(device="cpu")
    ticker.stop()
    print("  Layout model loaded\n")

    pages_md = []
    for page_idx, page_img in enumerate(pages, start=1):
        print(f"── Page {page_idx}/{total}")

        # Detect layout regions
        regions = detector.detect(page_img)
        print(f"  Detected {len(regions)} region(s): {[r['label'] for r in regions]}")

        if not regions:
            # Fallback: send the full page
            print("  No regions detected, sending full page to OCR")
            md = ocr_region(page_img, host, model, "full_page")
            pages_md.append(md)
            continue

        page_parts = []
        for i, region in enumerate(regions, 1):
            label = region["label"]
            bbox = region["bbox"]
            score = region["score"]
            print(f"  Region {i}/{len(regions)}: [{label}] score={score:.2f} bbox={bbox}")

            # Crop region from page image
            cropped = page_img.crop(bbox)

            # Send cropped region to GLM-OCR
            md = ocr_region(cropped, host, model, label)
            page_parts.append(md)

        pages_md.append("\n\n".join(page_parts))

    header    = f"<!-- source: {os.path.basename(pdf_path)} -->\n\n"
    separator = "\n\n---\n\n"
    full_md   = header + separator.join(pages_md)

    output_path = os.path.abspath(output_path)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(full_md, encoding="utf-8")
    print(f"\nDone! Markdown saved to: {output_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Convert PDF to Markdown using PP-DocLayoutV3 + GLM-OCR via Ollama.",
    )
    p.add_argument("pdf")
    p.add_argument("-o", "--output", default=None)
    p.add_argument("--host",  default=DEFAULT_HOST)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--dpi",   type=int, default=DEFAULT_DPI)
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.output is None:
        pdf         = Path(args.pdf)
        args.output = str(pdf.parent / (pdf.stem + "_layout.md"))
    convert(args.pdf, args.output, args.host, args.model, args.dpi)


if __name__ == "__main__":
    main()
