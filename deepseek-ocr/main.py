#!/usr/bin/env python3
"""
pdf_to_markdown.py — Convert a PDF to Markdown using a vision model via Ollama.

Dependencies:
    pip install pypdfium2 Pillow

Usage:
    python pdf_to_markdown.py input.pdf
    python pdf_to_markdown.py input.pdf --gpu
    python pdf_to_markdown.py input.pdf -o out.md
    python pdf_to_markdown.py input.pdf --dpi 96
    python pdf_to_markdown.py input.pdf --debug
    python pdf_to_markdown.py --probe
"""

import argparse
import base64
import io
import json
import os
import sys
import urllib.request
import urllib.error
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image


# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────

OLLAMA_API = "http://localhost:11434/api"

# Better vision model than deepseek-ocr in Ollama
MODEL = "deepseek-ocr:latest"

# Minimal prompt — avoids context exhaustion
PROMPT = "Extract all visible text from this document page as clean markdown."


# ──────────────────────────────────────────────────────────────────────────────
# OLLAMA HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _post(endpoint: str, payload: dict) -> str:
    """POST request to Ollama."""
    url = f"{OLLAMA_API}/{endpoint}"

    data = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req) as resp:
            return resp.read().decode("utf-8")

    except urllib.error.URLError as exc:
        sys.exit(
            f"[ERROR] Cannot reach Ollama at {OLLAMA_API}: {exc}\n"
            f"Make sure `ollama serve` is running."
        )


def _parse_ndjson(raw: str) -> list[dict]:
    """
    Parse Ollama NDJSON or single JSON response.
    """
    results = []

    for line in raw.strip().splitlines():
        line = line.strip()

        if not line:
            continue

        try:
            results.append(json.loads(line))

        except json.JSONDecodeError as exc:
            print(
                f"[WARN] Failed to parse JSON line: {exc}\n"
                f"Line: {line[:120]}",
                file=sys.stderr,
            )

    return results


def ollama_chat(
    img_b64: str,
    prompt: str,
    use_gpu: bool,
    debug: bool = False,
) -> str:
    """
    Send image to Ollama vision model.
    """

    options = {
        "temperature": 0,
        "num_predict": 1024,
    }

    if use_gpu:
        options["num_gpu"] = 999

    payload = {
        "model": MODEL,
        "stream": False,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [img_b64],
            }
        ],
        "options": options,
    }

    raw = _post("chat", payload)

    if debug:
        print(
            f"\n[DEBUG] Raw Ollama response:\n{raw}\n",
            file=sys.stderr,
        )

    objs = _parse_ndjson(raw)

    if not objs:
        raise RuntimeError("Empty response from Ollama.")

    for obj in objs:
        if "error" in obj:
            raise RuntimeError(obj["error"])

    content = "".join(
        obj.get("message", {}).get("content", "")
        for obj in objs
    ).strip()

    if debug:
        print(
            f"[DEBUG] Parsed content length: {len(content)}",
            file=sys.stderr,
        )

        if objs:
            print(
                f"[DEBUG] eval_count = {objs[-1].get('eval_count')}",
                file=sys.stderr,
            )

    return content


# ──────────────────────────────────────────────────────────────────────────────
# PDF RENDERING
# ──────────────────────────────────────────────────────────────────────────────

def page_to_b64_image(
    pdf_path: str,
    page_index: int,
    dpi: int = 96,
) -> str:
    """
    Render PDF page to JPEG base64.
    """

    scale = dpi / 72

    doc = pdfium.PdfDocument(pdf_path)

    try:
        page = doc[page_index]

        bitmap = page.render(scale=scale)

        image: Image.Image = bitmap.to_pil()

        if image.mode != "RGB":
            image = image.convert("RGB")

        # Downscale giant pages
        max_width = 1800

        if image.width > max_width:
            ratio = max_width / image.width

            image = image.resize(
                (
                    int(image.width * ratio),
                    int(image.height * ratio),
                ),
                Image.LANCZOS,
            )

        buf = io.BytesIO()

        image.save(
            buf,
            format="JPEG",
            quality=85,
        )

        return base64.b64encode(buf.getvalue()).decode("utf-8")

    finally:
        doc.close()


# ──────────────────────────────────────────────────────────────────────────────
# PROBE
# ──────────────────────────────────────────────────────────────────────────────

def probe(debug: bool = False) -> None:
    """
    Test vision functionality with tiny generated image.
    """

    print("[PROBE] Testing vision model...", file=sys.stderr)

    img = Image.new("RGB", (220, 80), "white")

    buf = io.BytesIO()

    img.save(buf, format="JPEG")

    img_b64 = base64.b64encode(buf.getvalue()).decode()

    try:
        result = ollama_chat(
            img_b64,
            "Reply with exactly: OK",
            use_gpu=False,
            debug=debug,
        )

        print(f"[PROBE] Model response: {result!r}", file=sys.stderr)

        if result:
            print("[PROBE] SUCCESS", file=sys.stderr)
        else:
            print("[PROBE] FAILED — empty response", file=sys.stderr)

    except Exception as exc:
        print(f"[PROBE] FAILED: {exc}", file=sys.stderr)

    sys.exit(0)


# ──────────────────────────────────────────────────────────────────────────────
# CONVERSION
# ──────────────────────────────────────────────────────────────────────────────

def convert_pdf(
    pdf_path: str,
    use_gpu: bool,
    dpi: int = 96,
    debug: bool = False,
) -> str:

    pages_md = []

    doc = pdfium.PdfDocument(pdf_path)

    total_pages = len(doc)

    doc.close()

    print(
        f"[INFO] Processing {total_pages} page(s) "
        f"(dpi={dpi}, gpu={use_gpu})",
        file=sys.stderr,
    )

    for idx in range(total_pages):

        print(
            f"  Page {idx + 1}/{total_pages}...",
            end=" ",
            flush=True,
            file=sys.stderr,
        )

        try:
            img_b64 = page_to_b64_image(
                pdf_path,
                idx,
                dpi=dpi,
            )

        except Exception as exc:
            print(
                f"\n[ERROR] Failed rendering page: {exc}",
                file=sys.stderr,
            )

            pages_md.append(
                f"<!-- Page {idx + 1}: render failed -->"
            )

            continue

        try:
            markdown = ollama_chat(
                img_b64,
                PROMPT,
                use_gpu,
                debug=debug,
            )

        except Exception as exc:
            print(
                f"\n[ERROR] Model failed: {exc}",
                file=sys.stderr,
            )

            pages_md.append(
                f"<!-- Page {idx + 1}: model failed -->"
            )

            continue

        if not markdown:
            print(
                "\n[WARN] Empty model response",
                file=sys.stderr,
            )

            pages_md.append(
                f"<!-- Page {idx + 1}: empty response -->"
            )

        else:
            pages_md.append(markdown)

            print("done", file=sys.stderr)

    return "\n\n---\n\n".join(pages_md)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description="Convert PDF to Markdown using Ollama vision model."
    )

    parser.add_argument(
        "pdf",
        nargs="?",
        help="Input PDF file",
    )

    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Enable GPU",
    )

    parser.add_argument(
        "-o",
        "--output",
        metavar="FILE",
        help="Write output to file",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=96,
        help="Rasterization DPI (default: 96)",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show raw Ollama responses",
    )

    parser.add_argument(
        "--probe",
        action="store_true",
        help="Test vision connectivity",
    )

    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():

    args = parse_args()

    if args.probe:
        probe(debug=args.debug)

    if not args.pdf:
        sys.exit(
            "[ERROR] Provide a PDF file or use --probe."
        )

    if not os.path.isfile(args.pdf):
        sys.exit(
            f"[ERROR] File not found: {args.pdf}"
        )

    print(
        f"[INFO] Using model: {MODEL}",
        file=sys.stderr,
    )

    markdown = convert_pdf(
        args.pdf,
        use_gpu=args.gpu,
        dpi=args.dpi,
        debug=args.debug,
    )

    if args.output:

        out_path = Path(args.output)

        out_path.write_text(
            markdown,
            encoding="utf-8",
        )

        print(
            f"[INFO] Wrote output to: {out_path}",
            file=sys.stderr,
        )

    else:
        print(markdown)


if __name__ == "__main__":
    main()