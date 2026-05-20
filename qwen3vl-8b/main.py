#!/usr/bin/env python3
"""
PDF → Markdown extractor using:
  - pdfplumber   : renders each PDF page to an image
  - qwen3-vl:8b : reads the image and outputs Markdown

Dependencies:
    pip install pdfplumber ollama pillow

Usage:
    python pdf_to_markdown.py document.pdf                  # CPU (default)
    python pdf_to_markdown.py document.pdf --gpu            # AMD iGPU via Vulkan
    python pdf_to_markdown.py document.pdf -o output.md
    python pdf_to_markdown.py document.pdf --pages 1,3,5-8
    python pdf_to_markdown.py document.pdf --dpi 200
    python pdf_to_markdown.py document.pdf --host http://localhost:11434
    python pdf_to_markdown.py document.pdf --no-think       # Disables thinking tokens (e.g., --think=false)

GPU notes (AMD iGPU on Windows):
    --gpu sets OLLAMA_VULKAN=1 in the environment before starting the Ollama
    server subprocess.  ROCm does not support Windows APUs/iGPUs; Vulkan is
    the correct acceleration path for Radeon 860M / 890M and similar.

    Prerequisites for --gpu:
      1. AMD Adrenalin drivers installed (supplies Vulkan runtime)
      2. Ollama NOT already running — this script will start it
         If Ollama is already running without OLLAMA_VULKAN=1, stop it first:
           ollama stop   (or kill the tray icon process)
"""

import argparse
import base64
import io
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

try:
    import pdfplumber
except ImportError:
    print("Error: pip install pdfplumber"); sys.exit(1)

try:
    import ollama
except ImportError:
    print("Error: pip install ollama"); sys.exit(1)


# ── Configuration ─────────────────────────────────────────────────────────────

DEFAULT_MODEL = "qwen3-vl:8b"
DEFAULT_DPI   = 150
DEFAULT_HOST  = "http://localhost:11434"

PROMPT = (
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
    """
    Prints an updating elapsed-time line in place:
        ⠹  waiting for first token ... 4s
    Stops (and clears the line) as soon as .stop() is called.
    """
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
            sys.stdout.write(f"\r  {frame}  {self.label} {elapsed}s")
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


# ── GPU / Ollama server management ────────────────────────────────────────────

def _ollama_is_running(host: str) -> bool:
    """Return True if an Ollama server is already answering on host."""
    try:
        import urllib.request
        urllib.request.urlopen(f"{host}/api/tags", timeout=2)
        return True
    except Exception:
        return False


def ensure_ollama_server(host: str, use_gpu: bool) -> subprocess.Popen | None:
    """
    Start Ollama with Vulkan enabled when --gpu is requested.

    Returns the Popen handle if we started the server (caller should keep it
    alive for the duration of the run), or None if the server was already up.

    If --gpu is requested but the server is already running WITHOUT Vulkan,
    we warn the user — we cannot inject env vars into an existing process.
    """
    if not use_gpu:
        return None                         # let Ollama manage itself normally

    if _ollama_is_running(host):
        print(
            "[GPU] WARNING: Ollama is already running.\n"
            "      OLLAMA_VULKAN=1 cannot be injected into an existing process.\n"
            "      Stop the Ollama tray icon / service and re-run with --gpu\n"
            "      to get Vulkan acceleration.\n"
        )
        return None

    env = os.environ.copy()
    env["OLLAMA_VULKAN"] = "1"

    print("[GPU] Starting Ollama with OLLAMA_VULKAN=1 ...")
    proc = subprocess.Popen(
        ["ollama", "serve"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Wait up to 10 s for the server to become ready
    ticker = Ticker("waiting for Ollama to start ...").start()
    for _ in range(100):
        time.sleep(0.1)
        if _ollama_is_running(host):
            ticker.stop()
            print("[GPU] Ollama started with Vulkan — Radeon 860M will be used.\n")
            return proc
    ticker.stop()
    print("[GPU] ERROR: Ollama did not start in time. Check that 'ollama' is on PATH.")
    proc.terminate()
    sys.exit(1)


# ── PDF rasterisation ─────────────────────────────────────────────────────────

def page_to_base64(page, resolution: int) -> str:
    """Render a pdfplumber page to a PNG and return it as a base64 string."""
    img = page.to_image(resolution=resolution).original

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    kb = len(buf.getvalue()) / 1024
    print(f"    {img.size[0]}×{img.size[1]}px  {kb:.0f} KB")
    return base64.b64encode(buf.read()).decode()


# ── OCR via Ollama with streaming ─────────────────────────────────────────────

def extract_page_markdown(b64_image: str, host: str, model: str, page_num: int, no_think: bool) -> str:
    """Stream Markdown tokens from the model, printing them as they arrive."""
    client = ollama.Client(host=host)

    ticker = Ticker("waiting for first token ...").start()
    first  = True
    parts  = []

    # Prepare model generation options
    options = {}
    if no_think:
        options["think"] = False

    try:
        stream = client.generate(
            model=model,
            prompt=PROMPT,
            images=[b64_image],
            stream=True,
            options=options,
        )
        for chunk in stream:
            token = chunk["response"]
            if first:
                ticker.stop()       # clear spinner the moment tokens start arriving
                first = False
            sys.stdout.write(token)
            sys.stdout.flush()
            parts.append(token)

    except ollama.ResponseError as exc:
        ticker.stop()
        print(f"\n  [!] Ollama error on page {page_num}: {exc}")
        return f"<!-- OCR failed for page {page_num}: {exc} -->"
    except Exception as exc:
        ticker.stop()
        print(f"\n  [!] Unexpected error on page {page_num}: {exc}")
        return f"<!-- OCR failed for page {page_num}: {exc} -->"

    print("\n")
    return "".join(parts).strip()


# ── Page-spec parser ──────────────────────────────────────────────────────────

def parse_pages(spec: str, total: int) -> list[int]:
    """Parse '1,3,5-8' into a sorted list of 0-based indices."""
    pages = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            pages.update(range(int(a), int(b) + 1))
        else:
            pages.add(int(part))
    return sorted(p - 1 for p in pages if 1 <= p <= total)


# ── Orchestration ─────────────────────────────────────────────────────────────

def extract(pdf_path: str, output_path: str, host: str, model: str,
            dpi: int, page_indices: list[int] | None, no_think: bool) -> None:
    pdf_path = os.path.abspath(pdf_path)
    if not os.path.isfile(pdf_path):
        print(f"Error: File not found — {pdf_path}"); sys.exit(1)

    print(f"PDF     : {pdf_path}")
    print(f"Output  : {output_path}")
    print(f"Host    : {host}")
    print(f"Model   : {model}")
    print(f"DPI     : {dpi}")
    print(f"Think   : {'False' if no_think else 'True'}\n")

    with pdfplumber.open(pdf_path) as pdf:
        total   = len(pdf.pages)
        targets = page_indices if page_indices is not None else list(range(total))
        print(f"[INFO] {Path(pdf_path).name}  —  {total} pages total, processing {len(targets)}\n")

        sections = []
        for idx in targets:
            page_num = idx + 1
            print(f"── Page {page_num}/{total}")
            b64 = page_to_base64(pdf.pages[idx], resolution=dpi)
            md  = extract_page_markdown(b64, host, model, page_num, no_think)
            sections.append(f"<!-- page {page_num} -->\n{md}")

    full_md = f"<!-- source: {os.path.basename(pdf_path)} -->\n\n"
    full_md += "\n\n---\n\n".join(sections)

    output_path = os.path.abspath(output_path)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(full_md, encoding="utf-8")
    print(f"Done! Markdown saved to: {output_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract a PDF to Markdown using pdfplumber + qwen3-vl:8b.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "GPU acceleration (AMD iGPU on Windows):\n"
            "  --gpu starts Ollama with OLLAMA_VULKAN=1.\n"
            "  ROCm does not support Windows APUs; Vulkan is the correct path\n"
            "  for Radeon 860M / 890M.  Requires AMD Adrenalin drivers and\n"
            "  that Ollama is NOT already running when the script starts.\n"
        ),
    )
    parser.add_argument("pdf",   help="Input PDF file")
    parser.add_argument("-o", "--output", default=None,         help="Output .md file (default: <pdf>.md)")
    parser.add_argument("--host",  default=DEFAULT_HOST,        help="Ollama host URL")
    parser.add_argument("--model", default=DEFAULT_MODEL,       help="Ollama model name")
    parser.add_argument("--dpi",   type=int, default=DEFAULT_DPI, help="Render resolution (default: 150)")
    parser.add_argument("--pages", type=str, default=None,       help="Pages to process, e.g. '1,3,5-8'")
    parser.add_argument(
        "--gpu",
        action="store_true",
        default=False,
        help="Enable AMD iGPU acceleration via Vulkan (starts Ollama with OLLAMA_VULKAN=1)",
    )
    parser.add_argument(
        "--no-think",
        action="store_true",
        default=False,
        help="Disable reasoning/thinking tokens during output extraction (adds think=false option)",
    )
    args = parser.parse_args()

    if args.output is None:
        pdf        = Path(args.pdf)
        args.output = str(pdf.parent / (pdf.stem + ".md"))

    # Start Ollama with Vulkan if requested (no-op if --gpu not passed)
    ollama_proc = ensure_ollama_server(args.host, args.gpu)

    try:
        with pdfplumber.open(args.pdf) as pdf:
            total = len(pdf.pages)

        page_indices = parse_pages(args.pages, total) if args.pages else None
        extract(args.pdf, args.output, args.host, args.model, args.dpi, page_indices, args.no_think)
    finally:
        # Shut down the server we started (don't leave a dangling process)
        if ollama_proc is not None:
            ollama_proc.terminate()
            print("[GPU] Ollama server stopped.")


if __name__ == "__main__":
    main()