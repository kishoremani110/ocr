#!/usr/bin/env python3
"""
pdf_to_markdown.py
------------------
Extracts a PDF document into Markdown format using MinerU.

On first run with a VLM-based backend (--gpu), MinerU downloads ~5-10 GB of
model weights from HuggingFace. This script starts mineru-api explicitly so
that download progress is ALWAYS visible in the terminal — no more silent
hangs or misleading timeout warnings.

Usage:
    # CPU mode (default — pipeline backend, instant start)
    python pdf_to_markdown.py input.pdf

    # GPU mode — hybrid backend (best quality, slow first-run model download)
    python pdf_to_markdown.py input.pdf --gpu

    # GPU mode — pipeline backend (fast start, good quality, ~6 GB VRAM)
    python pdf_to_markdown.py input.pdf --gpu --backend pipeline

    # Pre-download all models once (recommended before first --gpu run)
    python pdf_to_markdown.py --download-models

    # Custom output directory, language, page range
    python pdf_to_markdown.py input.pdf -o ./out --lang en --start 0 --end 9

Backend reference:
    pipeline           Pure ONNX models. CPU or GPU. Instant start. ~6 GB VRAM (GPU).
    hybrid-auto-engine VLM + pipeline hybrid. GPU only. Best accuracy. ~8 GB VRAM.
    vlm-auto-engine    Full VLM. GPU only. Highest accuracy. ~10 GB VRAM.

Dependencies:
    pip install "mineru[all]" pdfplumber
"""

import argparse
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

try:
    import pdfplumber
except ImportError:
    print("[ERROR] 'pdfplumber' is not installed. Run: pip install pdfplumber")
    sys.exit(1)


# ─────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────

VLM_BACKENDS = {"hybrid-auto-engine", "vlm-auto-engine"}

BACKEND_INFO = {
    "pipeline":           "ONNX models  — instant start  — CPU or GPU (~6 GB VRAM)",
    "hybrid-auto-engine": "VLM+pipeline — best accuracy  — GPU only   (~8 GB VRAM)",
    "vlm-auto-engine":    "Full VLM     — top accuracy   — GPU only   (~10 GB VRAM)",
}

# Max seconds to wait for mineru-api to accept TCP connections.
# On first run the VLM download alone can take 20+ minutes.
API_STARTUP_TIMEOUT = 1800  # 30 minutes


# ─────────────────────────────────────────────
#  Small helpers
# ─────────────────────────────────────────────

def check_tool(name: str) -> None:
    if not shutil.which(name):
        print(f"[ERROR] '{name}' not found. Install with:  pip install 'mineru[all]'")
        sys.exit(1)


def find_free_port() -> int:
    """Let the OS assign a free port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_for_port(host: str, port: int, timeout: int) -> bool:
    """Return True once the port accepts TCP connections, False on timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except OSError:
            time.sleep(2)
    return False


# ─────────────────────────────────────────────
#  PDF validation / pre-processing (pdfplumber)
# ─────────────────────────────────────────────

def validate_pdf(pdf_path: Path) -> dict:
    """
    Open the PDF with pdfplumber and return basic metadata.
    Exits if the file cannot be opened or has no pages.
    """
    print(f"[INFO] Validating PDF: {pdf_path}")
    try:
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            if total_pages == 0:
                print("[ERROR] The PDF has no pages.")
                sys.exit(1)

            first_page = pdf.pages[0]
            width  = round(first_page.width,  2)
            height = round(first_page.height, 2)

            # Sample first 3 pages to detect scanned vs text-based PDF
            sample_text = ""
            for page in pdf.pages[:min(3, total_pages)]:
                sample_text += page.extract_text() or ""
            has_text = bool(sample_text.strip())

        meta = {
            "total_pages": total_pages,
            "width":       width,
            "height":      height,
            "has_text":    has_text,
        }
        text_status = "yes" if has_text else "no (likely scanned — OCR will be used)"
        print(
            f"[INFO] PDF validated — pages: {total_pages}, "
            f"size: {width}x{height} pt, "
            f"extractable text: {text_status}"
        )
        return meta

    except Exception as exc:
        print(f"[ERROR] Failed to open PDF with pdfplumber: {exc}")
        sys.exit(1)


def clamp_page_range(start, end, total: int) -> tuple:
    s = max(0, start if start is not None else 0)
    e = min(total - 1, end if end is not None else total - 1)
    if s > e:
        print(f"[ERROR] --start ({s}) > --end ({e}) for a {total}-page document.")
        sys.exit(1)
    return s, e


# ─────────────────────────────────────────────
#  Backend resolution
# ─────────────────────────────────────────────

def resolve_backend(use_gpu: bool, backend_choice: str) -> str:
    if not use_gpu:
        if backend_choice not in ("auto", "pipeline"):
            print(
                f"[WARNING] --backend '{backend_choice}' requires --gpu. "
                "Falling back to 'pipeline' (CPU)."
            )
        return "pipeline"
    return "hybrid-auto-engine" if backend_choice == "auto" else backend_choice


# ─────────────────────────────────────────────
#  mineru-api lifecycle
# ─────────────────────────────────────────────

def start_api_server(port: int, backend: str) -> subprocess.Popen:
    """
    Launch mineru-api so its stdout/stderr flows directly to the terminal.

    This is the key fix: by NOT redirecting stdout/stderr, HuggingFace
    download progress bars and model-load logs are always visible.
    The user is never left staring at a silent, apparently frozen prompt.
    """
    check_tool("mineru-api")

    cmd = ["mineru-api", "--host", "127.0.0.1", "--port", str(port)]

    if backend in VLM_BACKENDS:
        # Preload the VLM during API startup so it is ready before the parse
        # task is submitted — avoids silent cold-start delays later.
        cmd += ["--enable-vlm-preload", "true"]
        print(
            "[INFO] VLM backend selected.\n"
            "[INFO] mineru-api will download and load the model now.\n"
            "[INFO] Model weights: ~5-10 GB (downloaded once, then cached).\n"
            "[INFO] You will see HuggingFace download progress below.\n"
        )
    else:
        print("[INFO] Pipeline backend — model weights are small, loading quickly.\n")

    print(f"[INFO] Starting mineru-api on port {port} ...")
    print(f"[INFO] Command: {' '.join(cmd)}")
    print("─" * 62)

    # stdout/stderr inherited → visible in the terminal in real time
    proc = subprocess.Popen(cmd)
    return proc


def stop_api_server(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        print("\n[INFO] Shutting down mineru-api ...")
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


# ─────────────────────────────────────────────
#  MinerU parse invocation
# ─────────────────────────────────────────────

def run_mineru(
    pdf_path:   Path,
    output_dir: Path,
    backend:    str,
    api_url:    str,
    lang,
    start:      int,
    end:        int,
    method:     str,
    formula:    bool,
    table:      bool,
) -> None:
    """Invoke the mineru CLI pointed at the already-running api server."""
    check_tool("mineru")

    cmd = [
        "mineru",
        "--api-url", api_url,
        "-p", str(pdf_path),
        "-o", str(output_dir),
        "-b", backend,
        "-m", method,
        "-s", str(start),
        "-e", str(end),
        "-f", str(formula).lower(),
        "-t", str(table).lower(),
    ]
    if lang:
        cmd += ["-l", lang]

    print(f"[INFO] Submitting parse task ...")
    print(f"[INFO] Command: {' '.join(cmd)}")
    print("─" * 62)

    result = subprocess.run(cmd, text=True)

    print("─" * 62)
    if result.returncode != 0:
        print(f"[ERROR] MinerU exited with code {result.returncode}.")
        sys.exit(result.returncode)

    print("[INFO] MinerU finished successfully.")


# ─────────────────────────────────────────────
#  Model pre-download helper
# ─────────────────────────────────────────────

def download_models() -> None:
    """
    Run mineru-models-download to pre-fetch all model weights.
    Run this once before the first --gpu parse to avoid cold-start delays.
    """
    check_tool("mineru-models-download")
    print("[INFO] Pre-downloading MinerU model weights (~5-10 GB) ...")
    print("[INFO] This is a one-time operation; models are cached afterwards.\n")
    result = subprocess.run(["mineru-models-download"])
    if result.returncode != 0:
        print(f"[ERROR] Model download failed (exit {result.returncode}).")
        sys.exit(result.returncode)
    print("\n[INFO] Models downloaded and cached. You can now run without --download-models.")


# ─────────────────────────────────────────────
#  Post-processing: locate Markdown output
# ─────────────────────────────────────────────

def find_markdown_files(output_dir: Path) -> list:
    return sorted(output_dir.rglob("*.md"))


# ─────────────────────────────────────────────
#  CLI argument parsing
# ─────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a PDF to Markdown using MinerU.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "pdf",
        metavar="PDF_FILE",
        nargs="?",
        help="Path to the input PDF file.",
    )
    parser.add_argument(
        "--download-models",
        action="store_true",
        default=False,
        help=(
            "Pre-download all MinerU model weights and exit. "
            "Run this once before your first --gpu parse."
        ),
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        default=False,
        help=(
            "Use GPU acceleration. Default backend: 'hybrid-auto-engine'. "
            "Override with --backend pipeline for a faster start."
        ),
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "pipeline", "hybrid-auto-engine", "vlm-auto-engine"],
        default="auto",
        metavar="BACKEND",
        help=(
            "MinerU parsing backend. 'auto' picks 'pipeline' (CPU) or "
            "'hybrid-auto-engine' (GPU). "
            "Choices: auto | pipeline | hybrid-auto-engine | vlm-auto-engine"
        ),
    )
    parser.add_argument(
        "-o", "--output",
        metavar="DIR",
        default=None,
        help=(
            "Output directory. "
            "Defaults to '<pdf_name>_output/' beside the input file."
        ),
    )
    parser.add_argument(
        "-m", "--method",
        choices=["auto", "txt", "ocr"],
        default="auto",
        help="Parsing method: auto (default), txt, or ocr.",
    )
    parser.add_argument(
        "-l", "--lang",
        default=None,
        metavar="LANG",
        help="Language code to improve OCR (e.g. en, ch, japan, korean).",
    )
    parser.add_argument(
        "-s", "--start",
        type=int, default=None, metavar="PAGE",
        help="0-based starting page (default: 0).",
    )
    parser.add_argument(
        "-e", "--end",
        type=int, default=None, metavar="PAGE",
        help="0-based ending page (default: last page).",
    )
    parser.add_argument(
        "--no-formula",
        action="store_true", default=False,
        help="Disable LaTeX formula extraction.",
    )
    parser.add_argument(
        "--no-table",
        action="store_true", default=False,
        help="Disable HTML table extraction.",
    )
    return parser


# ─────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────

def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    # ── Model pre-download shortcut ────────────────────────────────────────
    if args.download_models:
        download_models()
        return

    # ── Require a PDF when not downloading models ──────────────────────────
    if not args.pdf:
        parser.error("PDF_FILE is required (or use --download-models).")

    # ── Resolve paths ──────────────────────────────────────────────────────
    pdf_path = Path(args.pdf).resolve()
    if not pdf_path.exists():
        print(f"[ERROR] File not found: {pdf_path}")
        sys.exit(1)
    if not pdf_path.is_file():
        print(f"[ERROR] Not a file: {pdf_path}")
        sys.exit(1)
    if pdf_path.suffix.lower() != ".pdf":
        print(f"[WARNING] File does not have a .pdf extension: {pdf_path}")

    output_dir = (
        Path(args.output).resolve()
        if args.output
        else pdf_path.parent / f"{pdf_path.stem}_output"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Resolve backend ────────────────────────────────────────────────────
    backend     = resolve_backend(args.gpu, args.backend)
    mode_label  = f"GPU ({backend})" if args.gpu else "CPU (pipeline)"

    print("=" * 62)
    print("  PDF -> Markdown via MinerU")
    print("=" * 62)
    print(f"  Input   : {pdf_path}")
    print(f"  Output  : {output_dir}")
    print(f"  Mode    : {mode_label}")
    print(f"  Backend : {BACKEND_INFO.get(backend, backend)}")
    print("=" * 62 + "\n")

    # ── Validate PDF with pdfplumber ───────────────────────────────────────
    meta = validate_pdf(pdf_path)

    # ── Resolve page range ─────────────────────────────────────────────────
    start, end = clamp_page_range(args.start, args.end, meta["total_pages"])
    pages_to_process = end - start + 1
    print(
        f"[INFO] Processing pages {start}-{end} "
        f"({pages_to_process} of {meta['total_pages']})\n"
    )

    # ── Start mineru-api with visible output ───────────────────────────────
    port    = find_free_port()
    api_url = f"http://127.0.0.1:{port}"
    api_proc = start_api_server(port, backend)

    try:
        # Wait until the API is accepting connections
        print(f"\n[INFO] Waiting for mineru-api to be ready (port {port}) ...")
        if not wait_for_port("127.0.0.1", port, API_STARTUP_TIMEOUT):
            print(
                f"[ERROR] mineru-api did not become ready within "
                f"{API_STARTUP_TIMEOUT}s."
            )
            stop_api_server(api_proc)
            sys.exit(1)
        print(f"[INFO] mineru-api is ready at {api_url}\n")

        # ── Run the parse task ─────────────────────────────────────────────
        run_mineru(
            pdf_path   = pdf_path,
            output_dir = output_dir,
            backend    = backend,
            api_url    = api_url,
            lang       = args.lang,
            start      = start,
            end        = end,
            method     = args.method,
            formula    = not args.no_formula,
            table      = not args.no_table,
        )

    finally:
        stop_api_server(api_proc)

    # ── Report generated files ─────────────────────────────────────────────
    md_files = find_markdown_files(output_dir)
    if md_files:
        print(f"\n[INFO] Markdown file(s) generated ({len(md_files)}):")
        for md in md_files:
            size_kb = md.stat().st_size / 1024
            print(f"  +  {md}  ({size_kb:.1f} KB)")
    else:
        print(
            f"\n[WARNING] No .md files found. "
            f"Check output folder manually: {output_dir}"
        )

    print("\n[DONE] Extraction complete.")


if __name__ == "__main__":
    main()