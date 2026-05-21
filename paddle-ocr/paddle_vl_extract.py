"""
PDF to Markdown converter using PaddleOCR-VL (Vision-Language Model).

Uses PaddleOCR-VL-1.5 (0.9B parameter VLM) which reads documents visually,
bypassing traditional table cell detection. Better for complex/dense tables.

Dependencies:
  pip install paddlepaddle paddleocr[doc-parser]

For GPU support:
  pip install paddlepaddle-gpu paddleocr[doc-parser]
"""

import argparse
import base64
import os
import sys
from pathlib import Path

os.environ.setdefault("PADDLE_PDX_PDF_RENDER_SCALE", "6.0")
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.expanduser("~/.cache/huggingface"))
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")

import ssl
ssl._create_default_https_context = ssl._create_unverified_context
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""
os.environ["SSL_CERT_FILE"] = ""


def convert_pdf_to_markdown(
    input_pdf: str,
    output_dir: str = "./output",
    use_doc_orientation_classify: bool = False,
    use_doc_unwarping: bool = False,
    use_layout_detection: bool = True,
    device: str = "cpu",
) -> Path:
    """
    Convert a PDF file to Markdown using PaddleOCR-VL (Vision-Language Model).

    Args:
        input_pdf:                    Path to the input PDF file.
        output_dir:                   Directory where output is saved.
        use_doc_orientation_classify: Auto-rotate pages based on detected orientation.
        use_doc_unwarping:            Apply document unwarping (de-skew).
        use_layout_detection:         Enable layout analysis and ordering.
        device:                       Inference device: "cpu" or "gpu".

    Returns:
        Path to the output directory.
    """
    try:
        from paddleocr import PaddleOCRVL
    except ImportError:
        print(
            "ERROR: PaddleOCR is not installed.\n"
            "Run: pip install paddlepaddle \"paddleocr[doc-parser]\""
        )
        sys.exit(1)

    input_path = Path(input_pdf).resolve()
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        sys.exit(1)
    if input_path.suffix.lower() != ".pdf":
        print(f"ERROR: Input file must be a PDF. Got: {input_path.suffix}")
        sys.exit(1)

    output_path = Path(output_dir).resolve() / input_path.stem
    output_path.mkdir(parents=True, exist_ok=True)

    pages_dir = output_path / "pages"
    images_dir = output_path / "images"
    json_dir = output_path / "json"
    layout_dir = output_path / "layout"
    for d in (pages_dir, images_dir, json_dir, layout_dir):
        d.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Loading PaddleOCR-VL pipeline (device={device}, engine=transformers) ...")
    pipeline = PaddleOCRVL(
        device=device,
        engine="transformers",
        use_doc_orientation_classify=use_doc_orientation_classify,
        use_doc_unwarping=use_doc_unwarping,
        use_layout_detection=use_layout_detection,
    )

    print(f"[2/4] Processing PDF: {input_path}")
    print("      This may take a while (VLM inference is compute-intensive).")

    output = pipeline.predict(input=str(input_path))
    pages_res = list(output)

    if not pages_res:
        print("ERROR: No pages were processed. Is the PDF valid and non-empty?")
        sys.exit(1)

    print(f"[3/4] Restructuring {len(pages_res)} page(s) ...")
    restructured = pipeline.restructure_pages(
        pages_res,
        merge_tables=True,
        relevel_titles=True,
        concatenate_pages=True,
    )

    print("[4/4] Saving output ...")
    from PIL import Image as PILImage

    for page_idx, res in enumerate(pages_res, start=1):
        res.save_to_json(save_path=str(json_dir / f"page_{page_idx:03d}.json"))
        try:
            res.save_to_img(save_path=str(layout_dir))
        except Exception as e:
            print(f"      WARNING: Could not save layout image for page {page_idx}: {e}")

        md_info = res.markdown
        page_md_text = md_info.get("markdown_texts", "")
        (pages_dir / f"page_{page_idx:03d}.md").write_text(page_md_text, encoding="utf-8")

        page_images = md_info.get("markdown_images", {})
        for img_rel_path, img_data in page_images.items():
            img_dest = images_dir / img_rel_path
            img_dest.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(img_data, PILImage.Image):
                img_data.save(str(img_dest))
            elif isinstance(img_data, bytes):
                img_dest.write_bytes(img_data)
            else:
                img_dest.write_bytes(base64.b64decode(img_data))

    # Save final concatenated markdown
    for res in restructured:
        md_info = res.markdown
        markdown_text = md_info.get("markdown_texts", "")
        md_file = output_path / f"{input_path.stem}.md"
        md_file.write_text(markdown_text, encoding="utf-8")

        md_images = md_info.get("markdown_images", {})
        for img_rel_path, img_data in md_images.items():
            img_dest = images_dir / img_rel_path
            img_dest.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(img_data, PILImage.Image):
                img_data.save(str(img_dest))
            elif isinstance(img_data, bytes):
                img_dest.write_bytes(img_data)
            else:
                img_dest.write_bytes(base64.b64decode(img_data))

    print(f"\n✅ Done! Output saved to: {output_path}")
    print(f"   ├── {input_path.stem}.md  (final markdown)")
    print(f"   ├── pages/    (per-page markdown)")
    print(f"   ├── images/   (embedded images)")
    print(f"   ├── layout/   (layout detection visualizations)")
    print(f"   └── json/     (per-page structured results)")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Convert a PDF to Markdown using PaddleOCR-VL (Vision-Language Model)."
    )
    parser.add_argument(
        "input_pdf",
        help="Path to the input PDF file.",
    )
    parser.add_argument(
        "-o", "--output-dir",
        default="./output",
        help="Directory to save the output (default: ./output).",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU for inference (requires paddlepaddle-gpu).",
    )
    parser.add_argument(
        "--orientation",
        action="store_true",
        help="Enable automatic page orientation detection and correction.",
    )
    parser.add_argument(
        "--unwarp",
        action="store_true",
        help="Enable document unwarping (de-skew).",
    )
    parser.add_argument(
        "--no-layout",
        action="store_true",
        help="Disable layout analysis and ordering module.",
    )

    args = parser.parse_args()

    convert_pdf_to_markdown(
        input_pdf=args.input_pdf,
        output_dir=args.output_dir,
        use_doc_orientation_classify=args.orientation,
        use_doc_unwarping=args.unwarp,
        use_layout_detection=not args.no_layout,
        device="gpu" if args.gpu else "cpu",
    )


if __name__ == "__main__":
    main()
