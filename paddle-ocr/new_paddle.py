"""
PDF to Markdown converter using PaddleOCR's PPStructureV3 pipeline.

Dependencies (all Apache 2.0 or MIT licensed):
  - paddlepaddle  (Apache 2.0) : https://github.com/PaddlePaddle/Paddle
  - paddleocr     (Apache 2.0) : https://github.com/PaddlePaddle/PaddleOCR

Install:
  pip install paddlepaddle paddleocr

For GPU support (optional, faster):
  pip install paddlepaddle-gpu paddleocr
"""

import argparse
import base64
import os
import sys
from pathlib import Path

os.environ.setdefault("PADDLE_PDX_PDF_RENDER_SCALE", "6.0")
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

# Bypass SSL verification for corporate proxies with self-signed certs
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
    use_common_ocr: bool = True,
    use_seal_recognition: bool = True,
    use_table_recognition: bool = True,
    use_e2e_wired_table_rec_model: bool = False,
    lang: str = "en",
    text_recognition_model_name: str | None = None,
    paddlex_config: str | None = None,
    device: str = "cpu",
) -> Path:
    """
    Convert a PDF file to a single Markdown file using PaddleOCR PPStructureV3.

    Args:
        input_pdf:                    Path to the input PDF file.
        output_dir:                   Directory where the .md file (and images) are saved.
        use_doc_orientation_classify: Auto-rotate pages based on detected orientation.
        use_doc_unwarping:            Apply document unwarping (de-skew).
        use_common_ocr:               Enable general OCR for text extraction.
        use_seal_recognition:         Enable seal/stamp recognition.
        use_table_recognition:        Enable table structure recognition.
        use_e2e_wired_table_rec_model: Use end-to-end model for bordered/wired tables.
        lang:                         Language for text recognition ("en", "ch", etc.).
        text_recognition_model_name:  Override text recognition model (e.g. "en_PP-OCRv4_mobile_rec").
        device:                       Inference device: "cpu" or "gpu".

    Returns:
        Path to the generated Markdown file.
    """
    try:
        from paddleocr import PPStructureV3
    except ImportError:
        print(
            "ERROR: PaddleOCR is not installed.\n"
            "Run: pip install paddlepaddle paddleocr"
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

    # Intermediate artifact directories
    pages_dir = output_path / "pages"
    images_dir = output_path / "images"
    layout_dir = output_path / "layout"
    json_dir = output_path / "json"
    for d in (pages_dir, images_dir, layout_dir, json_dir):
        d.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Loading PPStructureV3 pipeline (device={device}, lang={lang}) ...")
    pipeline_kwargs: dict = {"device": device, "lang": lang}
    if text_recognition_model_name:
        pipeline_kwargs["text_recognition_model_name"] = text_recognition_model_name
    if paddlex_config:
        pipeline_kwargs["paddlex_config"] = paddlex_config
    pipeline = PPStructureV3(**pipeline_kwargs)

    print(f"[2/4] Processing PDF: {input_path}")
    print("      This may take a while depending on the number of pages and your hardware.")

    predictions = pipeline.predict(
        input=str(input_path),
        use_doc_orientation_classify=use_doc_orientation_classify,
        use_doc_unwarping=use_doc_unwarping,
        use_common_ocr=use_common_ocr,
        use_seal_recognition=use_seal_recognition,
        use_table_recognition=use_table_recognition,
        use_e2e_wired_table_rec_model=use_e2e_wired_table_rec_model,
    )

    print("[3/4] Collecting per-page Markdown results ...")
    markdown_list = []
    markdown_images_list = []

    for page_idx, res in enumerate(predictions, start=1):
        md_info = res.markdown
        markdown_list.append(md_info)
        page_images = md_info.get("markdown_images", {})
        markdown_images_list.append(page_images)
        print(f"      Page {page_idx}: {len(page_images)} embedded image(s)")

        # Save per-page intermediate artifacts
        res.save_to_json(save_path=str(json_dir / f"page_{page_idx:03d}.json"))
        try:
            res.save_to_img(save_path=str(layout_dir))
        except Exception as e:
            print(f"      WARNING: Could not save layout image for page {page_idx}: {e}")

        # Save per-page markdown
        page_md_text = md_info.get("markdown_texts", "")
        (pages_dir / f"page_{page_idx:03d}.md").write_text(page_md_text, encoding="utf-8")

    if not markdown_list:
        print("ERROR: No pages were processed. Is the PDF valid and non-empty?")
        sys.exit(1)

    print("[4/4] Concatenating pages and writing output ...")

    # Merge all pages into one Markdown document
    merged = pipeline.concatenate_markdown_pages(markdown_list)

    if isinstance(merged, dict):
        markdown_text = merged.get("markdown_texts") or merged.get("markdown") or ""
    else:
        markdown_text = str(merged)

    # Save embedded images referenced in the Markdown
    from PIL import Image as PILImage

    for page_images in markdown_images_list:
        for img_rel_path, img_data in page_images.items():
            img_dest = images_dir / img_rel_path
            img_dest.parent.mkdir(parents=True, exist_ok=True)

            if isinstance(img_data, PILImage.Image):
                img_data.save(str(img_dest))
            elif isinstance(img_data, bytes):
                img_dest.write_bytes(img_data)
            else:
                img_dest.write_bytes(base64.b64decode(img_data))

    # Write the final concatenated Markdown file
    md_file = output_path / f"{input_path.stem}.md"
    md_file.write_text(markdown_text, encoding="utf-8")

    print(f"\n✅ Done! Output saved to: {output_path}")
    print(f"   ├── {input_path.stem}.md  (final markdown)")
    print(f"   ├── pages/    (per-page markdown)")
    print(f"   ├── images/   (embedded images)")
    print(f"   ├── layout/   (layout detection visualizations)")
    print(f"   └── json/     (per-page structured results)")
    return md_file


def main():
    parser = argparse.ArgumentParser(
        description="Convert a PDF to Markdown using PaddleOCR PPStructureV3."
    )
    parser.add_argument(
        "input_pdf",
        help="Path to the input PDF file.",
    )
    parser.add_argument(
        "-o", "--output-dir",
        default="./output",
        help="Directory to save the output Markdown (default: ./output).",
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
        "--no-tables",
        action="store_true",
        help="Disable table structure recognition.",
    )
    parser.add_argument(
        "--no-seals",
        action="store_true",
        help="Disable seal/stamp recognition.",
    )
    parser.add_argument(
        "--lang",
        default="en",
        help=(
            "Language for text recognition (default: en). "
            "Common values: en, ch, chinese_cht, japan, korean, th, el, "
            "fr/french, de/german, es, pt, it, ru, ar, hi. "
            "Full list: https://paddlepaddle.github.io/PaddleOCR/latest/en/version3.x/pipeline_usage/PP-StructureV3.html#5-appendix"
        ),
    )
    parser.add_argument(
        "--text-rec-model",
        default=None,
        help=(
            "Override text recognition model name. "
            "Examples: PP-OCRv5_server_rec, PP-OCRv5_mobile_rec, "
            "en_PP-OCRv4_mobile_rec, latin_PP-OCRv5_mobile_rec."
        ),
    )
    parser.add_argument(
        "--no-e2e-wired",
        action="store_true",
        help="Disable end-to-end wired table recognition model.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to a PaddleX pipeline YAML config file for advanced customization.",
    )

    args = parser.parse_args()

    convert_pdf_to_markdown(
        input_pdf=args.input_pdf,
        output_dir=args.output_dir,
        use_doc_orientation_classify=args.orientation,
        use_doc_unwarping=args.unwarp,
        use_common_ocr=True,
        use_seal_recognition=not args.no_seals,
        use_table_recognition=not args.no_tables,
        use_e2e_wired_table_rec_model=not args.no_e2e_wired,
        lang=args.lang,
        text_recognition_model_name=args.text_rec_model,
        paddlex_config=args.config,
        device="gpu" if args.gpu else "cpu",
    )


if __name__ == "__main__":
    main()