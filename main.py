import argparse
import sys
from pathlib import Path

try:
    from paddleocr import PaddleOCR
except Exception:
    print('echo Missing dependency: paddleocr. Install from requirements.txt')
    raise


def ocr_pdf_to_markdown(pdf_path, output_md_path, lang='en'):
    ocr = PaddleOCR(use_angle_cls=True, lang=lang)
    lines = []
    lines.append(f"# OCR output\n")

    try:
        result = ocr.ocr(str(pdf_path), cls=True)
    except Exception as e:
        lines.append(f"```\nOCR error: {e}\n```")
        with open(output_md_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        return

    page_num = 1
    for page_result in result:
        if not page_result:
            lines.append(f"## Page {page_num}\n")
            lines.append("*No text recognized on this page.*")
            page_num += 1
            lines.append('\n---\n')
            continue

        lines.append(f"## Page {page_num}\n")
        page_texts = []

        # Parse OCR results
        for item in page_result:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                # item format: [box_coords, (text, confidence)]
                if isinstance(item[1], (list, tuple)) and len(item[1]) > 0:
                    txt = item[1][0]
                elif isinstance(item[1], dict):
                    txt = item[1].get('text', '')
                else:
                    txt = str(item[1])
                if txt.strip():
                    page_texts.append(txt)
            elif isinstance(item, dict) and 'text' in item:
                if item['text'].strip():
                    page_texts.append(item['text'])

        if page_texts:
            lines.append("\n".join(page_texts))
        else:
            lines.append("*No text recognized on this page.*")

        lines.append('\n---\n')
        page_num += 1

    with open(output_md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))




def main():
    parser = argparse.ArgumentParser(description='Convert PDF to Markdown using PaddleOCR')
    parser.add_argument('pdf', help='Input PDF file')
    parser.add_argument('-o', '--output', help='Output markdown file', default='output.md')
    parser.add_argument('--lang', help='OCR language (default: en)', default='en')
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f'echo PDF not found: {pdf_path}')
        sys.exit(2)

    out_md = Path(args.output).resolve()
    
    print(f'echo Processing {pdf_path}...')
    ocr_pdf_to_markdown(str(pdf_path), str(out_md), lang=args.lang)
    print(f'echo Done. Markdown written to {out_md}')


if __name__ == '__main__':
    main()
