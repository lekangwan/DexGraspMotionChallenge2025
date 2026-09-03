from __future__ import annotations

import argparse
import html
import re
import subprocess
from pathlib import Path

import markdown

try:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    from reportlab.platypus import (
        Image,
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


REPORT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = REPORT_DIR / "pdf"
FONT_NAME = "STSong-Light"


def inline_markup(text: str) -> str:
    text = html.escape(text, quote=True)
    # PDF是可独立移动的提交物：链接只保留可读标签，不写入失效的外部超链接。
    text = re.sub(r"\[([^]]+)]\(([^)]+)\)", lambda m: m.group(1), text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`([^`]+)`", r'<font color="#333333">\1</font>', text)
    return text


def styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "ChineseTitle", parent=base["Title"], fontName=FONT_NAME,
            fontSize=17, leading=22, alignment=TA_CENTER,
            textColor=colors.HexColor("#17365D"), spaceAfter=6 * mm,
        ),
        "h2": ParagraphStyle(
            "ChineseH2", parent=base["Heading2"], fontName=FONT_NAME,
            fontSize=11.5, leading=15, textColor=colors.HexColor("#17365D"),
            spaceBefore=2.5 * mm, spaceAfter=1.5 * mm, keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "ChineseBody", parent=base["BodyText"], fontName=FONT_NAME,
            fontSize=8.6, leading=12.8, alignment=TA_JUSTIFY,
            firstLineIndent=2 * 8.6, spaceAfter=1.6 * mm,
            textColor=colors.HexColor("#222222"),
        ),
        "caption": ParagraphStyle(
            "ChineseCaption", parent=base["BodyText"], fontName=FONT_NAME,
            fontSize=7.5, leading=10, alignment=TA_CENTER,
            textColor=colors.HexColor("#555555"), spaceAfter=1.5 * mm,
        ),
        "table": ParagraphStyle(
            "ChineseTable", parent=base["BodyText"], fontName=FONT_NAME,
            fontSize=7.2, leading=9.5, alignment=TA_LEFT,
        ),
    }


def image_flowable(source: Path, alt: str, style: ParagraphStyle):
    image = Image(str(source))
    max_width, max_height = 165 * mm, 56 * mm
    scale = min(max_width / image.imageWidth, max_height / image.imageHeight)
    image.drawWidth = image.imageWidth * scale
    image.drawHeight = image.imageHeight * scale
    image.hAlign = "CENTER"
    return [Spacer(1, 1.2 * mm), image, Paragraph(f"图：{inline_markup(alt)}", style)]


def table_flowable(rows: list[list[str]], style: ParagraphStyle) -> Table:
    cells = [[Paragraph(inline_markup(cell), style) for cell in row] for row in rows]
    columns = len(rows[0])
    available = 180 * mm
    if columns == 4:
        widths = [31 * mm, 69 * mm, 40 * mm, 40 * mm]
    else:
        widths = [available / columns] * columns
    table = Table(cells, colWidths=widths, repeatRows=1, hAlign="CENTER")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DCE6F1")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#17365D")),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#9EADBA")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F7FA")]),
    ]))
    return table


def parse_markdown(path: Path, report_styles: dict[str, ParagraphStyle]):
    lines = path.read_text(encoding="utf-8").splitlines()
    story = []
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if not line:
            index += 1
            continue
        if line == "<!-- PAGE_BREAK -->":
            story.append(PageBreak())
            index += 1
            continue
        if line.startswith("# "):
            story.append(Paragraph(inline_markup(line[2:]), report_styles["title"]))
            index += 1
            continue
        if line.startswith("## "):
            story.append(Paragraph(inline_markup(line[3:]), report_styles["h2"]))
            index += 1
            continue
        image_match = re.fullmatch(r"!\[([^]]*)]\(([^)]+)\)", line)
        if image_match:
            source = (path.parent / image_match.group(2)).resolve()
            story.extend(image_flowable(source, image_match.group(1), report_styles["caption"]))
            index += 1
            continue
        if line.startswith("|"):
            table_lines = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                table_lines.append(lines[index].strip())
                index += 1
            rows = [[cell.strip() for cell in row.strip("|").split("|")] for row in table_lines]
            if len(rows) > 1 and all(set(cell) <= {"-", ":"} for cell in rows[1]):
                rows.pop(1)
            story.append(table_flowable(rows, report_styles["table"]))
            story.append(Spacer(1, 1.5 * mm))
            continue
        paragraph = [line]
        index += 1
        while index < len(lines):
            next_line = lines[index].strip()
            if not next_line or next_line.startswith(("#", "|", "![", "<!-- PAGE_BREAK")):
                break
            paragraph.append(next_line)
            index += 1
        story.append(Paragraph(inline_markup(" ".join(paragraph)), report_styles["body"]))
    return story


def footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont(FONT_NAME, 7.5)
    canvas.setFillColor(colors.HexColor("#666666"))
    canvas.drawCentredString(A4[0] / 2, 7 * mm, f"第 {doc.page} 页")
    canvas.restoreState()


def build(markdown_name: str, pdf_name: str) -> None:
    markdown_path = REPORT_DIR / markdown_name
    pdf_path = OUTPUT_DIR / pdf_name
    document = SimpleDocTemplate(
        str(pdf_path), pagesize=A4,
        leftMargin=15 * mm, rightMargin=15 * mm,
        topMargin=12 * mm, bottomMargin=13 * mm,
        title=markdown_path.stem,
    )
    document.build(parse_markdown(markdown_path, styles()), onFirstPage=footer, onLaterPages=footer)


def build_with_chrome(markdown_name: str, pdf_name: str) -> None:
    """ReportLab不可用时，用本机Chrome从自包含HTML生成A4 PDF。"""
    source = REPORT_DIR / markdown_name
    text = source.read_text(encoding="utf-8")
    text = re.sub(r"(?<!!)\[([^]]+)]\(([^)]+)\)", r"\1", text)
    text = text.replace("<!-- PAGE_BREAK -->", '<div class="page-break"></div>')
    body = markdown.markdown(text, extensions=["tables"])
    document = f"""<!doctype html><html><head><meta charset="utf-8"><style>
@page {{ size: A4; margin: 12mm 15mm 13mm; }}
* {{ box-sizing: border-box; }}
body {{ font-family: 'Noto Sans CJK SC','Noto Sans CJK JP',sans-serif; color:#222;
       font-size:8.6pt; line-height:1.48; margin:0; }}
h1 {{ color:#17365D; text-align:center; font-size:17pt; margin:0 0 6mm; }}
h1 + p {{ text-align:center; text-indent:0; }}
h2 {{ color:#17365D; font-size:11.5pt; margin:2.5mm 0 1.5mm; break-after:avoid; }}
p {{ margin:0 0 1.6mm; text-align:justify; text-indent:2em; }}
table {{ width:100%; border-collapse:collapse; font-size:7.2pt; margin:1.5mm 0; }}
th,td {{ border:0.35pt solid #9EADBA; padding:1.1mm; text-align:left; }}
th {{ background:#DCE6F1; color:#17365D; }}
tr:nth-child(odd) td {{ background:#F5F7FA; }}
img {{ display:block; max-width:165mm; max-height:56mm; margin:1.2mm auto 0; }}
.page-break {{ break-before:page; page-break-before:always; }}
code {{ color:#333; font-family:inherit; }}
</style></head><body>{body}</body></html>"""
    # 临时HTML与Markdown同目录，使`figures/...`图片路径可直接解析并嵌入PDF。
    html_path = REPORT_DIR / f".{Path(pdf_name).stem}.html"
    html_path.write_text(document, encoding="utf-8")
    pdf_path = (OUTPUT_DIR / pdf_name).resolve()
    subprocess.run([
        "/usr/bin/google-chrome", "--headless", "--no-sandbox", "--disable-gpu",
        "--allow-file-access-from-files", "--no-pdf-header-footer",
        f"--print-to-pdf={pdf_path}", html_path.resolve().as_uri(),
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    html_path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report", choices=("basic", "advanced", "all"), default="all",
        help="只重建指定报告；进阶结果未冻结时可只生成basic。",
    )
    args = parser.parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if REPORTLAB_AVAILABLE:
        pdfmetrics.registerFont(UnicodeCIDFont(FONT_NAME))
    builder = build if REPORTLAB_AVAILABLE else build_with_chrome
    if args.report in ("basic", "all"):
        builder("BASIC_RETARGETING_REPORT.md", "basic_retargeting_report.pdf")
    if args.report in ("advanced", "all"):
        builder("ADVANCED_POLICY_REPORT.md", "advanced_policy_report.pdf")


if __name__ == "__main__":
    main()
