from __future__ import annotations

import html
import re
from pathlib import Path

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


REPORT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = REPORT_DIR / "pdf"
FONT_NAME = "STSong-Light"


def inline_markup(text: str) -> str:
    text = html.escape(text, quote=True)
    text = re.sub(
        r"\[([^]]+)]\(([^)]+)\)",
        lambda m: f'<link href="{m.group(2)}" color="#1F5A94"><u>{m.group(1)}</u></link>',
        text,
    )
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


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pdfmetrics.registerFont(UnicodeCIDFont(FONT_NAME))
    build("BASIC_RETARGETING_REPORT.md", "basic_retargeting_report.pdf")
    build("ADVANCED_POLICY_REPORT.md", "advanced_policy_report.pdf")


if __name__ == "__main__":
    main()
