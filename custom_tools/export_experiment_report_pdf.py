"""Export EXPERIMENT_REPORT.md to compact A4 HTML and PDF."""

import argparse
from pathlib import Path
import shutil
import subprocess

import markdown


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "EXPERIMENT_REPORT.md"
DEFAULT_HTML = ROOT / "EXPERIMENT_REPORT.html"
DEFAULT_PDF = ROOT / "EXPERIMENT_REPORT.pdf"


CSS = """
@page { size: A4; margin: 9mm 10mm; }
* { box-sizing: border-box; }
body {
  margin: 0 auto; color: #202124; font-family: "Noto Sans CJK SC",
  "Microsoft YaHei", sans-serif; font-size: 9.2pt; line-height: 1.28;
}
h1 { margin: 0 0 5px; text-align: center; font-size: 15pt; }
h2 { margin: 7px 0 3px; font-size: 11.5pt; border-bottom: 1px solid #bbb; }
p { margin: 3px 0; text-align: justify; }
ol, ul { margin: 3px 0 4px; padding-left: 20px; }
li { margin: 1px 0; }
table { width: 100%; border-collapse: collapse; margin: 4px 0; font-size: 8.7pt; }
th, td { border: 1px solid #aaa; padding: 2px 4px; text-align: center; }
th { background: #eef3f7; }
img { max-width: 100%; height: auto; page-break-inside: avoid; }
a { color: #1f5f8b; text-decoration: none; }
"""


def parse_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--html", type=Path, default=DEFAULT_HTML)
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--html-only", action="store_true")
    return parser.parse_args()


def main():
    cli = parse_cli()
    source = cli.source.expanduser().resolve()
    html_path = cli.html.expanduser().resolve()
    pdf_path = cli.pdf.expanduser().resolve()
    body = markdown.markdown(
        source.read_text(encoding="utf-8"),
        extensions=("tables", "fenced_code"))
    document = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>DexGrasp Experiment Report</title><style>{}</style></head>
<body>{}</body></html>
""".format(CSS, body)
    html_path.write_text(document, encoding="utf-8")
    print("HTML={}".format(html_path))
    if cli.html_only:
        return
    chrome = (
        shutil.which("google-chrome") or shutil.which("chromium")
        or shutil.which("chromium-browser"))
    if not chrome:
        raise FileNotFoundError("Google Chrome or Chromium was not found")
    command = [
        chrome, "--headless", "--disable-gpu", "--no-sandbox",
        "--allow-file-access-from-files",
        "--print-to-pdf={}".format(pdf_path),
        "--no-pdf-header-footer", html_path.as_uri(),
    ]
    subprocess.run(command, cwd=str(ROOT), check=True)
    if not pdf_path.is_file() or pdf_path.stat().st_size == 0:
        raise RuntimeError("PDF export did not create a valid file")
    print("PDF={}".format(pdf_path))


if __name__ == "__main__":
    main()
