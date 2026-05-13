#!/usr/bin/env python3
"""Generate HW2_report.pdf from the three HW2 ANSWERS.md files.

Uses pandoc + xelatex so that LaTeX math and Traditional-Chinese text render
properly. Image paths are rewritten to centered, bounded LaTeX graphics before
invoking pandoc.

Usage:
    conda run -n quantum-computing python HW2/gen_pdf.py
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
HW2 = REPO / "HW2"
OUTPUT = HW2 / "HW2_report.pdf"

ANSWERS = [
    ("Problem 1", HW2 / "problem1" / "ANSWERS.md", HW2 / "problem1"),
    ("Problem 2", HW2 / "problem2" / "ANSWERS.md", HW2 / "problem2"),
    ("Problem 3", HW2 / "problem3" / "ANSWERS.md", HW2 / "problem3"),
]

CJK_FONT_CANDIDATES = [
    "Hiragino Sans GB",
    "Hiragino Maru Gothic ProN",
    "Heiti TC",
    "Heiti SC",
    "PingFang TC",
    "PingFang SC",
    "Noto Sans CJK TC",
    "Noto Sans CJK SC",
]

MAIN_FONT_CANDIDATES = [
    "Arial Unicode MS",
    "Arial",
    "Noto Sans",
]

MONO_FONT_CANDIDATES = [
    "Menlo",
    "SF Mono",
    "Andale Mono",
]


def available_font_families() -> str:
    try:
        result = subprocess.run(
            ["fc-list", "--format=%{family}\n"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout
    except Exception:
        return ""


def pick_font(candidates: list[str], fallback: str, available: str) -> str:
    for font in candidates:
        if font in available:
            return font
    return fallback


def pick_fonts() -> tuple[str, str, str]:
    available = available_font_families()
    return (
        pick_font(MAIN_FONT_CANDIDATES, "Arial Unicode MS", available),
        pick_font(MONO_FONT_CANDIDATES, "Menlo", available),
        pick_font(CJK_FONT_CANDIDATES, "Hiragino Sans GB", available),
    )


def absolutify_images(md_text: str, base_dir: Path) -> str:
    """Replace Markdown image links with centered, non-floating LaTeX images."""

    def replace_img(match: re.Match[str]) -> str:
        alt = match.group(1)
        path = match.group(2).strip()

        if path.startswith(("http://", "https://")):
            return match.group(0)

        resolved = (base_dir / path).resolve()
        if not resolved.exists():
            print(f"  [WARN] image not found: {resolved}", file=sys.stderr)
            return match.group(0)
        return (
            "\n\n"
            "\\begin{center}\n"
            f"\\includegraphics[width=0.92\\linewidth,height=0.72\\textheight,keepaspectratio]{{{resolved}}}\n"
            "\\end{center}"
            "\n\n"
        )

    return re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", replace_img, md_text)


COVER_BLOCK = """\
---
title: "HW2 Report"
author: "D11224001 黃俊銘"
date: "2026-05-13"
header-includes:
  - \\usepackage{graphicx}
---

# Links

| | URL |
|---|---|
| GitHub | <https://github.com/jimmy60504/quantum_computing> |

\\newpage

"""


def build_combined_markdown() -> str:
    parts: list[str] = [COVER_BLOCK]

    for label, md_path, base_dir in ANSWERS:
        print(f"  Processing {md_path.relative_to(REPO)} ...")
        md_text = md_path.read_text(encoding="utf-8")
        md_text = absolutify_images(md_text, base_dir)
        parts.append(f"\n\n# {label}\n\n")
        parts.append(md_text)
        parts.append("\n\n\\newpage\n\n")

    return "".join(parts)


def run_pandoc(combined_md: str, main_font: str, mono_font: str, cjk_font: str) -> None:
    cmd = [
        "pandoc",
        "-",
        "-o",
        str(OUTPUT),
        "--pdf-engine=xelatex",
        "--toc",
        "--toc-depth=3",
        "--syntax-highlighting=tango",
        "--variable",
        "geometry:margin=2cm",
        "--variable",
        "fontsize=11pt",
        "--variable",
        "linestretch=1.25",
        "--variable",
        "colorlinks=true",
        "--variable",
        "linkcolor=blue",
        "--variable",
        "urlcolor=blue",
        "--variable",
        f"mainfont={main_font}",
        "--variable",
        f"sansfont={main_font}",
        "--variable",
        f"monofont={mono_font}",
        "--variable",
        f"CJKmainfont={cjk_font}",
        "--variable",
        f"CJKsansfont={cjk_font}",
        "--variable",
        f"CJKmonofont={cjk_font}",
        "--from",
        "markdown+tex_math_dollars+raw_tex",
    ]

    print(f"  Main font: {main_font}")
    print(f"  Mono font: {mono_font}")
    print(f"  CJK font: {cjk_font}")
    result = subprocess.run(cmd, input=combined_md.encode("utf-8"), capture_output=True)

    if result.returncode != 0:
        print("pandoc FAILED. stderr:", file=sys.stderr)
        print(result.stderr.decode("utf-8", errors="replace"), file=sys.stderr)
        sys.exit(result.returncode)

    if result.stderr:
        stderr_text = result.stderr.decode("utf-8", errors="replace")
        for line in stderr_text.splitlines():
            if line.strip():
                print(f"  [pandoc] {line}", file=sys.stderr)


def main() -> None:
    print("Building combined HW2 markdown ...")
    combined = build_combined_markdown()

    tmp = Path(tempfile.gettempdir()) / "HW2_combined.md"
    tmp.write_text(combined, encoding="utf-8")
    print(f"  Combined markdown written to: {tmp} ({len(combined):,} chars)")

    print(f"Rendering PDF -> {OUTPUT} ...")
    run_pandoc(combined, *pick_fonts())

    size_mb = OUTPUT.stat().st_size / 1_048_576
    print(f"Done. PDF written to {OUTPUT} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
