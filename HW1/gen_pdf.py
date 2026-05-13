#!/usr/bin/env python3
"""
gen_pdf.py — Generate HW1_report.pdf from the three ANSWERS.md files.

Uses pandoc + xelatex so that LaTeX math ($...$ and $$...$$) renders
correctly.  Traditional-Chinese text is supported via a CJK font.

Usage:
    conda run -n quantum-computing python HW1/gen_pdf.py
    # or simply:
    python HW1/gen_pdf.py
"""

import re
import subprocess
import sys
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO = Path(__file__).parent.parent  # quantum_computing/
HW1  = REPO / "HW1"

# Each entry: (section heading, path to ANSWERS.md, base dir for image resolution)
ANSWERS = [
    (
        "Problem 1",
        HW1 / "problem1" / "hf_space" / "ANSWERS.md",
        HW1 / "problem1" / "hf_space",
        HW1 / "problem1",          # parent dir (for ../report_figs/...)
    ),
    (
        "Problem 2",
        HW1 / "problem2" / "hf_space" / "ANSWERS.md",
        HW1 / "problem2" / "hf_space",
        HW1 / "problem2",
    ),
    (
        "Problem 3",
        HW1 / "problem3" / "hf_space" / "ANSWERS.md",
        HW1 / "problem3" / "hf_space",
        HW1 / "problem3",
    ),
]

OUTPUT = HW1 / "HW1_report.pdf"

# ---------------------------------------------------------------------------
# CJK font selection — try in order, pick first that fc-list reports
# ---------------------------------------------------------------------------
CJK_FONT_CANDIDATES = [
    "Hiragino Sans GB",
    "Hiragino Maru Gothic ProN",
    "Heiti TC",
    "Heiti SC",
    "Hiragino Kaku Gothic ProN",
]

def pick_cjk_font() -> str:
    try:
        result = subprocess.run(
            ["fc-list", "--format=%{family}\n"],
            capture_output=True, text=True, check=True,
        )
        available = result.stdout
    except Exception:
        available = ""
    for font in CJK_FONT_CANDIDATES:
        if font in available:
            return font
    # Fallback — xelatex will warn but not fail for ASCII-only content
    return "Hiragino Sans GB"

# ---------------------------------------------------------------------------
# Image-path absolutifier
# ---------------------------------------------------------------------------

def absolutify_images(md_text: str, hf_space_dir: Path, problem_dir: Path) -> str:
    """
    Replace every ![alt](path) with ![alt](/absolute/path).

    Resolution rules (mirrors the old WeasyPrint approach):
      - http(s)://...  → left unchanged
      - assets/...     → hf_space_dir / path
      - ../...         → problem_dir / path_without_leading_dotdot
                         (i.e.  ../report_figs/foo.png  → problem_dir/report_figs/foo.png)
      - anything else  → hf_space_dir / path
    """
    def replace_img(m: re.Match) -> str:
        alt  = m.group(1)
        path = m.group(2).strip()

        if path.startswith("http://") or path.startswith("https://"):
            return m.group(0)

        if path.startswith("../"):
            # Strip leading ../  and resolve relative to problem_dir
            rel   = path[3:]          # remove "../"
            resolved = (problem_dir / rel).resolve()
        else:
            resolved = (hf_space_dir / path).resolve()

        if not resolved.exists():
            print(f"  [WARN] image not found: {resolved}", file=sys.stderr)
            return m.group(0)   # leave as-is rather than break the PDF

        return f"![{alt}]({resolved})"

    return re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', replace_img, md_text)

# ---------------------------------------------------------------------------
# Cover page / YAML frontmatter
# ---------------------------------------------------------------------------

COVER_BLOCK = """\
---
title: "HW1 Report"
author: "jimmy60504"
date: "2026-04-16"
---

# Links

| | URL |
|---|---|
| GitHub | <https://github.com/jimmy60504/quantum_computing> |
| Problem 1 HF | <https://huggingface.co/spaces/jimmy60504/Data-Reuploading-Demo> |
| Problem 2 HF | <https://huggingface.co/spaces/jimmy60504/QML-Classifier-Explorer> |
| Problem 3 HF | <https://huggingface.co/spaces/jimmy60504/Hybrid-QNN-Explorer> |

\\newpage

"""

# ---------------------------------------------------------------------------
# Build combined markdown
# ---------------------------------------------------------------------------

def build_combined_markdown() -> str:
    parts: list[str] = [COVER_BLOCK]

    for label, md_path, hf_space_dir, problem_dir in ANSWERS:
        print(f"  Processing {md_path.relative_to(REPO)} …")
        md_text = md_path.read_text(encoding="utf-8")
        md_text = absolutify_images(md_text, hf_space_dir, problem_dir)

        # Insert a section heading + page-break before each problem
        parts.append(f"\n\n# {label}\n\n")
        parts.append(md_text)
        parts.append("\n\n\\newpage\n\n")

    return "".join(parts)

# ---------------------------------------------------------------------------
# Run pandoc
# ---------------------------------------------------------------------------

def run_pandoc(combined_md: str, cjk_font: str) -> None:
    cmd = [
        "pandoc",
        "-",                        # read from stdin
        "-o", str(OUTPUT),
        "--pdf-engine=xelatex",
        "--toc",
        "--toc-depth=3",
        "--highlight-style=tango",
        # Page layout
        "--variable", "geometry:margin=2cm",
        "--variable", "fontsize=11pt",
        "--variable", "linestretch=1.4",
        # Link colours
        "--variable", "colorlinks=true",
        "--variable", "linkcolor=blue",
        "--variable", "urlcolor=blue",
        # CJK support
        "--variable", f"CJKmainfont={cjk_font}",
        "--variable", f"CJKsansfont={cjk_font}",
        "--variable", f"CJKmonofont={cjk_font}",
        # Pandoc markdown extensions we want
        "--from", "markdown+tex_math_dollars+raw_tex",
    ]

    print(f"  pandoc command:\n    {' '.join(cmd[:6])} ...")
    print(f"  CJK font: {cjk_font}")

    result = subprocess.run(
        cmd,
        input=combined_md.encode("utf-8"),
        capture_output=True,
    )

    if result.returncode != 0:
        print("pandoc FAILED.  stderr:", file=sys.stderr)
        print(result.stderr.decode("utf-8", errors="replace"), file=sys.stderr)
        sys.exit(result.returncode)

    if result.stderr:
        # Print warnings (xelatex often emits harmless ones)
        stderr_text = result.stderr.decode("utf-8", errors="replace")
        for line in stderr_text.splitlines():
            if line.strip():
                print(f"  [pandoc] {line}", file=sys.stderr)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Building combined markdown …")
    combined = build_combined_markdown()

    cjk_font = pick_cjk_font()

    # Write temp file for debugging / inspection
    tmp = Path(tempfile.gettempdir()) / "HW1_combined.md"
    tmp.write_text(combined, encoding="utf-8")
    print(f"  Combined markdown written to: {tmp}  ({len(combined):,} chars)")

    print(f"Rendering PDF → {OUTPUT} …")
    run_pandoc(combined, cjk_font)

    size_mb = OUTPUT.stat().st_size / 1_048_576
    print(f"Done.  PDF written to {OUTPUT}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
