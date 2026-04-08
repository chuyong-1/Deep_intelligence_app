"""
report_generator.py  v2

In-memory PDF report generator for the Verity OSINT Dashboard.

v2 changes
──────────
  • New section: "Forensic Reasoning" — renders the bullet list produced by
    image_forensics.generate_forensic_reasoning() as a clearly styled,
    numbered block with a teal left-rule accent.  Works for both image and
    (future) text reports.
  • Reasoning bullets flow through multi_cell() so long sentences wrap
    naturally across page boundaries without overflowing.
  • The `forensic_reasoning` key is optional in data_dict — missing or empty
    list silently skips the section so existing text-analysis reports are
    unaffected.

Produces a clean 1–2 page PDF containing:
  - Branded header  (Verity OSINT Report + timestamp)
  - Credibility Summary section  (score, verdict, modality, snippet)
  - Forensic Reasoning section   (NEW — numbered analyst bullets)
  - Detailed Forensic Signals section  (EXIF, deepfake, stylistic)

Install:
    pip install fpdf2
"""

from __future__ import annotations

import io
from datetime import datetime, timezone
from typing import Any

try:
    from fpdf import FPDF, XPos, YPos
except ImportError as e:
    raise ImportError(
        "fpdf2 is required for PDF generation. "
        "Run: pip install fpdf2"
    ) from e


# ── Colour palette ────────────────────────────────────────────────────────────
_BLACK      = (10,  11,  16)
_DARK_GREY  = (28,  32,  48)
_MID_GREY   = (74,  80, 104)
_LIGHT_GREY = (212, 207, 199)
_WHITE      = (245, 240, 226)
_BLUE       = (126, 184, 247)
_GREEN      = (74,  222, 128)
_AMBER      = (251, 191,  36)
_RED        = (248, 113, 113)
_TEAL       = (42,  160, 128)   # NEW — reasoning section accent

_PAGE_W    = 210   # A4 mm
_MARGIN    = 18
_CONTENT_W = _PAGE_W - 2 * _MARGIN


class VerityReport(FPDF):
    """
    FPDF subclass with Verity branding applied to every page.

    Header: branded title bar + generation timestamp.
    Footer: page number + disclaimer.
    """

    def __init__(self, timestamp: str):
        super().__init__(orientation="P", unit="mm", format="A4")
        self._timestamp = timestamp
        self.set_margins(_MARGIN, _MARGIN, _MARGIN)
        self.set_auto_page_break(auto=True, margin=20)

    # ── FPDF overrides ─────────────────────────────────────────────────────────

    def header(self):
        self.set_fill_color(*_BLACK)
        self.rect(0, 0, _PAGE_W, 18, style="F")

        self.set_font("Helvetica", "B", 13)
        self.set_text_color(*_BLUE)
        self.set_xy(_MARGIN, 4)
        self.cell(80, 8, "VERITY  OSINT  REPORT", new_x=XPos.RIGHT, new_y=YPos.TOP)

        self.set_font("Courier", "", 8)
        self.set_text_color(*_MID_GREY)
        self.set_xy(_PAGE_W - _MARGIN - 70, 6)
        self.cell(70, 5, f"Generated: {self._timestamp}", align="R",
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        self.set_draw_color(*_BLUE)
        self.set_line_width(0.4)
        self.line(_MARGIN, 19, _PAGE_W - _MARGIN, 19)
        self.ln(6)

    def footer(self):
        self.set_y(-14)
        self.set_font("Courier", "", 7)
        self.set_text_color(*_MID_GREY)
        self.cell(0, 5,
                  "Verity OSINT Dashboard  ·  This report is for informational "
                  "purposes only and does not replace human fact-checking.",
                  align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_font("Courier", "", 7)
        self.cell(0, 4, f"Page {self.page_no()}", align="C")

    # ── Layout helpers ─────────────────────────────────────────────────────────

    def section_title(self, title: str):
        """Uppercase section header with a coloured rule beneath."""
        self.ln(4)
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(*_BLUE)
        self.cell(_CONTENT_W, 5, title.upper(), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        self.set_draw_color(*_DARK_GREY)
        self.set_line_width(0.3)
        self.line(_MARGIN, self.get_y(), _PAGE_W - _MARGIN, self.get_y())
        self.ln(3)

    def kv_row(self, key: str, value: str, value_color: tuple | None = None):
        """Key / value pair on one line."""
        col_w = 52

        self.set_font("Helvetica", "B", 8)
        self.set_text_color(*_MID_GREY)
        self.cell(col_w, 5, key, new_x=XPos.RIGHT, new_y=YPos.TOP)

        self.set_font("Helvetica", "", 8)
        self.set_text_color(*(value_color or _LIGHT_GREY))
        self.multi_cell(_CONTENT_W - col_w, 5, str(value),
                        new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def verdict_banner(self, verdict: str, score: float):
        """Full-width coloured verdict banner."""
        colour = _GREEN if verdict == "Real" else (_RED if verdict == "Fake" else _AMBER)
        self.ln(3)
        self.set_fill_color(*colour)
        self.set_text_color(10, 10, 10)
        self.set_font("Helvetica", "B", 14)
        text = (
            f"{'✓' if verdict=='Real' else ('✗' if verdict=='Fake' else '?')}  "
            f"{verdict.upper()}  —  {score:.1f}% Credibility"
        )
        self.cell(_CONTENT_W, 12, text, align="C", fill=True,
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(3)

    def reasoning_bullet(self, index: int, text: str):
        """
        Render one forensic reasoning bullet as a numbered, indented paragraph
        with a teal left-rule accent drawn as a filled rectangle.

        Layout:
            │ 01  <bullet text wrapping across multiple lines if needed>
            │
            │ 02  ...

        The left rule is a 1.5 mm wide teal bar that spans the full height of
        the text block.  We achieve this by:
          1. Recording y_before.
          2. Writing the index badge and the text.
          3. Recording y_after.
          4. Drawing the rule rect between y_before and y_after.
        """
        RULE_W     = 1.5    # mm — width of teal left accent bar
        RULE_GAP   = 2.5    # mm — gap between bar and index badge
        INDEX_W    = 8.0    # mm — width of the numeric index cell
        TEXT_W     = _CONTENT_W - RULE_W - RULE_GAP - INDEX_W
        ROW_H      = 4.5    # mm — line height inside multi_cell

        x0 = _MARGIN
        y0 = self.get_y()

        # Index badge
        self.set_xy(x0 + RULE_W + RULE_GAP, y0)
        self.set_font("Courier", "B", 7)
        self.set_text_color(*_TEAL)
        self.cell(INDEX_W, ROW_H, f"{index:02d}", new_x=XPos.RIGHT, new_y=YPos.TOP)

        # Bullet text (wraps as needed)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(*_LIGHT_GREY)
        self.multi_cell(TEXT_W, ROW_H, text,
                        new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        y1 = self.get_y()

        # Draw the teal left accent rule retroactively
        self.set_fill_color(*_TEAL)
        self.rect(x0, y0, RULE_W, y1 - y0, style="F")

        # Small spacer between bullets
        self.ln(1.5)

    def signal_row(self, label: str, value: Any, signal_type: str = "neutral"):
        """One stylistic signal bar row."""
        colour = _RED if signal_type == "fake" else (_GREEN if signal_type == "real" else _AMBER)
        self.set_font("Helvetica", "", 7)
        self.set_text_color(*_MID_GREY)
        self.cell(70, 4, str(label), new_x=XPos.RIGHT, new_y=YPos.TOP)

        self.set_font("Courier", "B", 7)
        self.set_text_color(*colour)
        self.cell(30, 4, str(value), new_x=XPos.LMARGIN, new_y=YPos.NEXT)


# ── Public API ─────────────────────────────────────────────────────────────────

def generate_pdf_report(data_dict: dict) -> bytes:
    """
    Generate an in-memory PDF and return raw bytes.

    Args:
        data_dict: Analysis result dict.  Expected keys
                   (all optional — missing keys render as "N/A"):

            Common
            ──────
            modality             str    "Text" | "Image"
            final_score          float  0–100
            verdict              str    "Real" | "Uncertain" | "Fake"
            snippet              str    article excerpt or image filename
            forensic_reasoning   list[str]   NEW — bullet strings from
                                             generate_forensic_reasoning()

            Text analysis
            ─────────────
            word_count           int
            reading_time         str
            feature_type         str
            ml_score             float
            entity_score         float
            claim_score          float
            stylistic_signals    list[tuple[str, Any, str]]
            exif_warnings        list[str]

            Image analysis
            ──────────────
            ai_probability       float  0–100
            image_filename       str
            exif_data            dict
            exif_warnings        list[str]
            face_count           int
            deepfake_results     list[dict]

    Returns:
        bytes  — raw PDF content for st.download_button(data=...).
    """
    g  = lambda k, default="N/A": data_dict.get(k, default)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d  %H:%M  UTC")

    pdf = VerityReport(timestamp=ts)
    pdf.add_page()

    modality    = str(g("modality", "Text"))
    final_score = float(g("final_score", 0.0))
    verdict     = str(g("verdict", "Uncertain"))
    snippet     = str(g("snippet", ""))

    # ── 1. Credibility Summary ─────────────────────────────────────────────────
    pdf.section_title("Credibility Summary")
    pdf.verdict_banner(verdict, final_score)

    pdf.kv_row("Modality",    modality)
    pdf.kv_row("Final Score", f"{final_score:.1f} / 100",
               _GREEN if final_score >= 70 else (_RED if final_score < 40 else _AMBER))
    pdf.kv_row("Verdict",     verdict,
               _GREEN if verdict == "Real" else (_RED if verdict == "Fake" else _AMBER))

    if snippet:
        pdf.kv_row("Content Snippet", snippet[:120] + ("…" if len(snippet) > 120 else ""))

    if modality == "Text":
        for label, key in [("Word Count","word_count"),("Reading Time","reading_time"),
                           ("Feature Type","feature_type"),("ML Score","ml_score"),
                           ("Entity Score","entity_score"),("Claim Score","claim_score")]:
            val = g(key)
            if val != "N/A":
                display = f"{float(val):.1f}%" if key in ("ml_score","entity_score","claim_score") else str(val).upper() if key == "feature_type" else str(val)
                pdf.kv_row(label, display)

    if modality == "Image":
        if g("image_filename") != "N/A":
            pdf.kv_row("Image File", str(g("image_filename")))
        if g("ai_probability") != "N/A":
            ai_pct = float(g("ai_probability"))
            pdf.kv_row("AI-Gen. Probability", f"{ai_pct:.1f}%",
                       _RED if ai_pct >= 85 else (_AMBER if ai_pct >= 75 else _GREEN))
        if g("face_count") != "N/A":
            pdf.kv_row("Faces Detected", str(g("face_count")))

    # ── 2. Forensic Reasoning  [NEW v2] ───────────────────────────────────────
    reasoning: list[str] = g("forensic_reasoning", []) or []
    if reasoning:
        pdf.section_title("Forensic Reasoning")

        # Short preamble so the section has context even when read in isolation
        pdf.set_font("Helvetica", "I", 7)
        pdf.set_text_color(*_MID_GREY)
        pdf.multi_cell(
            _CONTENT_W, 4,
            "The following observations were generated automatically by the Verity "
            "forensic reasoning engine, combining signals from AI-generation detection, "
            "deepfake analysis, and EXIF metadata provenance.",
            new_x=XPos.LMARGIN, new_y=YPos.NEXT,
        )
        pdf.ln(2)

        for i, bullet in enumerate(reasoning, start=1):
            pdf.reasoning_bullet(i, bullet)

        pdf.ln(1)

    # ── 3. Detailed Forensic Signals ──────────────────────────────────────────
    pdf.section_title("Detailed Forensic Signals")

    # EXIF warnings
    exif_warnings: list[str] = g("exif_warnings", []) or []
    if exif_warnings:
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_text_color(*_AMBER)
        pdf.cell(_CONTENT_W, 5, "EXIF / Metadata Warnings",
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(*_LIGHT_GREY)
        for w in exif_warnings:
            pdf.multi_cell(_CONTENT_W, 4, f"  ⚠  {w}",
                           new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(2)

    # EXIF data table
    exif_data: dict = g("exif_data", {}) or {}
    if exif_data:
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_text_color(*_MID_GREY)
        pdf.cell(_CONTENT_W, 5, "EXIF Metadata",
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        for k, v in list(exif_data.items())[:20]:
            pdf.kv_row(str(k), str(v)[:80])
        pdf.ln(2)

    # Deepfake results
    deepfake_results: list[dict] = g("deepfake_results", []) or []
    if deepfake_results:
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_text_color(*_MID_GREY)
        pdf.cell(_CONTENT_W, 5, "Deepfake Analysis Per Face",
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        for i, dr in enumerate(deepfake_results, 1):
            df_pct = float(dr.get("deepfake_probability", 0.0))
            lbl    = str(dr.get("label", "Unknown"))
            conf   = str(dr.get("confidence", "N/A"))
            colour = _RED if df_pct >= 80 else (_AMBER if df_pct >= 60 else _GREEN)
            pdf.kv_row(f"Face {i}",
                       f"{df_pct:.1f}%  {lbl}  ({conf} confidence)",
                       colour)
        pdf.ln(2)

    # Stylistic signals
    stylistic: list[tuple] = g("stylistic_signals", []) or []
    if stylistic:
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_text_color(*_MID_GREY)
        pdf.cell(_CONTENT_W, 5, "Stylistic Signal Breakdown (16 features)",
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(1)
        col_w = _CONTENT_W // 2
        for idx in range(0, len(stylistic), 2):
            left  = stylistic[idx]
            right = stylistic[idx + 1] if idx + 1 < len(stylistic) else None
            y_before = pdf.get_y()
            pdf.set_x(_MARGIN)
            _render_signal_cell(pdf, left, col_w)
            if right:
                pdf.set_xy(_MARGIN + col_w + 2, y_before)
                _render_signal_cell(pdf, right, col_w - 2)
            else:
                pdf.ln(5)

    # ── Closing note ──────────────────────────────────────────────────────────
    pdf.ln(6)
    pdf.set_font("Helvetica", "I", 7)
    pdf.set_text_color(*_MID_GREY)
    pdf.multi_cell(
        _CONTENT_W, 4,
        "This report was generated automatically by the Verity OSINT Dashboard. "
        "Scores are probabilistic estimates and should be treated as decision-support "
        "tools, not definitive verdicts. Always verify claims through independent sources.",
        new_x=XPos.LMARGIN, new_y=YPos.NEXT,
    )

    return bytes(pdf.output())


def _render_signal_cell(pdf: VerityReport, signal: tuple, width: float):
    """Render one (label, value, signal_type) tuple inside a table cell."""
    label, value, stype = signal
    colour = _RED if stype == "fake" else (_GREEN if stype == "real" else _AMBER)

    pdf.set_font("Helvetica", "", 7)
    pdf.set_text_color(*_MID_GREY)
    pdf.cell(width * 0.65, 4, str(label)[:26], new_x=XPos.RIGHT, new_y=YPos.TOP)

    pdf.set_font("Courier", "B", 7)
    pdf.set_text_color(*colour)
    pdf.cell(width * 0.35, 4, str(value)[:12],
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)