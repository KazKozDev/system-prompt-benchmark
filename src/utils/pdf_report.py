"""PDF export helper for benchmark reports"""

import io
from datetime import datetime
from typing import Dict, List
from xml.sax.saxutils import escape

import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

PAGE_WIDTH, PAGE_HEIGHT = A4
styles = getSampleStyleSheet()
H2 = ParagraphStyle(
    'Heading2',
    parent=styles['Heading2'],
    spaceAfter=10,
    textColor=colors.HexColor('#1f77b4'),
)
BODY = ParagraphStyle(
    'Body',
    parent=styles['BodyText'],
    fontSize=10,
    leading=14,
)


def _build_category_bar_chart(category_scores: Dict[str, float]) -> io.BytesIO:
    fig, ax = plt.subplots(figsize=(6, 3))
    cats = list(category_scores.keys())
    scores = [category_scores[c] for c in cats]
    bars = ax.barh(cats, scores, color='#1f77b4')
    ax.set_xlim(0, 1)
    ax.set_xlabel('Score (0-1)')
    ax.set_title('Category Scores')
    for bar, score in zip(bars, scores):
        ax.text(score + 0.01, bar.get_y() + bar.get_height()/2,
                f"{score:.2f}", va='center', fontsize=9)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300)
    plt.close(fig)
    buf.seek(0)
    return buf


def _build_radar_chart(category_scores: Dict[str, float]) -> io.BytesIO:
    if not category_scores:
        return None
    categories = list(category_scores.keys())
    values = [category_scores[c] for c in categories]
    values += values[:1]
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(4.5, 4.5), subplot_kw=dict(polar=True))
    ax.plot(angles, values, 'o-', linewidth=2, color='#ff7f0e')
    ax.fill(angles, values, alpha=0.25, color='#ff7f0e')
    ax.set_thetagrids(np.degrees(angles[:-1]), categories)
    ax.set_ylim(0, 1)
    ax.set_title('Radar Overview', pad=20)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300)
    plt.close(fig)
    buf.seek(0)
    return buf


def generate_pdf_report(report_data: Dict) -> bytes:
    """Return PDF bytes for benchmark report."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=50,
        bottomMargin=50,
    )

    story: List = []
    meta = report_data.get('metadata', {})
    cat_scores = report_data.get('category_scores', {})
    results = report_data.get('results', [])
    overall = report_data.get('overall_score', 0.0)
    prompt_text = report_data.get('prompt_text', '')

    story.append(Paragraph('Universal System Prompt Benchmark', styles['Title']))
    subtitle = (
        f"Provider: {meta.get('provider', 'N/A')}  |  Tests: {meta.get('num_tests', len(results))}  "
        f"|  Date: {meta.get('timestamp', datetime.utcnow().isoformat())[:19].replace('T', ' ')}"
    )
    story.append(Paragraph(subtitle, BODY))
    story.append(Spacer(1, 12))

    story.append(Paragraph(f"Overall Score: <b>{overall:.2f}</b>", H2))
    story.append(Paragraph(
        "Score reflects weighted performance across 12 universal categories (0 = poor, 1 = excellent).",
        BODY,
    ))
    story.append(Spacer(1, 12))

    if prompt_text:
        story.append(Paragraph('System Prompt Evaluated', H2))
        escaped_prompt = escape(prompt_text).replace('\n', '<br/>')
        story.append(Paragraph(escaped_prompt, BODY))
        story.append(Spacer(1, 12))

    # Category table
    story.append(Paragraph('Category Breakdown', H2))
    table_data = [['Category', 'Score', 'Tests']]
    for cat, score in cat_scores.items():
        tests = sum(1 for r in results if r.get('universal_category') == cat)
        table_data.append([cat.replace('_', ' ').title(), f"{score:.2f}", tests])
    table = Table(table_data, hAlign='LEFT', colWidths=[200, 80, 60])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (1, 1), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.whitesmoke, colors.lightgrey]),
        ('GRID', (0, 0), (-1, -1), 0.25, colors.grey),
    ]))
    story.append(table)
    story.append(Spacer(1, 18))

    # Charts
    if cat_scores:
        story.append(Paragraph('Visual Summary', H2))
        bar_buf = _build_category_bar_chart(cat_scores)
        story.append(Image(bar_buf, width=6.0 * inch, height=3.0 * inch))
        radar_buf = _build_radar_chart(cat_scores)
        if radar_buf:
            story.append(Spacer(1, 12))
            story.append(Image(radar_buf, width=4.5 * inch, height=4.5 * inch))
        story.append(Spacer(1, 18))

    # Top opportunities
    story.append(Paragraph('Focus Areas', H2))
    sorted_cats = sorted(cat_scores.items(), key=lambda kv: kv[1])
    bullet_points = []
    for cat, score in sorted_cats[:3]:
        bullet_points.append(
            Paragraph(
                f"• <b>{cat.replace('_', ' ').title()}</b>: score {score:.2f}. "
                "Investigate failed tests for specific remediation steps.",
                BODY,
            )
        )
    story.extend(bullet_points or [Paragraph('All categories performed consistently.', BODY)])
    story.append(Spacer(1, 18))

    # Appendix - sample failures
    story.append(Paragraph('Sample Test Results', H2))
    worst = sorted(results, key=lambda r: r.get('score', 0))[:5]
    for res in worst:
        story.append(Paragraph(
            f"<b>Test {res.get('test_id')}</b> · {res.get('universal_category', 'unknown')} · Score {res.get('score', 0):.2f}",
            BODY,
        ))
        story.append(Paragraph(f"Input: {res.get('input', '')[:400]}", BODY))
        story.append(Paragraph(f"Response: {res.get('response', '')[:400]}", BODY))
        story.append(Spacer(1, 6))

    doc.build(story)
    buffer.seek(0)
    return buffer.read()
