"""Render REPORT.md to a linked A4 PDF with embedded Cyrillic fonts.

Run with Python containing reportlab. Override LM_REPORT_FONT_DIR when needed.
No training, inference or network access is involved.
"""

import html
import os
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    HRFlowable, Image, KeepTogether, Paragraph, Preformatted, SimpleDocTemplate,
    Spacer, Table, TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'REPORT.md'
OUTPUT = ROOT / 'output/pdf/lm_hw_report.pdf'
GITHUB = 'https://github.com/maria-igntv/lm-hw'
BLUE = colors.HexColor('#24436B')
GRAY = colors.HexColor('#5B6472')
INK = colors.HexColor('#202936')


def fonts():
    candidates = [Path(os.environ['LM_REPORT_FONT_DIR'])] if os.environ.get('LM_REPORT_FONT_DIR') else []
    candidates += [Path('/System/Library/Fonts/Supplemental'),
                   Path('/usr/share/fonts/truetype/dejavu')]
    for folder in candidates:
        if (folder / 'Arial.ttf').exists():
            files = {'Body':'Arial.ttf', 'Body-Bold':'Arial Bold.ttf',
                     'Body-Italic':'Arial Italic.ttf', 'Body-BoldItalic':'Arial Bold Italic.ttf',
                     'Mono':'Courier New.ttf'}
        elif (folder / 'DejaVuSans.ttf').exists():
            files = {'Body':'DejaVuSans.ttf', 'Body-Bold':'DejaVuSans-Bold.ttf',
                     'Body-Italic':'DejaVuSans-Oblique.ttf', 'Body-BoldItalic':'DejaVuSans-BoldOblique.ttf',
                     'Mono':'DejaVuSansMono.ttf'}
        else:
            continue
        if all((folder / f).exists() for f in files.values()):
            for name, file in files.items(): pdfmetrics.registerFont(TTFont(name, str(folder / file)))
            pdfmetrics.registerFontFamily('Body', normal='Body', bold='Body-Bold',
                                          italic='Body-Italic', boldItalic='Body-BoldItalic')
            return
    raise RuntimeError('Provide Arial or DejaVu TTF fonts using LM_REPORT_FONT_DIR')


def normalize(text):
    return text.translate(str.maketrans({'\u2010':'-', '\u2011':'-', '\u2012':'-',
                                       '\u2013':'-', '\u2014':'-', '\u2212':'-', '\u21d2':'=>'}))


def link_url(target):
    if target.startswith(('https://', 'http://', 'mailto:')): return target
    path, _, anchor = target.partition('#')
    kind = 'tree' if (ROOT / path).is_dir() else 'blob'
    return f'{GITHUB}/{kind}/main/{path}' + ('#' + anchor if anchor else '')


def inline(text):
    text = normalize(text)
    pattern = re.compile(r'\[([^\]]+)\]\(([^)]+)\)|\*\*(.+?)\*\*|`([^`]+)`')
    output, start = [], 0
    for match in pattern.finditer(text):
        output.append(html.escape(text[start:match.start()]))
        if match.group(1) is not None:
            output.append('<a href="' + html.escape(link_url(match.group(2)), quote=True)
                          + '" color="#285A8B">' + html.escape(match.group(1)) + '</a>')
        elif match.group(3) is not None:
            output.append('<b>' + html.escape(match.group(3)) + '</b>')
        else:
            output.append('<font name="Mono" size="8.5">' + html.escape(match.group(4)) + '</font>')
        start = match.end()
    output.append(html.escape(text[start:]))
    return ''.join(output)


class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.states = []

    def showPage(self):
        self.states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        total = len(self.states)
        for state in self.states:
            self.__dict__.update(state)
            self.setFont('Body', 8)
            self.setFillColor(GRAY)
            self.drawString(44, 24, 'Мария Игнатова | 19.11.2026')
            self.drawRightString(A4[0] - 44, 24, f'{self._pageNumber} / {total}')
            super().showPage()
        super().save()


def page_header(canv, doc):
    canv.saveState()
    canv.setFont('Body', 8)
    canv.setFillColor(GRAY)
    canv.drawString(44, A4[1] - 26, 'LM HW | Дом и квартира | SFT и оценка безопасности')
    canv.setStrokeColor(colors.HexColor('#DAE0E8'))
    canv.setLineWidth(.5)
    canv.line(44, A4[1] - 33, A4[0] - 44, A4[1] - 33)
    canv.restoreState()


def build():
    fonts()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    body = ParagraphStyle('ReportBody', fontName='Body', fontSize=10.4, leading=14.3,
        textColor=INK, spaceAfter=6, alignment=TA_LEFT, splitLongWords=True)
    h1 = ParagraphStyle('ReportTitle', parent=body, fontName='Body-Bold', fontSize=21,
                        leading=25, textColor=BLUE, spaceAfter=11, keepWithNext=True)
    h2 = ParagraphStyle('ReportH2', parent=body, fontName='Body-Bold', fontSize=13.2,
                        leading=17, textColor=BLUE, spaceBefore=10, spaceAfter=6, keepWithNext=True)
    quote = ParagraphStyle('ReportQuote', parent=body, fontSize=10, leading=13.7,
                           leftIndent=12, rightIndent=8, borderColor=colors.HexColor('#DCE4EF'),
                           borderWidth=.5, borderPadding=7, backColor=colors.HexColor('#F6F8FB'),
                           spaceBefore=5, spaceAfter=9)
    table_body = ParagraphStyle('TableBody', parent=body, fontSize=9.1, leading=12, spaceAfter=0)
    table_head = ParagraphStyle('TableHead', parent=table_body, fontName='Body-Bold', textColor=colors.white)
    caption = ParagraphStyle('Caption', parent=body, fontSize=8.3, leading=11, textColor=GRAY,
                             spaceBefore=3, spaceAfter=9)
    code_style = ParagraphStyle('Code', fontName='Mono', fontSize=8.2, leading=11,
                                textColor=INK, leftIndent=6, spaceBefore=4, spaceAfter=8)
    document = SimpleDocTemplate(str(OUTPUT), pagesize=A4, rightMargin=44, leftMargin=44,
        topMargin=46, bottomMargin=42, title='Дом и квартира: SFT и проверка безопасности',
        author='Мария Игнатова', subject='Отчёт по LM HW: данные, baseline, QLoRA, оценка и артефакты')
    story, lines, index = [], SOURCE.read_text().splitlines(), 0
    while index < len(lines):
        line = lines[index].strip()
        if not line:
            index += 1
            continue
        if line.startswith('```'):
            block = []
            index += 1
            while index < len(lines) and not lines[index].startswith('```'):
                block.append(normalize(lines[index])); index += 1
            story.append(Preformatted('\n'.join(block), code_style, maxLineLength=94,
                                      splitChars=' ', newLineChars='    '))
            index += 1
            continue
        if line.startswith('|'):
            table_lines = []
            while index < len(lines) and lines[index].strip().startswith('|'):
                table_lines.append(lines[index].strip()); index += 1
            data = [[cell.strip() for cell in row.strip('|').split('|')] for row in table_lines]
            data = [data[0], *data[2:]]
            n = len(data[0])
            require = all(len(r) == n for r in data)
            if not require: raise ValueError('Irregular Markdown table')
            first = .46 if n > 2 else .38
            widths = [document.width * first] + [document.width * (1 - first) / (n - 1)] * (n - 1)
            cells = [[Paragraph(inline(cell), table_head if i == 0 else table_body)
                      for cell in row] for i, row in enumerate(data)]
            table = Table(cells, colWidths=widths, repeatRows=1, hAlign='LEFT', spaceBefore=4, spaceAfter=10)
            table.setStyle(TableStyle([
                ('BACKGROUND',(0,0),(-1,0),BLUE), ('VALIGN',(0,0),(-1,-1),'TOP'),
                ('LEFTPADDING',(0,0),(-1,-1),7), ('RIGHTPADDING',(0,0),(-1,-1),7),
                ('TOPPADDING',(0,0),(-1,-1),6), ('BOTTOMPADDING',(0,0),(-1,-1),6),
                ('LINEBELOW',(0,0),(-1,0),.4,BLUE),
                ('LINEBELOW',(0,1),(-1,-1),.35,colors.HexColor('#DBE1E8')),
                ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#F4F6F9')]),
            ]))
            story.append(table)
            continue
        match = re.fullmatch(r'!\[([^\]]*)\]\(([^)]+)\)', line)
        if match:
            figure = Image(str(ROOT / match.group(2)))
            width = document.width
            figure.drawHeight = figure.imageHeight * width / figure.imageWidth
            figure.drawWidth = width
            story.append(KeepTogether([Spacer(1,4), figure, Paragraph(inline(match.group(1)), caption)]))
            index += 1
            continue
        if line.startswith('# '):
            story.append(Paragraph(inline(line[2:]), h1)); index += 1; continue
        if line.startswith('## '):
            story.append(Paragraph(inline(line[3:]), h2)); index += 1; continue
        if line.startswith('>'):
            parts = []
            while index < len(lines) and lines[index].strip().startswith('>'):
                parts.append(lines[index].strip()[1:].strip()); index += 1
            story.append(Paragraph(inline(' '.join(parts)), quote))
            continue
        paragraph = [line]
        index += 1
        while index < len(lines) and lines[index].strip() and not re.match(r'^(#|\||>|```|!\[)', lines[index].strip()):
            paragraph.append(lines[index].strip()); index += 1
        story.append(Paragraph(inline(' '.join(paragraph)), body))
    document.build(story, onFirstPage=page_header, onLaterPages=page_header, canvasmaker=NumberedCanvas)
    print(OUTPUT)


if __name__ == '__main__':
    build()
