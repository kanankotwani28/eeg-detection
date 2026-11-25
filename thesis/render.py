"""Render thesis/content.py to LaTeX (thesis/latex/thesis.tex) and Word (thesis/Thesis_Kanan_Kotwani.docx).
Run: python thesis/render.py  (python-docx needed for the Word output)."""
from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import content  # noqa: E402

D = content.build()
META, BLOCKS, REFS = D["meta"], D["blocks"], D["refs"]

# ------------------------------------------------------------------ numbering pre-pass
FIGNUM, TABNUM, CIT = {}, {}, {}
SEC_NUM = []
ch = sec = nf = nt = 0
for b in BLOCKS:
    if b[0] == "chapter":
        ch += 1
        sec = nf = nt = 0
    elif b[0] == "section":
        sec += 1
    elif b[0] == "fig":
        nf += 1
        FIGNUM[b[1]] = f"{ch}.{nf}"
    elif b[0] == "table":
        nt += 1
        TABNUM[b[1]] = f"{ch}.{nt}"


def scan_cites(text):
    for k in re.findall(r"\[@([A-Za-z0-9_]+)\]", text):
        if k not in REFS:
            raise KeyError(f"unknown reference {k}")
        CIT.setdefault(k, len(CIT) + 1)


for b in BLOCKS:
    for part in b[1:]:
        if isinstance(part, str):
            scan_cites(part)
        elif isinstance(part, list):
            for x in part:
                if isinstance(x, str):
                    scan_cites(x)
                elif isinstance(x, list | tuple):
                    for y in x:
                        if isinstance(y, str):
                            scan_cites(y)
for k in REFS:
    if k not in CIT:
        print("warning: reference never cited:", k)


# ------------------------------------------------------------------ LaTeX
UNI = [("e^ε", r"$e^{\varepsilon}$"), ("10⁻³", r"$10^{-3}$"), ("10⁻⁵", r"$10^{-5}$"), ("ε", r"$\varepsilon$"),
       ("δ", r"$\delta$"), ("σ", r"$\sigma$"), ("α", r"$\alpha$"), ("β", r"$\beta$"), ("γ", r"$\gamma$"),
       ("θ", r"$\theta$"), ("Δ", r"$\Delta$"), ("Σ", r"$\Sigma$"), ("≥", r"$\geq$"), ("≤", r"$\leq$"),
       ("≈", r"$\approx$"), ("∞", r"$\infty$"), ("∈", r"$\in$"), ("×", r"$\times$"), ("→", r"$\rightarrow$"),
       ("±", r"$\pm$"), ("µ", r"\textmu{}"), ("−", r"$-$"), ("·", r"$\cdot$"), ("²", r"$^2$"), ("—", "---"),
       ("–", "--"), ("’", "'"), ("“", "``"), ("”", "''")]


def tex_inline(s: str) -> str:
    keep = []

    def stash(txt):
        keep.append(txt)
        return f"\x00{len(keep) - 1}\x00"

    s = re.sub(r"\[@([A-Za-z0-9_]+)\]", lambda m: stash(r"\cite{" + m.group(1) + "}"), s)
    s = re.sub(r"⟦fig:([A-Za-z0-9_]+)⟧", lambda m: stash(r"Figure~\ref{fig:" + m.group(1) + "}"), s)
    s = re.sub(r"⟦tab:([A-Za-z0-9_]+)⟧", lambda m: stash(r"Table~\ref{tab:" + m.group(1) + "}"), s)
    s = s.replace("\\", r"\textbackslash{}")
    for a, b in (("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}"),
                 ("~", r"\textasciitilde{}")):
        s = s.replace(a, b)
    s = s.replace("^", r"\textasciicircum{}").replace(r"e\textasciicircum{}ε", "e^ε")
    for a, b in UNI:
        s = s.replace(a, b)
    s = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", s)
    s = re.sub(r"(?<![\\\w])\*(.+?)\*", r"\\emph{\1}", s)
    s = re.sub(r"(\d) (?=\d{3}\b)", r"\1\\,", s)
    s = re.sub("\x00(\\d+)\x00", lambda m: keep[int(m.group(1))], s)
    return s


def short(cap: str) -> str:
    """List-of-figures entry: the caption up to its first sentence end."""
    m = re.match(r"(.+?[^.])\. ", cap)
    return m.group(1) if m else cap.rstrip(".")


def tex_colspec(spec: str) -> str:
    return "".join({"l": "l", "r": "r", "c": "c", "X": "X"}[ch_] for ch_ in spec)


def render_tex() -> str:
    m = META
    T = []
    w = T.append
    w(r"""\documentclass[12pt,a4paper,oneside]{report}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{mathptmx}
\usepackage{textcomp}
\usepackage[a4paper,left=2.8cm,right=2.5cm,top=2.5cm,bottom=2.5cm]{geometry}
\usepackage{graphicx,booktabs,tabularx,array,amsmath,setspace,enumitem,float}
\usepackage[labelfont=bf,labelsep=period,font=small]{caption}
\usepackage{titlesec,tocloft,fancyhdr}
\usepackage[hidelinks]{hyperref}
\setstretch{1.3}
\setlength{\parindent}{0pt}
\setlength{\emergencystretch}{2em}
\frenchspacing
\usepackage[section]{placeins}
\usepackage{xcolor}
\definecolor{titlegreen}{HTML}{EEF8EE}
\setlength{\parskip}{6pt}
\graphicspath{{figures/}}
\renewcommand{\thefigure}{\thechapter.\arabic{figure}}
\renewcommand{\thetable}{\thechapter.\arabic{table}}
\captionsetup[table]{position=above,skip=4pt}
\renewcommand{\tabularxcolumn}[1]{p{#1}}
% chapter and section headings in the institute style
\titleformat{\chapter}[display]{\centering\bfseries}{\large CHAPTER \thechapter}{8pt}{\LARGE\MakeUppercase}
\titlespacing*{\chapter}{0pt}{-20pt}{18pt}
\titleformat{\section}{\normalsize\bfseries}{\thesection}{0.8em}{\MakeUppercase}
\titlespacing*{\section}{0pt}{12pt}{4pt}
\titleformat{\subsection}{\normalsize\bfseries}{\thesubsection}{0.8em}{}
% contents
\renewcommand{\cfttoctitlefont}{\hfill\Large\bfseries}\renewcommand{\cftaftertoctitle}{\hfill}
\renewcommand{\cftloftitlefont}{\hfill\Large\bfseries}\renewcommand{\cftafterloftitle}{\hfill}
\renewcommand{\cftlottitlefont}{\hfill\Large\bfseries}\renewcommand{\cftafterlottitle}{\hfill}
\renewcommand{\contentsname}{CONTENTS}
\renewcommand{\listfigurename}{LIST OF FIGURES}
\renewcommand{\listtablename}{LIST OF TABLES}
\renewcommand{\bibname}{REFERENCES}
\setlength{\cftbeforechapskip}{2pt}
\renewcommand{\cftchapfont}{\bfseries}\renewcommand{\cftchappagefont}{\bfseries}
\setlength{\cftfignumwidth}{2.6em}\setlength{\cfttabnumwidth}{2.6em}
\fancypagestyle{mainpages}{\fancyhf{}\renewcommand{\headrulewidth}{0pt}\fancyfoot[R]{Page $|$ \thepage}}
\fancypagestyle{frontpages}{\fancyhf{}\renewcommand{\headrulewidth}{0pt}\fancyfoot[C]{\thepage}}
\fancypagestyle{plain}{\fancyhf{}\renewcommand{\headrulewidth}{0pt}\fancyfoot[C]{\thepage}}
\newcommand{\frontheading}[1]{\begin{center}{\Large\bfseries #1}\end{center}\vspace{6pt}\addcontentsline{toc}{chapter}{#1}}
\newcommand{\sigline}[1]{\rule{#1}{0.4pt}}
\begin{document}
\pagenumbering{roman}
\pagestyle{frontpages}
""")
    # ---- title page
    w(r"""\pagecolor{titlegreen}
\begin{titlepage}\centering\setstretch{1.05}
\vspace*{-0.8cm}
{\large\bfseries """ + tex_inline(m["title"]) + r"""\par}\vspace{4pt}
{\normalsize\bfseries """ + tex_inline(m["subtitle"].upper()) + r"""\par}\vspace{14pt}
{\large A Minor Project Report\par}\vspace{12pt}
Submitted in partial fulfillment of the requirements for the award of the Degree of\par\vspace{12pt}
{\large\bfseries """ + m["degree"] + r"""\par}\vspace{6pt}
IN\par\vspace{6pt}
{\bfseries """ + m["branch"] + r"""\par """ + m["spec"] + r"""\par}\vspace{14pt}
BY\par\vspace{6pt}
{\large\bfseries """ + m["student"] + r"""\par}{\bfseries (""" + m["roll"] + r""")\par}\vspace{14pt}
Under the guidance of\par{\bfseries """ + m["guide"] + r"""\par}\vspace{12pt}
\includegraphics[width=3.4cm]{bit_logo.png}\par\vspace{14pt}
{\bfseries """ + m["dept"] + r"""\par """ + m["inst"] + r"""\par """ + tex_inline(m["campus"]) + r"""\par}\vspace{6pt}
""" + m["semester"] + r"""\par
\end{titlepage}
\nopagecolor
\setcounter{page}{2}
""")
    title_q = "``" + tex_inline(m["title_case"]) + ": " + tex_inline(m["subtitle"]) + "''"
    w(r"""\frontheading{APPROVAL OF THE GUIDE}
\setstretch{1.6}
Recommended that the project entitled """ + title_q + " presented by " + m["student_case"] + " (" + m["roll"] +
      r""") under my supervision and guidance be accepted as fulfilling this part of the requirements for the award of the Degree of Bachelor of Technology in """ + tex_inline(m["branch_case"]) +
      r""". To the best of my knowledge, the content of this project did not form a basis for the award of any previous degree to anyone else.
\setstretch{1.3}\vspace{2cm}

Date: \sigline{3cm}\hfill\sigline{5cm}\\
\mbox{}\hfill """ + m["guide"] + r"""\\
\mbox{}\hfill Designation: \sigline{3cm}

\begin{center}Department of Computer Science and Engineering\\Birla Institute of Technology, Mesra, Off-Campus Patna\end{center}
\vspace{2.2cm}
\begin{center}\sigline{5cm}\\Head of the Department\\ """ + m["hod"] + r"""\\Department of Computer Science and Engineering\\Birla Institute of Technology, Mesra, Off-Campus Patna\end{center}
\newpage
\frontheading{DECLARATION CERTIFICATE}
I certify that
\begin{enumerate}[label=(\alph*),leftmargin=2.2em]
\item the work contained in this project is original and has been done by myself under the general supervision of my supervisor;
\item the work has not been submitted to any other Institute for any other degree or diploma;
\item I have followed the guidelines provided by the Institute in writing the report;
\item I have conformed to the norms and guidelines given in the Ethical Code of Conduct of the Institute;
\item whenever I have used materials (data, theoretical analysis, figures and text) from other sources, I have given due credit to them by citing them in the text of the report and giving their details in the references; and
\item whenever I have quoted written materials from other sources, I have put them under quotation marks and given due credit to the sources by citing them and giving the required details in the references.
\end{enumerate}
\vspace{2.5cm}
\mbox{}\hfill\sigline{5cm}\\
\mbox{}\hfill """ + m["student_case"] + r"""\\
\mbox{}\hfill (""" + m["roll"] + r""")

Date: \sigline{3cm}
\newpage
\frontheading{CERTIFICATE OF APPROVAL}
\setstretch{1.6}
This is to certify that the work embodied in this project entitled """ + title_q + ", carried out by " + m["student_case"] + " (" + m["roll"] +
      r"""), has been approved for the degree of Bachelor of Technology in """ + tex_inline(m["branch_case"]) +
      r""" of Birla Institute of Technology, Mesra, Ranchi.
\setstretch{1.3}\vspace{1.4cm}

Date: \sigline{4cm}

Place: \sigline{4cm}
\vspace{2cm}

\begin{tabular}{@{}p{7cm}p{7cm}@{}}
\sigline{5cm} & \sigline{5cm}\\
Internal Examiner & External Examiner\\
Name: \sigline{3.8cm} & Name: \sigline{3.8cm}
\end{tabular}
\vspace{2cm}
\begin{center}\sigline{5.5cm}\\Chairman, Head of the Department\\ """ + m["hod"] + r"""\\Department of Computer Science and Engineering\end{center}
\newpage
\frontheading{ABSTRACT}
""")
    for p in D["abstract"]:
        w(tex_inline(p) + "\n\n")
    w(r"\textbf{Keywords:} " + tex_inline(D["keywords"]) + "\n\\newpage\n")
    w(r"""\frontheading{ACKNOWLEDGEMENT}
This project owes a great deal to my supervisor, """ + m["guide"] + r""". The guidance I received pushed me to define exactly what the benchmark should measure, and the patience shown to me allowed the design to be rebuilt whenever the evidence demanded it.

My thanks go to """ + m["hod"] + r""", Head of the Department of Computer Science and Engineering at Birla Institute of Technology, Mesra, Off-Campus Patna, and to the teachers of the department, whose courses gave me the background this work required.

The study would not have been possible without the people who recorded and released the data: the Brainclinics team behind TDBRAIN, and the authors of the Mumtaz recordings. I am equally indebted to the volunteers who maintain the open-source packages on which every experiment here runs.

I am especially grateful to my friend Arpit, whose constant support, encouragement and willingness to discuss ideas helped me through the difficult stages of this project.

Last, and most of all, I thank my family and friends, who were patient with the long evenings this project took and stood by me throughout.
\vspace{1.5cm}

Date: \sigline{3cm}\hfill """ + m["student_case"] + r"""\\
\mbox{}\hfill (""" + m["roll"] + r""")
\newpage
\tableofcontents
\newpage
\addcontentsline{toc}{chapter}{LIST OF FIGURES}
\listoffigures
\newpage
\addcontentsline{toc}{chapter}{LIST OF TABLES}
\listoftables
\newpage
\pagenumbering{arabic}
\pagestyle{mainpages}
\fancypagestyle{plain}{\fancyhf{}\renewcommand{\headrulewidth}{0pt}\fancyfoot[R]{Page $|$ \thepage}}
""")
    in_list = None
    for b in BLOCKS:
        k = b[0]
        if k == "chapter":
            w("\n\\chapter{" + tex_inline(b[1]) + "}\n")
        elif k == "section":
            w("\n\\section{" + tex_inline(b[1]) + "}\n")
        elif k == "p":
            w(tex_inline(b[1]) + "\n\n")
        elif k == "fig":
            key, cap, width = b[1], b[2], b[3]
            w("\\begin{figure}[!htbp]\\centering\n\\includegraphics[width=" + f"{width:.2f}" +
              "\\textwidth]{" + key + ".png}\n\\caption[" + tex_inline(short(cap)) + "]{" + tex_inline(cap) + "}\\label{fig:" + key + "}\n\\end{figure}\n")
        elif k == "table":
            key, cap, header, rows, spec = b[1:6]
            w("\\begin{table}[!htbp]\\centering\\small\\setstretch{1.1}\n\\caption{" + tex_inline(cap) + "}\\label{tab:" + key +
              "}\n\\begin{tabularx}{\\textwidth}{" + tex_colspec(spec) + "}\n\\toprule\n")
            w(" & ".join(tex_inline(h) for h in header) + "\\\\\n\\midrule\n")
            for r in rows:
                w(" & ".join(tex_inline(str(x)) for x in r) + "\\\\\n")
            w("\\bottomrule\n\\end{tabularx}\n\\end{table}\n")
        elif k == "list":
            env = "enumerate" if b[1] else "itemize"
            w("\\begin{" + env + "}[leftmargin=1.6em,itemsep=2pt,topsep=2pt]\n")
            for it in b[2]:
                w("\\item " + tex_inline(it) + "\n")
            w("\\end{" + env + "}\n")
        elif k == "deflist":
            w("\\begin{description}[leftmargin=3.2em,labelwidth=2.6em,itemsep=2pt]\n")
            for lab, txt in b[1]:
                w("\\item[" + tex_inline(lab) + "] " + tex_inline(txt) + "\n")
            w("\\end{description}\n")
        elif k == "eq":
            w("\\begin{equation}\n" + b[1] + "\n\\end{equation}\n")
    w("\n\\clearpage\\phantomsection\\addcontentsline{toc}{chapter}{REFERENCES}\n"
      "\\begin{thebibliography}{99}\\small\\setstretch{1.1}\n")
    for key, _ in sorted(CIT.items(), key=lambda kv: kv[1]):
        w("\\bibitem{" + key + "} " + tex_inline(REFS[key]) + "\n")
    w("\\end{thebibliography}\n\\end{document}\n")
    return "".join(T)


# ------------------------------------------------------------------ Word
def render_docx(path: Path):
    from docx import Document
    from docx.enum.section import WD_SECTION
    from docx.enum.table import WD_TABLE_ALIGNMENT
    from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Cm, Pt, RGBColor

    doc = Document()
    sec0 = doc.sections[0]
    sec0.page_height, sec0.page_width = Cm(29.7), Cm(21.0)
    sec0.left_margin, sec0.right_margin, sec0.top_margin, sec0.bottom_margin = Cm(2.8), Cm(2.5), Cm(2.5), Cm(2.5)
    TEXT_W = 21.0 - 2.8 - 2.5
    st = doc.styles["Normal"]
    st.font.name = "Times New Roman"
    st.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    st.font.size = Pt(12)
    st.paragraph_format.line_spacing = 1.3
    st.paragraph_format.space_after = Pt(6)
    for name, size in (("Heading 1", 18), ("Heading 2", 12), ("Title", 14)):
        h = doc.styles[name]
        h.font.name = "Times New Roman"
        h.element.rPr.rFonts.set(qn("w:asciiTheme"), "") if h.element.rPr is not None and h.element.rPr.rFonts is not None else None
        h.font.size = Pt(size)
        h.font.bold = True
        h.font.color.rgb = RGBColor(0, 0, 0)
        h.font.italic = False
        rpr = h.element.get_or_add_rPr()
        rf = rpr.find(qn("w:rFonts"))
        if rf is None:
            rf = OxmlElement("w:rFonts")
            rpr.append(rf)
        for a in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
            rf.set(qn(a), "Times New Roman")
        for a in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):
            if rf.get(qn(a)) is not None:
                del rf.attrib[qn(a)]
    doc.styles["Heading 1"].paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.styles["Heading 1"].paragraph_format.space_after = Pt(18)
    doc.styles["Heading 2"].paragraph_format.space_before = Pt(12)
    doc.styles["Heading 2"].paragraph_format.space_after = Pt(4)
    from docx.enum.style import WD_STYLE_TYPE

    for nm in ("Figure Caption", "Table Caption"):
        s = doc.styles.add_style(nm, WD_STYLE_TYPE.PARAGRAPH)
        s.base_style = doc.styles["Normal"]
        s.font.size = Pt(10.5)
        s.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
        s.paragraph_format.space_after = Pt(8)
        s.paragraph_format.line_spacing = 1.1

    def add_field(par, instr, placeholder=""):
        r = par.add_run()
        f1 = OxmlElement("w:fldChar"); f1.set(qn("w:fldCharType"), "begin"); r._r.append(f1)
        r2 = par.add_run()
        it = OxmlElement("w:instrText"); it.set(qn("xml:space"), "preserve"); it.text = instr; r2._r.append(it)
        r3 = par.add_run()
        f2 = OxmlElement("w:fldChar"); f2.set(qn("w:fldCharType"), "separate"); r3._r.append(f2)
        r4 = par.add_run(placeholder)
        r5 = par.add_run()
        f3 = OxmlElement("w:fldChar"); f3.set(qn("w:fldCharType"), "end"); r5._r.append(f3)
        return r4

    def inline(par, text, size=None, bold_all=False):
        text = re.sub(r"\[@([A-Za-z0-9_]+)\]", lambda m_: f"[{CIT[m_.group(1)]}]", text)
        text = re.sub(r"⟦fig:([A-Za-z0-9_]+)⟧", lambda m_: f"Figure {FIGNUM[m_.group(1)]}", text)
        text = re.sub(r"⟦tab:([A-Za-z0-9_]+)⟧", lambda m_: f"Table {TABNUM[m_.group(1)]}", text)
        for tok in re.split(r"(\*\*.+?\*\*|(?<!\w)\*.+?\*)", text):
            if not tok:
                continue
            if tok.startswith("**"):
                r = par.add_run(tok[2:-2]); r.bold = True
            elif tok.startswith("*") and tok.endswith("*") and len(tok) > 2:
                r = par.add_run(tok[1:-1]); r.italic = True
            else:
                r = par.add_run(tok)
            if bold_all:
                r.bold = True
            if size:
                r.font.size = Pt(size)
        return par

    def para(text="", align=None, size=None, bold=False, space_after=None, spacing=None):
        p = doc.add_paragraph()
        if text:
            inline(p, text, size, bold)
        if align:
            p.alignment = {"c": WD_ALIGN_PARAGRAPH.CENTER, "r": WD_ALIGN_PARAGRAPH.RIGHT,
                           "j": WD_ALIGN_PARAGRAPH.JUSTIFY}[align]
        if space_after is not None:
            p.paragraph_format.space_after = Pt(space_after)
        if spacing:
            p.paragraph_format.line_spacing = spacing
        return p

    def page_break():
        doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)

    def front_heading(t):
        para(t, "c", 14, True, 12)

    m = META
    # ---- title page
    for txt, sz, b, sa in [(m["title"], 14, True, 2), (m["subtitle"].upper(), 12, True, 14),
                           ("A Minor Project Report", 13, False, 10),
                           ("Submitted in partial fulfillment of the requirements for the award of the Degree of", 11, False, 10),
                           (m["degree"], 14, True, 4), ("IN", 11, False, 4), (m["branch"], 11, True, 0),
                           (m["spec"], 11, True, 12), ("BY", 11, False, 4), (m["student"], 14, True, 0),
                           (f"({m['roll']})", 11, True, 12), ("Under the guidance of", 11, False, 0),
                           (m["guide"], 11, True, 10)]:
        para(txt, "c", sz, b, sa, 1.15)
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(str(HERE / "figures" / "bit_logo.png"), width=Cm(3.6))
    for txt, sz, b, sa in [(m["dept"], 11, True, 0), (m["inst"], 11, True, 0), (m["campus"], 11, True, 6),
                           (m["semester"], 11, False, 0)]:
        para(txt, "c", sz, b, sa, 1.15)
    page_break()
    title_q = f"“{m['title_case']}: {m['subtitle']}”"
    front_heading("APPROVAL OF THE GUIDE")
    para(f"Recommended that the project entitled {title_q} presented by {m['student_case']} ({m['roll']}) under my "
         f"supervision and guidance be accepted as fulfilling this part of the requirements for the award of the Degree "
         f"of Bachelor of Technology in {m['branch_case']}. To the best of my knowledge, the content of this project did "
         f"not form a basis for the award of any previous degree to anyone else.", "j", spacing=1.6)
    for _ in range(3):
        para()
    para("Date: ____________\t\t\t\t\t______________________")
    para(m["guide"], "r", space_after=0)
    para("Designation: ____________", "r")
    para("Department of Computer Science and Engineering", "c", space_after=0)
    para("Birla Institute of Technology, Mesra, Off-Campus Patna", "c")
    for _ in range(3):
        para()
    for t in ("______________________", "Head of the Department", m["hod"], "Department of Computer Science and Engineering",
              "Birla Institute of Technology, Mesra, Off-Campus Patna"):
        para(t, "c", space_after=0)
    page_break()
    front_heading("DECLARATION CERTIFICATE")
    para("I certify that")
    for i, t in enumerate(["the work contained in this project is original and has been done by myself under the general "
                           "supervision of my supervisor;",
                           "the work has not been submitted to any other Institute for any other degree or diploma;",
                           "I have followed the guidelines provided by the Institute in writing the report;",
                           "I have conformed to the norms and guidelines given in the Ethical Code of Conduct of the Institute;",
                           "whenever I have used materials (data, theoretical analysis, figures and text) from other sources, I "
                           "have given due credit to them by citing them in the text of the report and giving their details in "
                           "the references; and",
                           "whenever I have quoted written materials from other sources, I have put them under quotation marks "
                           "and given due credit to the sources by citing them and giving the required details in the "
                           "references."]):
        p = para(f"({'abcdef'[i]})\t{t}", "j")
        p.paragraph_format.left_indent = Cm(1.2)
        p.paragraph_format.first_line_indent = Cm(-0.8)
    for _ in range(3):
        para()
    para("______________________", "r", space_after=0)
    para(m["student_case"], "r", space_after=0)
    para(f"({m['roll']})", "r")
    para("Date: ____________")
    page_break()
    front_heading("CERTIFICATE OF APPROVAL")
    para(f"This is to certify that the work embodied in this project entitled {title_q}, carried out by "
         f"{m['student_case']} ({m['roll']}), has been approved for the degree of Bachelor of Technology in "
         f"{m['branch_case']} of Birla Institute of Technology, Mesra, Ranchi.", "j", spacing=1.6)
    para()
    para("Date: ________________")
    para("Place: ________________")
    for _ in range(2):
        para()
    para("______________________\t\t\t______________________", space_after=0)
    para("Internal Examiner\t\t\t\tExternal Examiner", space_after=0)
    para("Name: ________________\t\t\tName: ________________")
    for _ in range(3):
        para()
    for t in ("______________________", "Chairman, Head of the Department", m["hod"],
              "Department of Computer Science and Engineering"):
        para(t, "c", space_after=0)
    page_break()
    front_heading("ABSTRACT")
    for p_ in D["abstract"]:
        para(p_, "j")
    para(f"**Keywords:** {D['keywords']}", "j")
    page_break()
    front_heading("ACKNOWLEDGEMENT")
    for t in [f"This project owes a great deal to my supervisor, {m['guide']}. The guidance I received pushed me to "
              "define exactly what the benchmark should measure, and the patience shown to me allowed the design to be "
              "rebuilt whenever the evidence demanded it.",
              f"My thanks go to {m['hod']}, Head of the Department of Computer Science and Engineering at Birla Institute "
              "of Technology, Mesra, Off-Campus Patna, and to the teachers of the department, whose courses gave me the "
              "background this work required.",
              "The study would not have been possible without the people who recorded and released the data: the "
              "Brainclinics team behind TDBRAIN, and the authors of the Mumtaz recordings. I am equally indebted to the "
              "volunteers who maintain the open-source packages on which every experiment here runs.",
              "I am especially grateful to my friend Arpit, whose constant support, encouragement and willingness to "
              "discuss ideas helped me through the difficult stages of this project.",
              "Last, and most of all, I thank my family and friends, who were patient with the long evenings this project "
              "took and stood by me throughout."]:
        para(t, "j")
    para()
    para(f"Date: ____________\t\t\t\t\t\t{m['student_case']}", space_after=0)
    para(f"({m['roll']})", "r")
    page_break()
    # ---- contents (Word field, pre-filled so it reads correctly before updating)
    front_heading("CONTENTS")
    p = doc.add_paragraph()
    pre = []
    chn = 0
    for b in BLOCKS:
        if b[0] == "chapter":
            chn += 1
            secn = 0
            pre.append(f"{chn}  {b[1]}")
        elif b[0] == "section":
            secn += 1
            pre.append(f"      {chn}.{secn}  {b[1]}")
    pre.append("REFERENCES")
    add_field(p, 'TOC \\o "1-2" \\h \\z \\u', "\n".join(pre))
    page_break()
    front_heading("LIST OF FIGURES")
    p = doc.add_paragraph()
    add_field(p, 'TOC \\h \\z \\t "Figure Caption,1"',
              "\n".join(f"{FIGNUM[b[1]]}  {short(re.sub(r'[*]', '', b[2]))}" for b in BLOCKS if b[0] == "fig"))
    page_break()
    front_heading("LIST OF TABLES")
    p = doc.add_paragraph()
    add_field(p, 'TOC \\h \\z \\t "Table Caption,1"',
              "\n".join(f"{TABNUM[b[1]]}  {b[2]}" for b in BLOCKS if b[0] == "table"))
    # front-matter page numbers: lower roman, centred
    fsec = doc.sections[0]

    def footer_field(section, right, prefix):
        f = section.footer
        f.is_linked_to_previous = False
        p_ = f.paragraphs[0]
        p_.text = ""
        p_.alignment = WD_ALIGN_PARAGRAPH.RIGHT if right else WD_ALIGN_PARAGRAPH.CENTER
        if prefix:
            p_.add_run(prefix)
        add_field(p_, "PAGE", "1")

    def pg_fmt(section, fmt, start):
        sp = section._sectPr
        pg = sp.find(qn("w:pgNumType"))
        if pg is None:
            pg = OxmlElement("w:pgNumType"); sp.append(pg)
        pg.set(qn("w:fmt"), fmt); pg.set(qn("w:start"), str(start))

    footer_field(fsec, False, "")
    pg_fmt(fsec, "lowerRoman", 1)
    fsec.different_first_page_header_footer = True
    # ---- main matter
    msec = doc.add_section(WD_SECTION.NEW_PAGE)
    footer_field(msec, True, "Page | ")
    pg_fmt(msec, "decimal", 1)
    msec.different_first_page_header_footer = False
    chn = 0
    first = True
    for b in BLOCKS:
        k = b[0]
        if k == "chapter":
            chn += 1
            secn = 0
            if not first:
                page_break()
            first = False
            para(f"CHAPTER {chn}", "c", 14, True, 6)
            doc.add_heading(b[1].upper(), level=1)
        elif k == "section":
            secn += 1
            doc.add_heading(f"{chn}.{secn}   {b[1].upper()}", level=2)
        elif k == "p":
            para(b[1], "j")
        elif k == "fig":
            key, cap, width = b[1], b[2], b[3]
            p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.keep_with_next = True
            p.add_run().add_picture(str(HERE / "figures" / f"{key}.png"), width=Cm(TEXT_W * width))
            cp = doc.add_paragraph(style="Figure Caption")
            r = cp.add_run(f"Figure {FIGNUM[key]}. "); r.bold = True
            inline(cp, cap)
        elif k == "table":
            key, cap, header, rows, spec = b[1:6]
            cp = doc.add_paragraph(style="Table Caption")
            cp.paragraph_format.keep_with_next = True
            r = cp.add_run(f"Table {TABNUM[key]}. "); r.bold = True
            inline(cp, cap)
            t = doc.add_table(rows=1 + len(rows), cols=len(header))
            t.alignment = WD_TABLE_ALIGNMENT.CENTER
            tblPr = t._tbl.tblPr
            borders = OxmlElement("w:tblBorders")
            for edge, sz in (("top", 12), ("bottom", 12)):
                e = OxmlElement(f"w:{edge}"); e.set(qn("w:val"), "single"); e.set(qn("w:sz"), str(sz)); e.set(qn("w:color"), "000000")
                borders.append(e)
            tblPr.append(borders)
            nX = spec.count("X")
            fixed = {"l": 0.17, "r": 0.11, "c": 0.12}
            widths = []
            rest = 1.0 - sum(fixed.get(ch_, 0) for ch_ in spec if ch_ != "X")
            for ch_ in spec:
                widths.append(rest / nX if ch_ == "X" and nX else fixed.get(ch_, 0.15))
            s_ = sum(widths)
            widths = [w_ / s_ * TEXT_W for w_ in widths]
            for i, row_ in enumerate([header] + rows):
                for j, val in enumerate(row_):
                    cell = t.cell(i, j)
                    cell.width = Cm(widths[j])
                    cp = cell.paragraphs[0]
                    cp.paragraph_format.space_after = Pt(1)
                    cp.paragraph_format.line_spacing = 1.0
                    if spec[j] == "r":
                        cp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
                    inline(cp, str(val), 10, False)
                    if i == 0:
                        tcPr = cell._tc.get_or_add_tcPr()
                        tb = OxmlElement("w:tcBorders")
                        e = OxmlElement("w:bottom"); e.set(qn("w:val"), "single"); e.set(qn("w:sz"), "6"); e.set(qn("w:color"), "000000")
                        tb.append(e); tcPr.append(tb)
            para("", space_after=2)
        elif k == "list":
            for i, it in enumerate(b[2]):
                p = para(f"{i + 1}.\t{it}" if b[1] else f"•\t{it}", "j", space_after=3)
                p.paragraph_format.left_indent = Cm(0.9)
                p.paragraph_format.first_line_indent = Cm(-0.6)
        elif k == "deflist":
            for lab, txt in b[1]:
                p = para(f"**{lab}**\t{txt}", "j", space_after=3)
                p.paragraph_format.left_indent = Cm(1.4)
                p.paragraph_format.first_line_indent = Cm(-1.4)
        elif k == "eq":
            p = para("", "c")
            r = p.add_run(b[2]); r.italic = True
    page_break()
    doc.add_heading("REFERENCES", level=1)
    for key, num in sorted(CIT.items(), key=lambda kv: kv[1]):
        p = para(f"[{num}]\t{REFS[key]}", "j", 10.5, space_after=3, spacing=1.1)
        p.paragraph_format.left_indent = Cm(1.0)
        p.paragraph_format.first_line_indent = Cm(-1.0)
    # ask Word to refresh fields (contents, lists, page numbers) when the file is opened
    settings = doc.settings.element
    uf = OxmlElement("w:updateFields"); uf.set(qn("w:val"), "true"); settings.append(uf)
    doc.save(path)


def main():
    (HERE / "thesis.tex").write_text(render_tex())
    render_docx(HERE / "Thesis_Kanan_Kotwani.docx")
    print("wrote", HERE / "thesis.tex", "and", HERE / "Thesis_Kanan_Kotwani.docx")
    print(f"{len(FIGNUM)} figures, {len(TABNUM)} tables, {len(CIT)} references")


if __name__ == "__main__":
    main()
