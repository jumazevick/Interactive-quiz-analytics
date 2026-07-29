# Sage Foundation: Tech for Good Education Team 1

## Moodle/STACK Interactive Quiz Analytics

## Overview

This project is a Streamlit dashboard for analyzing Moodle STACK quiz exports. It helps lecturers and administrators inspect student performance, question difficulty, response/PRT behaviour, and engagement patterns across one or more quizzes — entirely client-side, with no quiz data ever leaving the browser session.

The app has two analysis pages, both linked from the home page and both driven by the same upload of a Moodle **Responses** export (optionally merged with a **Grades with question breakdown** export for more accurate per-question scoring). An upload made on either page is picked up by the other automatically — you never upload the same file twice.

**Question & Quiz Analysis** — what the class scored:

- **Question Analysis** (top, scoped to whichever quiz is selected) — question summary, difficulty/discrimination, per-question text with a right-answer/error drill-down, response distribution and PRT pass rates, a student-by-question performance matrix, and a consolidated metrics table.
- **Quiz Analysis** (bottom, combined across every uploaded quiz you choose to include) — merged attempt list, summary stats, grade distribution, engagement over time, attempts-vs-grade correlation, and metric trends across quizzes.

**Solution Process Visualization** — *how* students got there, across their retakes of one question:

- **Transition graphs** — a directed graph of one student's movement between PRT-classified answer types (unclassified wrong → a specific classified wrong answer → correct), plus the class-wide aggregate where edge thickness and color show how many students made each transition, with in-degree / out-degree / degree centrality per node.
- **3D distance charts** — every student's trajectory toward the correct answer over their attempts, measured two ways: PRT-node distance, and Tree Edit Distance between the submitted and correct CAS expressions.

Special thanks to:
- **Juma** for the original hackathon idea and implementation, quiz and question analysis research, and advising.
- **Ernest** for the question analysis research, technical setup and development, and implementation.
- **Sage** for the technical setup.
- **Otis** for the question analysis research.

The Solution Process Visualization page implements methods published by their authors, with thanks and full credit to:
- **Asahi Kurihara** and **Yasuyuki Nakamura** (Nagoya University), *Network Analysis of Solution Processes in Math Online Tests*, Companion Proceedings of the 15th International Conference on Learning Analytics & Knowledge (LAK25), 2025, pp. 257–259 — the answer-transition directed graphs and their network features.
- **Tomoki Takada** and **Yasuyuki Nakamura** (Nagoya University) and **Saburo Higuchi** (Ryukoku University), [*Visualization of Solution Processes to Reach the Correct Answer in Online Math Tests*](https://doi.org/10.5281/zenodo.15870221), Proceedings of the 18th International Conference on Educational Data Mining (EDM 2025), Palermo, Italy, pp. 635–639 — the 3D PRT-distance and Tree Edit Distance visualizations.

## Features

- **Flexible uploads**: supports `.csv`, `.xls`, and `.xlsx` exports; upload one or more quiz files at once.
- **Export-format flexibility**: recognizes common alternate Moodle column headers (e.g. `Username`/`Status`/`Duration` in place of `Email address`/`State`/`Time taken`) and normalizes them automatically, and parses STACK PRT (Potential Response Tree) fields by their `!`/`# = fraction` value shape rather than an assumed `prt1`/`prt2` name — so quizzes where a teacher renamed PRTs to something else (e.g. `Result`/`Result2`) still score and analyze correctly instead of silently reading as zero.
- **Persistent upload**: an uploaded file survives navigating to the home page and back, with an explicit "Clear / Reset All Uploaded Files" button when you want a clean slate.
- **Best-attempt handling**: automatically separates "all attempts" from "best attempt per student" for participation vs. performance metrics.
- **Anonymization**: an "Anonymize Student Data" toggle (on by default) replaces real names/emails with stable per-student pseudonyms everywhere — tables, charts, and PDF exports. If a Moodle export was already anonymized at the source (blank name/email columns), the app still assigns each row a stable, unique placeholder identity instead of merging every student into one.
- **LaTeX-aware rendering**: cleans up raw STACK/Moodle LaTeX and converts Maxima CAS expression syntax — `\(...\)`, `%pi`/`%e`/`%i`, `sqrt(...)`, `^(...)`, `abs(...)`, `floor(...)`/`ceiling(...)`, `nthroot(...)`, `matrix(...)` (rendered as a bracketed matrix/vector), boolean/constant keywords (`true`, `false`, `inf`, ...), comparison operators (`#`, `<=`, `>=`), and any other/unrecognized Maxima function call (via a generic operator-name fallback) — into properly rendered math wherever question text, submitted responses, and right answers are shown. The conversion is written to work identically in both places it's rendered: Streamlit/KaTeX on-screen and Matplotlib's `mathtext` in the PDF export, which don't support the same LaTeX subset (notably, `mathtext` has no support for `\begin{...}` environments). It also repairs the mojibake (UTF-8 text mis-decoded as Latin-1) that some Moodle export pipelines introduce into special characters like `π` and `·`.
- **Interactive Plotly charts**: every chart (box plots, heatmaps, scatter plots, line/density charts) is rendered with Plotly for a consistent look throughout.
- **Consistent, readable labels**: table headers and multiselect/filter options display as "Average Marks" / "Student ID" rather than the raw internal `average_marks` / `student_id` keys, everywhere the app surfaces them — on-screen and in the PDF.
- **Organized sidebar**: Question Analysis and Quiz Analysis controls are grouped into their own sidebar sections, each with a "Select All" / "Deselect All" button pair, plus a multiselect to choose which uploaded quizzes feed the Quiz Analysis aggregation.
- **Polished, theme-aware UI**: a monotone dark/light design system (switchable via System/Light/Dark in Streamlit's own "⋮" menu) with a widened sidebar, an always-visible collapse control, and a single unified scroll region.
- **Colorblind-friendly mode**: a "Colorblind Mode" toggle next to the Question & Quiz Analysis page title swaps every chart — bar, box, scatter, and line charts plus the PRT pass-rate heatmap — to a red-green colorblind-safe palette (an Okabe-Ito-derived qualitative palette and a blue/yellow/vermillion scale in place of the default red/yellow/green).
- **PDF export**: a single "Download PDF Report" button bundles the visible tables and a rasterized image of every visible chart into one PDF, with its own scope controls — include/exclude the Quiz Analysis summary or Question Analysis breakdown wholesale, or pick individual sections within each, plus which quiz(zes) get a full Question Analysis breakdown. Charts are rasterized in one batched pass rather than one browser launch per chart, so a report with a dozen charts renders in a few seconds instead of the better part of a minute.
  - **Auto-generated Table of Contents**: a first page listing every section title with its actual page number (plus native PDF outline/bookmarks for quick navigation in most PDF viewers), automatically included once a report has more than a couple of sections.
  - **Real math typesetting, not raw LaTeX**: STACK answer expressions and question/right-answer text are rasterized through Matplotlib's `mathtext` renderer directly into the PDF's tables, so fractions, radicals, superscripts, and Greek letters render as actual math instead of literal `$...$`/backslash-command text. Sizing is shared across each table column (based on that column's typical entry width, not its single longest outlier) so answers read at a consistent, legible size.
  - **Multi-quiz breakdown ordering**: when several quizzes are selected for the Question Analysis breakdown, the PDF gives each quiz its own complete run of sections 1–6 (quiz A's summary through metrics, then quiz B's, and so on) before moving on to the combined Quiz Analysis sections — rather than interleaving section 1 for every quiz, then section 2 for every quiz.
- **Data validation**: flags mismatches between calculated per-question scores and Moodle's own recorded grade, and other basic sanity checks, directly in the UI.
- **Solution process visualization** (its own page — see the two papers credited above):
  - **Per-student and class-wide transition graphs**: pick a question and click a student in the roster to see their answer transitions as a directed graph — node `c` is full marks, a numbered node is the classified wrong answer that part's PRT matched, and node `0` is an unclassified wrong answer. The class-wide aggregate superimposes every student's transitions, scaling each edge's thickness *and* color (green = few, red = many) by how many students made it, and reports in-degree / out-degree / degree centrality per node.
  - **Multi-part questions**: a STACK question split into several parts (`prt1`, `prt2`, ...) is scored and classified independently per part, so every graph and chart on the page is scoped to one selected part rather than silently reporting on part 1 alone.
  - **3D solution-process charts**: one polyline per student over Attempt × Students × distance-from-correct, with each point colored by its own distance (white at 0, neon red at 1, running up through orange/yellow/green/blue to black at the largest distance seen) so a trajectory that closes in on the answer visibly shifts color along its length. Students are ordered along the Students axis by their first attempt's distance, then their second within each of those groups, and so on.
  - **Two distance measures**: *PRT distance*, a generalization of the teacher-authored distance table in Takada et al., and *Tree Edit Distance*, the Zhang-Shasha edit distance between the expression tree of the submitted CAS answer and that of the correct one — which separates answers the PRT lumps together into one unclassified bucket.
  - **Fixed scene backdrop**: the 3D charts stay fully rotatable, but their walls are pinned to three fixed planes instead of Plotly's default panes, which jump from one side of the box to the other mid-drag. Both charts also open from the same fixed viewpoint every render.

## Project Structure

```
Home.py                              # Landing page: overview, nav button, walkthrough video, acknowledgements
streamlit_app.py                     # Thin entry point (re-exports Home.py) for Streamlit Cloud
.streamlit/config.toml               # Theme (colors, font, radius) — Streamlit only auto-discovers config here
packages.txt                         # apt packages for Streamlit Community Cloud (chromium, for chart export — see below)
pages/
  Question_and_Quiz_Analysis.py      # Question + quiz performance analysis
  Solution_Process_Visualization.py  # Answer-transition graphs and 3D distance charts
analytics/                           # Parsing, metrics, PDF export, and other shared logic
  parser.py                          # Moodle/STACK export parsing (responses, PRT traces, attempt pools)
  data_loader.py                     # Shared upload -> (quiz metadata, response DataFrame), used by both pages
  prt_transitions.py                 # Node classification, transition graphs, network features
  solution_distance.py               # PRT/TED distance series and the 3D scene
  expression_tree.py                 # Maxima CAS string -> ordered labelled expression tree
  tree_edit_distance.py              # Zhang-Shasha tree edit distance
  pdf_export.py                      # Chart rasterization + ReportLab report generation
tests/                               # Pytest suite for the analytics/parsing pipeline
```

## Technical Description

### Tools and Libraries Used

- **Streamlit**: web framework for the interactive dashboard.
- **Pandas**: data loading, cleaning, and aggregation.
- **Plotly** (+ **Kaleido**): interactive charts on-screen and their rasterized PNG versions embedded in PDF exports. Kaleido 1.x renders via a real headless Chrome (rather than a bundled one) and pays a multi-second startup cost per rasterization call, so the PDF export batches every chart in the report through a single `plotly.io.write_images` call instead of rasterizing one at a time. It also self-heals if no system Chrome is found (e.g. downloading a private copy via `kaleido.get_chrome_sync()`) and retries once — see the deployment note below.
- **SciPy**: gaussian KDE for the engagement/density chart.
- **Matplotlib**: date-axis utilities, and (via its `mathtext` renderer) rasterizing STACK LaTeX/Maxima math expressions directly into the PDF's tables so they typeset as real math instead of literal `$...$` text — no chart rendering.
- **ReportLab**: PDF report generation, including an auto-populated Table of Contents (`reportlab.platypus.tableofcontents`) built over two layout passes (`multiBuild`) so section page numbers resolve correctly.
- **OpenPyXL / xlrd**: reading `.xlsx` and `.xls` files.

No graph or tree-distance library is used. The transition graphs on the Solution Process Visualization page only ever hold a handful of nodes, so their centrality measures are computed directly from the edge counts and the nodes are laid out on a deterministic circle — no `networkx`. Likewise the Maxima expression parser and the Zhang-Shasha tree edit distance are implemented in `analytics/expression_tree.py` and `analytics/tree_edit_distance.py` rather than pulled in from `zss`. Both keep the dependency set unchanged.

### Deploying to Streamlit Community Cloud

Chart export (Kaleido) needs a Chrome/Chromium binary on the host. Streamlit Community Cloud's base container doesn't ship one, so this repo includes a `packages.txt` with `chromium` — Streamlit Cloud installs everything listed there via `apt-get` before your app starts, which gives Kaleido both a real browser and the OS-level shared libraries (`libnss3`, `libgbm1`, etc.) it needs to run headlessly. If you fork this repo and charts silently stop appearing in the PDF export on your own Cloud deployment, check that `packages.txt` made it into your fork and that the deployment actually redeployed after it was added.

## Usage

#### 1. Clone the repo

```
git clone https://github.com/jumazevick/Interactive-quiz-analytics.git
cd Interactive-quiz-analytics
```

#### 2. Environment setup and package install

Requires **Python 3.10, 3.11, or 3.12** (see the note below if you only have 3.13).

```
conda create -n hackathon-education python=3.10.8
conda activate hackathon-education
pip install poetry
poetry install
```

No conda? Any virtualenv tool works the same way, as long as it's created with a supported Python version, e.g.:

```
python3.10 -m venv .venv
source .venv/bin/activate
pip install poetry
poetry install
```

> **Python 3.13 note:** `poetry install` can fail while building `rpds-py` from source, with an error like `the configured Python interpreter version (3.13) is newer than PyO3's maximum supported version (3.12)`. This is a real incompatibility in a pinned transitive dependency (via `jsonschema`/`referencing`), not a problem with this project's own code — use Python 3.10–3.12 as shown above to avoid it entirely.

#### 3. Run the Streamlit app

From the repo root, with the environment from step 2 active:

```
streamlit run Home.py
```

#### 4. Run the tests

```
pytest
```

For Streamlit Cloud deployment, set the main file path to `Home.py` (or `streamlit_app.py`, which just re-exports it).
