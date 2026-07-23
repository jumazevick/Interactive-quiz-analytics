# Sage Foundation: Tech for Good Education Team 1

## Moodle/STACK Interactive Quiz Analytics

## Overview

This project is a Streamlit dashboard for analyzing Moodle STACK quiz exports. It helps lecturers and administrators inspect student performance, question difficulty, response/PRT behaviour, and engagement patterns across one or more quizzes — entirely client-side, with no quiz data ever leaving the browser session.

The app is a single **Question & Quiz Analysis** page, linked from the home page, driven by one upload of a Moodle **Responses** export (optionally merged with a **Grades with question breakdown** export for more accurate per-question scoring):

- **Question Analysis** (top, scoped to whichever quiz is selected) — question summary, difficulty/discrimination, per-question text with a right-answer/error drill-down, response distribution and PRT pass rates, a student-by-question performance matrix, and a consolidated metrics table.
- **Quiz Analysis** (bottom, combined across every uploaded quiz you choose to include) — merged attempt list, summary stats, grade distribution, engagement over time, attempts-vs-grade correlation, and metric trends across quizzes.

Special thanks to:
- **Juma** for the original hackathon idea and implementation, quiz and question analysis research, and advising.
- **Ernest** for the question analysis research, technical setup and development, and implementation.
- **Sage** for the technical setup.
- **Otis** for the question analysis research.

## Features

- **Flexible uploads**: supports `.csv`, `.xls`, and `.xlsx` exports; upload one or more quiz files at once.
- **Persistent upload**: an uploaded file survives navigating to the home page and back, with an explicit "Clear / Reset All Uploaded Files" button when you want a clean slate.
- **Best-attempt handling**: automatically separates "all attempts" from "best attempt per student" for participation vs. performance metrics.
- **Anonymization**: an "Anonymize Student Data" toggle (on by default) replaces real names/emails with stable per-student pseudonyms everywhere — tables, charts, and PDF exports. If a Moodle export was already anonymized at the source (blank name/email columns), the app still assigns each row a stable, unique placeholder identity instead of merging every student into one.
- **LaTeX-aware rendering**: cleans up raw STACK/Moodle LaTeX and Maxima expression syntax (`\(...\)`, `%pi`, `%e`, `sqrt(...)`, `^(...)`, etc.) into properly rendered math wherever question text, submitted responses, and right answers are shown, and repairs the mojibake (UTF-8 text mis-decoded as Latin-1) that some Moodle export pipelines introduce into special characters like `π` and `·`.
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

## Project Structure

```
Home.py                              # Landing page: overview, nav button, walkthrough video, acknowledgements
streamlit_app.py                     # Thin entry point (re-exports Home.py) for Streamlit Cloud
.streamlit/config.toml               # Theme (colors, font, radius) — Streamlit only auto-discovers config here
packages.txt                         # apt packages for Streamlit Community Cloud (chromium, for chart export — see below)
pages/
  Question_and_Quiz_Analysis.py      # The single unified analysis page
analytics/                           # Parsing, metrics, PDF export, and other shared logic
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
