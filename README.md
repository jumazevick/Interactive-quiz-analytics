# Sage Foundation: Tech for Food Education Team 1

## Moodle/STACK Interactive Quiz Analytics

## Overview

This project is a Streamlit dashboard for analyzing Moodle STACK quiz exports. It helps lecturers and administrators inspect student performance, question difficulty, response/PRT behaviour, and engagement patterns across one or more quizzes — entirely client-side, with no quiz data ever leaving the browser session.

The app is a single **Question & Quiz Analysis** page, linked from the home page, driven by one upload of a Moodle **Responses** export (optionally merged with a **Grades with question breakdown** export for more accurate per-question scoring):

- **Question Analysis** (top, scoped to whichever quiz is selected) — question summary, difficulty/discrimination, per-question text with a right-answer/error drill-down, response distribution and PRT pass rates, a student-by-question performance matrix, and a consolidated metrics table.
- **Quiz Analysis** (bottom, combined across every uploaded quiz you choose to include) — merged attempt list, summary stats, grade distribution, engagement over time, attempts-vs-grade correlation, and metric trends across quizzes.

Special thanks to:
- **Ernest** for the question analysis research, technical setup and development, and implementation.
- **Sage** for the technical setup.
- **Otis** for the question analysis research.

## Features

- **Flexible uploads**: supports `.csv`, `.xls`, and `.xlsx` exports; upload one or more quiz files at once.
- **Persistent upload**: an uploaded file survives navigating to the home page and back, with an explicit "Clear / Reset All Uploaded Files" button when you want a clean slate.
- **Best-attempt handling**: automatically separates "all attempts" from "best attempt per student" for participation vs. performance metrics.
- **Anonymization**: an "Anonymize Student Data" toggle replaces real names/emails with stable per-student pseudonyms everywhere — tables, charts, and PDF exports.
- **LaTeX-aware rendering**: cleans up raw STACK/Moodle LaTeX and Maxima expression syntax (`\(...\)`, `%pi`, `sqrt(...)`, etc.) into properly rendered math wherever question text, submitted responses, and right answers are shown.
- **Interactive Plotly charts**: every chart (box plots, heatmaps, scatter plots, line/density charts) is rendered with Plotly for a consistent look throughout.
- **Organized sidebar**: Question Analysis and Quiz Analysis controls are grouped into their own sidebar sections, including a multiselect to choose which uploaded quizzes feed the Quiz Analysis aggregation.
- **PDF export**: a single "Download PDF Report" button bundles the visible tables and a rasterized image of every visible chart into one PDF, with its own scope controls (include/exclude the Quiz Analysis summary, and pick which quiz(zes) get a full Question Analysis breakdown).
- **Data validation**: flags mismatches between calculated per-question scores and Moodle's own recorded grade, and other basic sanity checks, directly in the UI.

## Project Structure

```
Home.py                              # Landing page: overview, nav button, walkthrough video, acknowledgements
streamlit_app.py                     # Thin entry point (re-exports Home.py) for Streamlit Cloud
pages/
  Question_and_Quiz_Analysis.py      # The single unified analysis page
analytics/                           # Parsing, metrics, PDF export, and other shared logic
tests/                               # Pytest suite for the analytics/parsing pipeline
```

## Technical Description

### Tools and Libraries Used

- **Streamlit**: web framework for the interactive dashboard.
- **Pandas**: data loading, cleaning, and aggregation.
- **Plotly** (+ **Kaleido**): interactive charts on-screen and their rasterized PNG versions embedded in PDF exports.
- **SciPy**: gaussian KDE for the engagement/density chart.
- **Matplotlib**: date-axis utilities only (no chart rendering).
- **ReportLab**: PDF report generation.
- **OpenPyXL / xlrd**: reading `.xlsx` and `.xls` files.

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
