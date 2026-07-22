# Sage Foundation: tech for good education team 1

## Moodle/STACK Interactive Quiz Analytics

## Overview

This project is a Streamlit dashboard for analyzing Moodle STACK quiz exports. It helps lecturers and administrators inspect student performance, question difficulty, response/PRT behaviour, and engagement patterns across one or more quizzes — entirely client-side, with no quiz data ever leaving the browser session.

The app has two main sections, linked from the home page:

- **Quiz Analysis Section** — cohort-level trends from a Moodle **Grades** export (merged attempts, summary stats, grade distribution, engagement over time, attempts-vs-grades correlation, metric trends across quizzes).
- **Question Analysis Section** — per-question analytics from a Moodle **Responses** export, optionally merged with a **Grades (with question breakdown)** export for more accurate per-question scoring (question summary, difficulty/discrimination, response distribution and PRT pass rates, a student-by-question performance matrix, and a consolidated metrics table).

## Features

- **Flexible uploads**: supports `.csv`, `.xls`, and `.xlsx` exports; upload one or more files per section.
- **Shared upload memory**: a file uploaded on one section stays available if you switch to the other section, so you don't need to re-upload it.
- **Best-attempt handling**: automatically separates "all attempts" from "best attempt per student" for participation vs. performance metrics.
- **Interactive Plotly charts**: every chart (box plots, heatmaps, scatter plots, line/density charts) is rendered with Plotly for a consistent look across both sections.
- **PDF export**: each section has a "Download PDF Report" button that bundles the visible tables and a rasterized image of every visible chart into a single PDF, respecting whichever sections are currently checked in the sidebar.
- **Data validation**: flags mismatches between calculated per-question scores and Moodle's own recorded grade, and other basic sanity checks, directly in the UI.
- **Sample data**: a ready-to-use sample quiz export is downloadable from the home page if you want to try the app without your own data.

## Project Structure

```
Home.py                    # Landing page: overview, export instructions, sample data
streamlit_app.py           # Thin entry point (re-exports Home.py) for Streamlit Cloud
pages/
  Quiz_Analysis_Section.py     # Grades-export cohort analytics
  Question_Analysis_Section.py # Responses-export per-question analytics
analytics/                 # Parsing, metrics, PDF export, and other shared logic
sample_data/                # Sample Moodle export, packaged as a .zip
tests/                      # Pytest suite for the analytics/parsing pipeline
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

#### 1. Environment setup and package install

```
conda create -n hackathon-education python=3.10.8
conda activate hackathon-education
pip install poetry
cd Interactive-quiz-analytics
poetry install
```

#### 2. Run the Streamlit app

From the repo root, run:

```
streamlit run Home.py
```

#### 3. Run the tests

```
pytest
```

For Streamlit Cloud deployment, set the main file path to `Home.py` (or `streamlit_app.py`, which just re-exports it).
