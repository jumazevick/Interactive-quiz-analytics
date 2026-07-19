# Sage Foundation: tech for good education team 1

## Moodle STACK Analytics Hub

## Overview

This project is a Streamlit dashboard for analyzing Moodle STACK quiz attempt exports. It helps lecturers and administrators inspect student performance, quiz difficulty, engagement patterns, and attempt behavior across multiple quizzes.

## Features

- **Upload quiz attempt data**: Supports `.csv`, `.xls`, and `.xlsx` exports from Moodle.
- **Merge multiple quizzes**: Combine several quiz files into one analysis view.
- **Normalize grades**: Converts quiz grades to a common 0-10 scale.
- **Parse attempt metadata**: Converts start/end timestamps and time taken into analysis-friendly values.
- **Interactive analytics**: View summary tables, grade distributions, engagement density plots, scatter plots, and line charts.
- **Quiz-level filtering**: Select one or more quiz IDs to focus the analysis.

## Technical Description

### Tools and Libraries Used

- **Streamlit**: Web framework used for the interactive dashboard.
- **Pandas**: Data loading, cleaning, and aggregation.
- **Seaborn**: Statistical plotting.
- **Matplotlib**: Base plotting for charts.
- **Altair**: Used for interactive line charts in the quiz metrics view.
- **OpenPyXL / xlrd**: Used for reading `.xlsx` and `.xls` files.

## Usage

#### 1. Conda env setup and packages install
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
streamlit run streamlit_app.py
```

If you plan to use the full quiz analytics page, make sure the environment also includes `altair` and `xlrd`, since the Streamlit code imports them.

For Streamlit Cloud deployment, set the main file path to `streamlit_app.py`.
