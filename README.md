# Moodle/STACK Interactive Quiz Analytics

A Streamlit dashboard for analyzing Moodle **STACK** quiz exports. Upload one or more Responses reports and it reports question difficulty and discrimination, response and PRT behaviour, cohort-level grade and engagement statistics, and — for questions students may retake — how each student moved between answer types on their way to the correct answer. Everything runs client-side in your own browser session; no quiz data is sent anywhere, and the whole analysis can be exported as one PDF report.

## Sections

| Section | What it shows |
|---|---|
| **Home** | Overview, walkthrough video, acknowledgements. |
| **Question Analysis** | One quiz at a time: question summary, difficulty/discrimination, per-question text with a right-answer and error drill-down, response distribution and PRT pass rates, a student-by-question matrix, and a consolidated metrics table. |
| **↳ Solution Process Visualization** | One question and part at a time: a student's answer transitions as a directed graph, the class-wide aggregate graph with per-node network features, and two 3D charts of each student's distance from the correct answer over their attempts — by PRT-node depth and by Tree Edit Distance between the submitted and correct CAS expressions. |
| **Quiz Analysis** | Combined across every uploaded file: merged attempt list, per-quiz stats, grade distributions, engagement over time, attempts-vs-grade correlation, and metric trends. |

Uploads are shared across all four sections — upload once and every section sees the same quizzes. Each section's sidebar has checkboxes to show or hide its own modules, and all three analysis sections end with the same **PDF Report Options** panel: tick which of the three sections to include, which modules within each, and which quizzes, then generate one report (Question Analysis per quiz, then Solution Process Visualization, then Quiz Analysis).

Light is the default theme; System/Light/Dark is switchable from Streamlit's "⋮" menu, and a **Colorblind Mode** toggle on each analysis page swaps every chart to a colorblind-safe palette.

## Expected input

A Moodle **Responses** report, exported as `.csv`, `.xls`, or `.xlsx`:

1. In Moodle, open your quiz → **Quiz results** → **Responses** report.
2. Under **Display options**, check **Question text**, **Response**, and **Right answer**.
3. **Display report**, then download it.

Columns, left to right:

```
Last name | First name | Email address | State | Started on | Completed | Time taken | Grade/10.00
Question 1 | Response 1 | Right answer 1 | Question 2 | Response 2 | Right answer 2 | ...
```

Common alternates are recognized automatically (`Username` for `Email address`, `Status` for `State`, `Started` for `Started on`, `Duration` for `Time taken`), as are PRTs renamed away from `prt1`/`prt2`. A **Grades with question breakdown** export for the same quiz can be uploaded alongside a Responses report for more accurate per-question scoring. [Sample files](https://drive.google.com/drive/folders/1r7c1asoMFwaLORaQVKisJk7xpWazzC5I?usp=sharing) are available if you just want to see the app work.

The Solution Process Visualization section needs **multiple attempts per student** (several rows for the same student) to show a meaningful trajectory.

## Run it locally

Requires **Python 3.10–3.12**.

```
git clone https://github.com/jumazevick/Interactive-quiz-analytics.git
cd Interactive-quiz-analytics
python3.10 -m venv .venv && source .venv/bin/activate
pip install poetry && poetry install
streamlit run Home.py
```

Tests: `pytest`.

> **Python 3.13:** `poetry install` can fail building `rpds-py` (`the configured Python interpreter version (3.13) is newer than PyO3's maximum supported version (3.12)`) — a pinned transitive dependency via `jsonschema`, not this project's code. Use 3.10–3.12.

## Deployment notes

On Streamlit Community Cloud, set the main file to `Home.py` (or `streamlit_app.py`, which re-exports it).

PDF chart export goes through Kaleido, which needs a Chrome/Chromium binary on the host. Cloud's base image doesn't ship one, so `packages.txt` lists `chromium`; Cloud `apt-get`s everything in that file before the app starts, which also pulls in the shared libraries (`libnss3`, `libgbm1`, …) headless Chrome needs. If you fork this and charts silently stop appearing in exported PDFs, check `packages.txt` made it into your fork and that the deployment actually redeployed. The app also self-heals by downloading a private Chrome via `kaleido.get_chrome_sync()` and retrying once.

## Built with

Streamlit, Pandas, Plotly (+ Kaleido for chart rasterization), SciPy, Matplotlib (`mathtext`, for typesetting STACK math into PDF tables), ReportLab, OpenPyXL/xlrd. No graph or tree-distance library: the transition graphs only ever hold a handful of nodes, so their centrality measures and layout are computed directly, and the Maxima expression parser and Zhang-Shasha tree edit distance live in `analytics/expression_tree.py` and `analytics/tree_edit_distance.py`.

## Credits

Originally a hackathon project. Thanks to **Juma** (original idea and implementation, question/quiz analysis research, advising), **Ernest** (question analysis research, technical setup, development, UI, implementation), **Sage Foundation** (technical setup), and **Otis** (question analysis research).

The Solution Process Visualization section implements published methods, with full credit to their authors:

- **Asahi Kurihara** and **Yasuyuki Nakamura** (Nagoya University), *Network Analysis of Solution Processes in Math Online Tests*, LAK25 Companion Proceedings, 2025, pp. 257–259 — the answer-transition directed graphs and network features.
- **Tomoki Takada** and **Yasuyuki Nakamura** (Nagoya University) and **Saburo Higuchi** (Ryukoku University), [*Visualization of Solution Processes to Reach the Correct Answer in Online Math Tests*](https://doi.org/10.5281/zenodo.15870221), EDM 2025, pp. 635–639 — the 3D PRT-distance and Tree Edit Distance visualizations.

Contributions welcome — open a Pull Request.
