// Generic renderer for the {summary, sections} contract used by every view
// local_quizanalytics renders: course-wide comparison, per-quiz Question
// Analytics, and per-quiz Solution Process Visualization.
//
// Expected shape of a `result` object passed to QuizAnalyticsRenderer.render():
// {
//   "summary": { "student_count": 34, "mean_grade": 7.1, ... },
//   "sections": [
//     {
//       "id": "difficulty",
//       "title": "2. Question Difficulty Analysis",
//       "caption": "...",
//       "table": { "columns": ["question", "difficulty_index"], "rows": [["Q1", 0.62]] },
//       "charts": [ { "id": "difficulty-bar", "title": "Top Difficult Questions", "plotly_json": {"data": [...], "layout": {...}} } ],
//       "notes": ["..."]
//     }
//   ]
// }
//
// Backwards-compatible fallback: if `result.figures` is present instead of
// `result.sections` (the shape /analyze returned before Question Analysis
// enrichment), each figure is rendered flat with its own heading — this
// reproduces the exact pre-enrichment output with no visible change.
//
// After injecting each container's HTML, KaTeX's auto-render extension is
// run over it so any \( \)/\[ \] delimited math in table cells or notes
// renders as real math. Must run per-container (not once globally) since
// content is injected dynamically after this script's own top-level code
// has already executed.
(function (global) {
    'use strict';

    // A handful of keys/columns come through as raw Python identifiers
    // (quiz_name, total_questions, invalid_rate, ...) since they double as
    // dict keys/DataFrame column names on the Python side — this turns those
    // into presentable labels for both the summary table and generic data
    // tables, without needing to rename anything Python-side code depends on.
    // Applying it to already-nice strings (e.g. "Quiz Name") is a no-op.
    var LABEL_OVERRIDES = {
        'id': 'ID', 'ids': 'IDs', 'prt': 'PRT', 'prts': 'PRTs',
        'ted': 'TED', 'stack': 'STACK', 'url': 'URL',
    };

    // A few raw column names abbreviate in a way that word-by-word
    // capitalization can't fix on its own ("attempt_idx" -> "Attempt Idx",
    // not the "Attempt Number" a reader actually wants) — matched against
    // the whole key, case-insensitively, before falling through to the
    // generic word-splitting logic below.
    var FULL_LABEL_OVERRIDES = {
        'attempt_idx': 'Attempt Number',
        'attempt_count': 'No. of Attempts',
        'completed_dt': 'Completed On',
        'first_grade': 'First Grade',
        'latest_grade': 'Latest Grade',
    };

    function humanizeLabel(key) {
        if (typeof key !== 'string' || !key) {
            return key;
        }
        var fullOverride = FULL_LABEL_OVERRIDES[key.toLowerCase()];
        if (fullOverride) {
            return fullOverride;
        }
        return key
            .replace(/[_\-]+/g, ' ')
            .trim()
            .split(/\s+/)
            .map(function (word) {
                var override = LABEL_OVERRIDES[word.toLowerCase()];
                if (override) {
                    return override;
                }
                return word.charAt(0).toUpperCase() + word.slice(1).toLowerCase();
            })
            .join(' ');
    }

    // Matches the ISO-ish timestamps pandas emits for datetime columns
    // (completed_dt, started_on, ...), e.g. "2026-06-03T10:53:29.000" or
    // "2026-06-03 10:53:29" — reformatted to the reader's own locale instead
    // of shown as a raw sortable-but-not-readable timestamp string.
    var ISO_DATETIME_RE = /^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(\.\d+)?$/;

    function formatDateTimeValue(value) {
        if (typeof value !== 'string' || !ISO_DATETIME_RE.test(value)) {
            return null;
        }
        var parsed = new Date(value.replace(' ', 'T'));
        if (isNaN(parsed.getTime())) {
            return null;
        }
        return parsed.toLocaleString(undefined, {
            year: 'numeric', month: 'short', day: 'numeric',
            hour: 'numeric', minute: '2-digit',
        });
    }

    // Long floating-point results (grade_variance, attempt_rate, and similar
    // computed stats) come through with full binary-float precision (e.g.
    // 1.2962962963) — displayed values are capped to 2 decimal places.
    // Integers (attempt counts, ids that happen to be numeric, etc.) are
    // left exactly as-is rather than padded to "4.00".
    function formatCellValue(value) {
        if (typeof value === 'number' && !Number.isInteger(value)) {
            return String(Math.round(value * 100) / 100);
        }
        var asDateTime = formatDateTimeValue(value);
        if (asDateTime !== null) {
            return asDateTime;
        }
        return (value === null || value === undefined) ? '' : String(value);
    }

    // Long tables (many student rows especially) get capped to a scrollable
    // viewport instead of pushing the rest of the page down — same idea as
    // the fixed-height, scrollable dataframe viewer in the Streamlit app this
    // mirrors. Width is always scrollable too, independent of row count,
    // since a wide table (e.g. Question Metrics' ~12 columns) can overflow
    // its column even with only a handful of rows.
    var SCROLL_ROW_THRESHOLD = 12;

    function wrapScrollable(el, rowCount) {
        var wrapper = document.createElement('div');
        wrapper.style.overflowX = 'auto';
        // `overflow` (X here, X+Y below for a tall table) puts this wrapper
        // in its own block-formatting context, which stops the wrapped
        // table's own bottom margin from collapsing outward the normal way
        // — without an explicit margin on the wrapper itself, that margin
        // gets trapped inside it, leaving nothing between this table and
        // whatever renders right after it (the next section's own
        // heading/chart, with no visible gap at all).
        wrapper.style.marginBottom = '1.5rem';
        if (rowCount > SCROLL_ROW_THRESHOLD) {
            wrapper.style.maxHeight = '420px';
            wrapper.style.overflowY = 'auto';
            wrapper.style.border = '1px solid #dee2e6';
        }
        wrapper.appendChild(el);
        return wrapper;
    }

    function renderSummaryTable(root, summary) {
        if (!root || !summary) {
            return;
        }
        var table = document.createElement('table');
        table.className = 'generaltable';
        Object.keys(summary).forEach(function (key) {
            var row = table.insertRow();
            row.insertCell().textContent = humanizeLabel(key);
            row.insertCell().textContent = formatCellValue(summary[key]);
        });
        root.appendChild(table);
    }

    // Compact Moodle-native summary shown above an individual quiz's own
    // Question Analytics — replaces renderSummaryTable() there (see
    // renderResult()'s own branch below) with the {attempt counts, overall
    // average, per-question Facility Index} snapshot data_fetcher.php's
    // get_quiz_snapshot() reads straight from Moodle's own SQL, plus a link
    // to that quiz's own Moodle attempts report.
    function renderQuizSnapshot(root, snapshot) {
        if (!root || !snapshot) {
            return;
        }

        var heading = document.createElement('h4');
        heading.textContent = 'Quiz Snapshot';
        root.appendChild(heading);

        var overviewRows = [
            ['Overall average', (snapshot.quiz_average === null ? 'N/A' : Number(snapshot.quiz_average).toFixed(2)) +
                (snapshot.quiz_average_finished ? ' (Finished: ' + formatCellValue(snapshot.quiz_average_finished) + ')' : '')],
            ['Students with attempts', snapshot.students_with_attempts],
            ['Total attempts', snapshot.attempts_total],
            ['Finished', snapshot.attempts_finished],
            ['In progress', snapshot.attempts_inprogress],
        ];
        if (snapshot.attempts_other) {
            overviewRows.push(['Other', snapshot.attempts_other]);
        }
        var overview = document.createElement('div');
        overview.style.marginBottom = '0.75rem';
        renderDataTable(overview, {
            columns: ['Metric', 'Value'],
            rows: overviewRows.map(function (values) {
                return [values[0], formatCellValue(values[1])];
            }),
        });
        root.appendChild(overview);

        var link = document.createElement('a');
        link.href = snapshot.quiz_report_url;
        link.target = '_blank';
        link.rel = 'noopener noreferrer';
        link.textContent = 'View quiz attempts in Moodle ↗';
        root.appendChild(link);
    }

    function renderDataTable(root, table) {
        if (!table || !table.columns || !table.rows) {
            return;
        }
        var el = document.createElement('table');
        el.className = 'generaltable';
        var thead = el.createTHead();
        var headRow = thead.insertRow();
        table.columns.forEach(function (col) {
            var th = document.createElement('th');
            th.textContent = humanizeLabel(col);
            headRow.appendChild(th);
        });
        var tbody = el.createTBody();
        table.rows.forEach(function (rowValues) {
            var row = tbody.insertRow();
            rowValues.forEach(function (value) {
                var cell = row.insertCell();
                cell.innerHTML = formatCellValue(value);
            });
        });
        root.appendChild(wrapScrollable(el, table.rows.length));
    }

    var chartCounter = 0;

    // A figure whose own layout sets a tall fixed height (e.g. the Student
    // Performance Matrix heatmap, sized to 24px per student row — see
    // question_charts.py::build_student_matrix_figure) gets capped to a
    // scrollable viewport instead of stretching the page, matching the
    // Streamlit app's own fixed-height chart containers.
    var CHART_SCROLL_HEIGHT_THRESHOLD = 500;
    var CHART_SCROLL_MAX_HEIGHT = 500;

    // "Line Graph of Various Metrics" sets an explicit layout.width of
    // 220px per quiz (question_analysis.php: max(800, 220 * $ncategories))
    // so a course-wide view's per-quiz tick labels stay legible instead of
    // overlapping — for a large course (dozens of quizzes) that easily runs
    // past any reasonable page width. Without a bounding box, that stretched
    // the whole page horizontally rather than just the one chart. This
    // doesn't affect the PDF export's own chart capture: collectChartImages()
    // below reads the inner chart div's own offsetWidth, which keeps its
    // full intrinsic size regardless of an ancestor's overflow/max-width.
    var CHART_SCROLL_WIDTH_THRESHOLD = 900;
    // The wrapper's own box stays 100% of its parent — matching the width of
    // every other element on the page (tables, headings, the other charts)
    // — with the (potentially much wider) chart scrolling *inside* it,
    // rather than the wrapper itself shrinking to a fixed pixel width
    // narrower than everything around it.
    var CHART_SCROLL_MAX_WIDTH = '100%';

    function renderChart(root, chart, prefix) {
        if (!chart || !chart.plotly_json) {
            return;
        }
        if (chart.title) {
            var heading = document.createElement('h5');
            heading.textContent = chart.title;
            root.appendChild(heading);
        }
        var container = document.createElement('div');
        container.id = chart.id ? (prefix + '-chart-' + chart.id) : (prefix + '-chart-auto-' + (chartCounter++));
        container.style.marginBottom = '2rem';

        // 3D scene charts (the PRT/TED distance charts) are exempt: their
        // height is a fixed, deliberate choice giving aspectmode="cube"
        // enough room to render the cube at a reasonable size (see
        // solution_distance.php's own comment on this), not a list that
        // grows unbounded with the data the way the student-performance
        // heatmap's height does — capping it to a small scrollable window
        // would just shrink the visible chart back down to the same
        // "looks tiny" problem that height was set to fix.
        var isScene3d = !!(chart.plotly_json.layout && chart.plotly_json.layout.scene);
        // The Question Response Overview chart (one horizontal bar row per
        // question) grows its own height with the question count the same
        // way the old Student Performance Matrix heatmap did — but unlike
        // that heatmap, its whole point is to let a teacher scan every
        // question at a glance, so it's exempt from the scroll cap too
        // rather than getting shrunk back into a small scrollable window.
        var isResponseStatusChart = chart.id === 'response-status';
        var declaredHeight = chart.plotly_json.layout && chart.plotly_json.layout.height;
        var declaredWidth = chart.plotly_json.layout && chart.plotly_json.layout.width;
        var needsHeightScroll = !isScene3d && !isResponseStatusChart &&
            declaredHeight && declaredHeight > CHART_SCROLL_HEIGHT_THRESHOLD;
        var needsWidthScroll = declaredWidth && declaredWidth > CHART_SCROLL_WIDTH_THRESHOLD;
        if (needsHeightScroll || needsWidthScroll) {
            var scrollWrapper = document.createElement('div');
            if (needsHeightScroll) {
                scrollWrapper.style.maxHeight = CHART_SCROLL_MAX_HEIGHT + 'px';
                scrollWrapper.style.overflowY = 'auto';
            }
            if (needsWidthScroll) {
                scrollWrapper.style.maxWidth = CHART_SCROLL_MAX_WIDTH;
            }
            scrollWrapper.style.overflowX = 'auto';
            scrollWrapper.style.border = '1px solid #dee2e6';
            scrollWrapper.style.marginBottom = '2rem';
            container.style.marginBottom = '0';
            scrollWrapper.appendChild(container);
            root.appendChild(scrollWrapper);
        } else {
            root.appendChild(container);
        }

        // Pass the element itself, not container.id: renderSection() appends
        // its wrapper to the live document before populating it (so this
        // container has real, laid-out dimensions for Plotly's "responsive"
        // sizing to measure), but document.getElementById() would still be
        // one more indirection than necessary — passing the element directly
        // also sidesteps a null lookup if that ever regresses.
        global.Plotly.newPlot(container, chart.plotly_json.data, chart.plotly_json.layout, {
            responsive: true,
        });
        return container;
    }

    // Renders the scatter chart plus a radio group ("Compare attempts
    // against: Highest/Average/Minimum Grade") that swaps the active Plotly
    // figure client-side via Plotly.react() — every variant's figure was
    // already computed server-side (see course_analysis::build_analysis()'s
    // $scattervariants) and shipped in section.scatter_variants, so
    // switching never reloads the page.
    function renderScatterControls(wrapper, section, prefix) {
        var variants = section.scatter_variants || {};
        var types = Object.keys(variants);
        if (!types.length) {
            return;
        }

        var controls = document.createElement('div');
        controls.className = 'mb-3';
        var label = document.createElement('span');
        label.textContent = 'Compare attempts against: ';
        controls.appendChild(label);

        var initial = section.selected_scatter_type || types[0];
        var chart = section.charts && section.charts[0];
        var chartContainer = renderChart(wrapper, chart, prefix);
        var caption = wrapper.querySelector('p');

        types.forEach(function (type) {
            var id = prefix + '-scatter-' + type.toLowerCase().replace(/[^a-z]+/g, '-');
            var radio = document.createElement('input');
            radio.type = 'radio';
            radio.name = prefix + '-scatter-grade-type';
            radio.id = id;
            radio.value = type;
            radio.checked = type === initial;
            radio.style.marginLeft = '0.75rem';

            var radioLabel = document.createElement('label');
            radioLabel.htmlFor = id;
            radioLabel.textContent = variants[type].label;
            radioLabel.style.marginLeft = '0.25rem';
            controls.appendChild(radio);
            controls.appendChild(radioLabel);

            radio.addEventListener('change', function () {
                var variant = variants[radio.value];
                global.Plotly.react(chartContainer, variant.plotly_json.data, variant.plotly_json.layout, {
                    responsive: true,
                });
                if (caption) {
                    caption.textContent = variant.caption;
                }
            });
        });

        wrapper.insertBefore(controls, chartContainer.parentNode === wrapper ? chartContainer : chartContainer.parentNode);
    }

    function renderNotes(root, notes) {
        if (!notes || !notes.length) {
            return;
        }
        var list = document.createElement('ul');
        notes.forEach(function (note) {
            var item = document.createElement('li');
            item.innerHTML = note;
            list.appendChild(item);
        });
        // Same scrollable-viewport treatment as long data tables (e.g. a
        // grade-mismatch audit can list one line per affected student) —
        // reuses wrapScrollable's row-count threshold, one <li> per row.
        root.appendChild(wrapScrollable(list, notes.length));
    }

    // Turns each row of the Cross-Attempt Comparison table into a link that
    // reloads the current page with that student selected as the drill-down
    // (?spvstudent=...) — same plain GET-reload convention as every other
    // selector in this plugin family, so the result is still a cacheable
    // URL, just reached with one click on a row instead of picking the
    // student from the dropdown above and clicking Go. `studentIds` is
    // section.row_student_ids from the API response, parallel to the
    // table's own rows (see app.py's /solution-process route).
    function makeCrossAttemptRowsClickable(sectionRoot, studentIds) {
        var table = sectionRoot.querySelector('table');
        if (!table || !table.tBodies.length) {
            return;
        }
        var rows = table.tBodies[0].rows;
        Array.prototype.forEach.call(rows, function (row, i) {
            var studentId = studentIds[i];
            var nameCell = row.cells[0]; // "Student Name" is always the first column here.
            if (!studentId || !nameCell) {
                return;
            }
            var url = new URL(global.location.href);
            url.searchParams.set('spvstudent', studentId);

            var link = document.createElement('a');
            link.href = url.toString();
            link.textContent = nameCell.textContent;
            nameCell.textContent = '';
            nameCell.appendChild(link);
        });
    }

    // Every section wrapper gets the same top rule + generous top/bottom
    // spacing, so consecutive sections (and the "Generate PDF Report" block
    // right after the last one) read as clearly separate blocks instead of
    // running straight into each other — there was previously no CSS at all
    // marking this boundary, just whatever margin a section's own last
    // element (a table, a scrollable notes list, ...) happened to have.
    function styleSectionWrapper(wrapper) {
        wrapper.style.borderTop = '1px solid #dee2e6';
        wrapper.style.marginTop = '2rem';
        wrapper.style.paddingTop = '1.5rem';
        wrapper.style.marginBottom = '1rem';
    }

    function renderSection(root, section, prefix) {
        var wrapper = document.createElement('div');
        wrapper.className = 'qa-section';
        styleSectionWrapper(wrapper);
        if (section.id) {
            wrapper.id = prefix + '-section-' + section.id;
        }
        // Attached to the live document BEFORE being populated: a chart drawn
        // into a still-detached container has no real layout width for
        // Plotly's "responsive" sizing to measure (getBoundingClientRect()
        // on a detached element is 0x0), so it falls back to Plotly's small
        // default size and never grows to fill the page afterward — nothing
        // re-triggers that measurement once the wrapper is later attached.
        // This was the cause of charts rendering visibly narrower than their
        // surrounding text.
        root.appendChild(wrapper);
        if (section.title) {
            var heading = document.createElement('h4');
            heading.textContent = section.title;
            wrapper.appendChild(heading);
        }
        if (section.caption) {
            var caption = document.createElement('p');
            if (section.caption_html) {
                caption.innerHTML = section.caption_html;
            } else {
                caption.textContent = section.caption;
            }
            wrapper.appendChild(caption);
        }
        if (section.column_help) {
            var help = document.createElement('details');
            help.className = 'mb-3';
            var summary = document.createElement('summary');
            summary.textContent = 'Column guide';
            help.appendChild(summary);
            var helpList = document.createElement('ul');
            Object.keys(section.column_help).forEach(function (column) {
                var item = document.createElement('li');
                var label = document.createElement('strong');
                label.textContent = humanizeLabel(column) + ': ';
                item.appendChild(label);
                item.appendChild(document.createTextNode(section.column_help[column]));
                helpList.appendChild(item);
            });
            help.appendChild(helpList);
            wrapper.appendChild(help);
        }
        if (section.table) {
            renderDataTable(wrapper, section.table);
            if (section.id === 'cross-attempt' && Array.isArray(section.row_student_ids)) {
                makeCrossAttemptRowsClickable(wrapper, section.row_student_ids);
            }
        }
        if (section.id === 'scatter' && section.scatter_variants) {
            renderScatterControls(wrapper, section, prefix);
        } else if (section.charts) {
            section.charts.forEach(function (chart) {
                renderChart(wrapper, chart, prefix);
            });
        }
        if (section.notes) {
            renderNotes(wrapper, section.notes);
        }
        typesetMath(wrapper);
    }

    var questionBlockCounter = 0;

    function renderQuestionDetails(root, prefix, questions, snapshot) {
        var names = Object.keys(questions || {});
        if (!names.length) {
            return;
        }
        var wrapper = document.createElement('div');
        wrapper.className = 'qa-section';
        styleSectionWrapper(wrapper);
        wrapper.id = prefix + '-section-questiondetails';

        var heading = document.createElement('h4');
        heading.textContent = 'Question Review';
        wrapper.appendChild(heading);

        var caption = document.createElement('p');
        caption.textContent = 'Inspect the question, expected answer, and common response ' +
            'patterns to understand where students had difficulty.';
        wrapper.appendChild(caption);

        var selectId = prefix + '-question-select-' + (questionBlockCounter++);
        var label = document.createElement('label');
        label.setAttribute('for', selectId);
        label.textContent = 'Select Question: ';
        label.style.fontWeight = '600';
        label.style.marginRight = '0.5rem';
        wrapper.appendChild(label);

        var select = document.createElement('select');
        select.id = selectId;
        select.style.minWidth = '10rem';
        select.style.padding = '0.55rem 2rem 0.55rem 0.75rem';
        select.style.fontSize = '1rem';
        select.style.border = '2px solid #0f6cbf';
        select.style.borderRadius = '0.35rem';
        select.style.backgroundColor = '#eef6fc';
        names.forEach(function (name) {
            var option = document.createElement('option');
            option.value = name;
            option.textContent = name;
            select.appendChild(option);
        });
        wrapper.appendChild(select);

        var blocksRoot = document.createElement('div');
        names.forEach(function (name, i) {
            var detail = questions[name];
            var block = document.createElement('div');
            block.className = 'qa-question-block';
            block.setAttribute('data-question', name);
            block.style.display = (i === 0) ? 'block' : 'none';
            block.style.marginTop = '1rem';

            var versions = detail.versions || [];
            versions.forEach(function (version) {
                // Each version gets its own bordered card so a question with
                // several randomized variants doesn't read as one continuous
                // block — the version boundary is the thing a reader most
                // needs to spot at a glance here.
                var versionBox = document.createElement('div');
                versionBox.className = 'qa-version-box';
                versionBox.style.border = '1px solid #d3d9e0';
                versionBox.style.borderRadius = '0.5rem';
                versionBox.style.padding = '1rem 1.25rem';
                versionBox.style.marginBottom = '1.25rem';
                versionBox.style.backgroundColor = '#fbfcfd';

                var versionHeading = document.createElement('h5');
                versionHeading.textContent = version.label;
                versionHeading.style.marginTop = '0';
                versionBox.appendChild(versionHeading);

                // Each heading/content pair gets a consistent gap above (from
                // whatever came before) and below (from its own content) —
                // previously these sat with no explicit spacing at all, so a
                // heading landed flush against the content right above it.
                var qHeading = document.createElement('h6');
                qHeading.textContent = 'Question Text';
                qHeading.style.marginTop = '1.1rem';
                qHeading.style.marginBottom = '0.4rem';
                versionBox.appendChild(qHeading);
                var qText = document.createElement('div');
                qText.innerHTML = version.question_text_html || '';
                qText.style.marginBottom = '0.5rem';
                versionBox.appendChild(qText);

                var aHeading = document.createElement('h6');
                aHeading.textContent = 'Expected Answer';
                aHeading.style.marginTop = '1.1rem';
                aHeading.style.marginBottom = '0.4rem';
                versionBox.appendChild(aHeading);
                var aText = document.createElement('div');
                aText.innerHTML = version.right_answer_html || '';
                aText.style.marginBottom = '0.5rem';
                versionBox.appendChild(aText);

                var dHeading = document.createElement('h6');
                dHeading.textContent = 'Most Common Incorrect Responses';
                dHeading.style.marginTop = '1.1rem';
                dHeading.style.marginBottom = '0.5rem';
                versionBox.appendChild(dHeading);
                var common = version.common_responses || [];
                if (common.length) {
                    renderDataTable(versionBox, {
                        columns: ['Response', 'Students'],
                        rows: common.map(function (item) {
                            return [item.response, item.students];
                        }),
                    });
                } else {
                    var noCommon = document.createElement('p');
                    noCommon.textContent = 'No incorrect response patterns were found.';
                    versionBox.appendChild(noCommon);
                }

                if (version.review_url) {
                    var versionLink = document.createElement('a');
                    versionLink.href = version.review_url;
                    versionLink.target = '_blank';
                    versionLink.rel = 'noopener noreferrer';
                    versionLink.textContent = 'Review an example attempt in Moodle ↗';
                    versionLink.style.display = 'inline-block';
                    versionLink.style.marginTop = '1rem';
                    versionLink.style.marginBottom = '0';
                    versionBox.appendChild(versionLink);
                }

                block.appendChild(versionBox);
            });

            blocksRoot.appendChild(block);
        });
        wrapper.appendChild(blocksRoot);

        select.addEventListener('change', function () {
            var chosen = select.value;
            Array.prototype.forEach.call(blocksRoot.children, function (block) {
                block.style.display = (block.getAttribute('data-question') === chosen) ? 'block' : 'none';
            });
        });

        if (snapshot && snapshot.quiz_responses_url) {
            var reportLink = document.createElement('a');
            reportLink.href = snapshot.quiz_responses_url;
            reportLink.target = '_blank';
            reportLink.rel = 'noopener noreferrer';
            reportLink.textContent = 'View individual responses in Moodle ↗';
            reportLink.style.display = 'inline-block';
            reportLink.style.marginTop = '1rem';
            wrapper.appendChild(reportLink);
        }

        root.appendChild(wrapper);
        typesetMath(wrapper);
    }

    function renderAudit(root, prefix, audit) {
        if (!audit) {
            return;
        }
        var wrapper = document.createElement('div');
        wrapper.className = 'qa-section';
        styleSectionWrapper(wrapper);
        wrapper.id = prefix + '-section-audit';

        var heading = document.createElement('h4');
        heading.textContent = '7. Interpretation Notes & Export';
        wrapper.appendChild(heading);

        if (audit.issues && audit.issues.length) {
            renderNotes(wrapper, audit.issues);
        } else {
            var ok = document.createElement('p');
            ok.textContent = 'No data-quality issues detected.';
            wrapper.appendChild(ok);
        }

        root.appendChild(wrapper);
        typesetMath(wrapper);
    }

    function renderStudentDrilldown(root, prefix, drilldown) {
        if (!drilldown || !drilldown.sections || !drilldown.sections.length) {
            return;
        }
        var wrapper = document.createElement('div');
        wrapper.className = 'qa-section';
        styleSectionWrapper(wrapper);
        wrapper.id = prefix + '-section-student-drilldown';
        // Attached before populating — renderSection() below draws charts
        // into it, which need a live (attached) ancestor to size against;
        // see the matching comment inside renderSection() for the full story.
        root.appendChild(wrapper);

        var heading = document.createElement('h4');
        heading.textContent = 'Student drill-down: ' + (drilldown.student_name || drilldown.student_id || '');
        wrapper.appendChild(heading);

        drilldown.sections.forEach(function (section) {
            renderSection(wrapper, section, prefix);
        });

        typesetMath(wrapper);
    }

    function renderLegacyFigures(root, figures, prefix) {
        figures.forEach(function (fig, i) {
            var heading = document.createElement('h4');
            heading.textContent = fig.title || ('Chart ' + (i + 1));
            root.appendChild(heading);

            var container = document.createElement('div');
            container.id = prefix + '-chart-' + i;
            container.style.marginBottom = '2rem';
            root.appendChild(container);

            global.Plotly.newPlot(container, fig.plotly_json.data, fig.plotly_json.layout, {
                responsive: true,
            });
        });
    }

    function typesetMath(container) {
        if (typeof global.renderMathInElement !== 'function') {
            return;
        }
        global.renderMathInElement(container, {
            delimiters: [
                // STACK/Moodle's own question text uses \( \) / \[ \] natively.
                {left: '\\(', right: '\\)', display: false},
                {left: '\\[', right: '\\]', display: true},
                // analytics.latex_utils.extract_stack_answer_latex() and
                // compute_repeated_wrong_answers() wrap converted Maxima
                // expressions in single/double $ instead (the Streamlit/
                // MathJax convention) — recognized in addition since these
                // strings are entirely programmatically constructed by
                // those two functions, not free-form text where a literal
                // "$" could cause a false-positive match.
                {left: '$$', right: '$$', display: true},
                {left: '$', right: '$', display: false},
            ],
            throwOnError: false,
        });
    }

    /**
     * @param {string} prefix DOM id prefix — e.g. "qa" gives #qa-summary/#qa-sections.
     * @param {object} result The {summary, sections} (or legacy {summary, figures}) payload.
     */
    function render(prefix, result) {
        var summaryRoot = document.getElementById(prefix + '-summary');
        var sectionsRoot = document.getElementById(prefix + '-sections');
        if (!result) {
            return;
        }

        // Question Analytics results carry a Moodle-native snapshot instead
        // of the generic key/value summary table every other view (course-
        // wide, Solution Process) still uses.
        if (result.snapshot) {
            renderQuizSnapshot(summaryRoot, result.snapshot);
        } else {
            renderSummaryTable(summaryRoot, result.summary);
        }
        typesetMath(summaryRoot);

        if (sectionsRoot && Array.isArray(result.sections)) {
            // The per-question drill-down follows the response overview,
            // matching this simplified page's own reduced section list.
            result.sections.forEach(function (section) {
                renderSection(sectionsRoot, section, prefix);
                if (section.id === 'question-response-overview') {
                    renderQuestionDetails(sectionsRoot, prefix, result.questions, result.snapshot);
                }
            });
            renderAudit(sectionsRoot, prefix, result.audit);
            renderStudentDrilldown(sectionsRoot, prefix, result.student_drilldown);
        } else if (sectionsRoot && Array.isArray(result.figures)) {
            renderLegacyFigures(sectionsRoot, result.figures, prefix);
        }
    }

    // --- PDF export: client-side chart capture -----------------------------
    //
    // pdf.php has no headless-browser/rasterization dependency (the whole
    // point of this plugin being pure PHP), so it can't turn a Plotly figure
    // into an image itself. Instead, on "Generate PDF Report" click, every
    // chart already drawn on this page (by renderChart() above, into a
    // "{prefix}-chart-{chart.id}" container) is snapshotted client-side via
    // Plotly's own toImage(), and the resulting PNG data URLs are POSTed
    // alongside the section checkboxes — pdf.php embeds whichever ones match
    // the sections it re-derives server-side, and shows a plain "unavailable"
    // note for any section whose chart wasn't captured (e.g. a chart added
    // to the page after this script last ran).
    function collectChartImages(prefix) {
        var selector = '[id^="' + prefix + '-chart-"]';
        var containers = document.querySelectorAll(selector);
        var ids = [];
        var promises = [];
        Array.prototype.forEach.call(containers, function (container) {
            ids.push(container.id.slice(prefix.length + 7)); // strip "{prefix}-chart-"
            promises.push(global.Plotly.toImage(container, {
                format: 'png',
                width: container.offsetWidth || 800,
                height: container.offsetHeight || 400,
            }));
        });
        return Promise.all(promises).then(function (images) {
            var map = {};
            ids.forEach(function (chartId, i) {
                map[chartId] = images[i];
            });
            return map;
        });
    }

    function setupPdfForm(form) {
        var prefix = form.getAttribute('data-chart-prefix');
        var imagesInput = form.querySelector('input[name="chart_images"]');
        var submitBtn = form.querySelector('input[type="submit"]');
        if (!prefix || !imagesInput || !submitBtn) {
            return;
        }
        form.addEventListener('submit', function (e) {
            if (form.dataset.imagesReady === '1') {
                return; // Images already captured on a previous submit attempt.
            }
            e.preventDefault();
            var originalLabel = submitBtn.value;
            submitBtn.value = 'Generating chart images…';
            submitBtn.disabled = true;
            collectChartImages(prefix).then(function (images) {
                imagesInput.value = JSON.stringify(images);
                form.dataset.imagesReady = '1';
                submitBtn.value = originalLabel;
                submitBtn.disabled = false;
                form.submit();
            }).catch(function (err) {
                submitBtn.value = originalLabel;
                submitBtn.disabled = false;
                global.alert('Could not capture chart images for the PDF: ' +
                    (err && err.message ? err.message : err));
            });
        });
    }

    function initPdfForms() {
        var forms = document.querySelectorAll('form.qa-pdf-form');
        Array.prototype.forEach.call(forms, setupPdfForm);
    }

    // This script tag runs before the PDF form's own HTML exists in the page
    // (render_pdf_form() is echoed later in index.php's document order) —
    // DOMContentLoaded waits for the whole initial HTML response to finish
    // parsing regardless of where this <script> sits within it, so the form
    // is reliably present by the time this fires.
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initPdfForms);
    } else {
        initPdfForms();
    }

    global.QuizAnalyticsRenderer = {render: render};
})(window);
