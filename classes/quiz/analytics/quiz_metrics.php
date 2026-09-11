<?php
// This file is part of Moodle - http://moodle.org/
//
// Moodle is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// Moodle is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with Moodle.  If not, see <http://www.gnu.org/licenses/>.

/**
 * PHP port of analytics-service/analytics/quiz_metrics.py.
 *
 * @package local_quizanalytics
 * @copyright  2026 Ernest Ting <eting@caltech.edu>
 * @license    http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */

namespace local_quizanalytics\quiz\analytics;

/**
 * Course-wide (cross-quiz) engagement, scatter, and metric-trend charts, and the quiz stats/attempt-list tables.
 */
class quiz_metrics {
    /** @var string Okabe-Ito vermillion — a colorblind-safer stand-in for DEFAULT_ACCENT. */
    const COLORBLIND_ACCENT = '#D55E00';

    /** @var string Default reddish mean-grade overlay line/marker color. */
    const DEFAULT_ACCENT = '#FF474C';

    /** @var string[] Column order for the internal per-attempt analytics frame. */
    const ATTEMPT_FRAME_COLUMNS = [
        'quiz_name', 'student_name', 'student_id', 'attempt_idx', 'attempt_id', 'attempt_number', 'cmid',
        'overall_grade', 'completed_dt', 'started_on',
    ];

    /**
     * Marker size range for the Attempts vs Grades scatter's density cue: a
     * point on its own renders at the low end, a coordinate shared by
     * several students renders larger, clamped so the largest overlaps
     * don't balloon into a dominant bubble chart.
     *
     * @var int
     */
    const SCATTER_MARKER_SIZE_MIN = 12;

    /** @var int Upper bound of the scatter marker size range - see SCATTER_MARKER_SIZE_MIN. */
    const SCATTER_MARKER_SIZE_MAX = 22;

    /** @var int Overlap count at which the scatter marker size range saturates to SCATTER_MARKER_SIZE_MAX. */
    const SCATTER_MARKER_SIZE_SATURATES_AT = 6;

    /**
     * Splits a label into its word tokens (each token keeping its trailing separator/whitespace).
     */
    private static function label_tokens(string $label): array {
        preg_match_all('/[^\s_-]+[\s_-]*/', $label, $m);
        return $m[0];
    }

    /**
     * Wrap a long quiz name onto up to $maxlines short horizontal lines
     * (joined with <br>) instead of leaving Plotly to render it as one long
     * diagonal tick label that eats most of the chart's vertical space.
     * Tokenizes on underscores and hyphens as well as whitespace.
     */
    public static function wrap_category_label(string $label, int $maxchars = 22, int $maxlines = 2): string {
        $tokens = self::label_tokens($label);
        if (empty($tokens)) {
            return $label;
        }
        $lines = [];
        $current = '';
        $remaining = $tokens;
        while (!empty($remaining) && count($lines) < $maxlines) {
            $token = $remaining[0];
            if ($current === '' && strlen($token) > $maxchars) {
                $lines[] = rtrim(substr($token, 0, $maxchars - 1)) . "\u{2026}";
                array_shift($remaining);
                continue;
            }
            $candidate = $current . $token;
            if (strlen($candidate) > $maxchars && $current !== '') {
                $lines[] = rtrim($current);
                $current = '';
                continue;
            }
            $current = $candidate;
            array_shift($remaining);
        }
        if ($current !== '' && count($lines) < $maxlines) {
            $lines[] = rtrim($current);
        }
        if (!empty($remaining) && !(!empty($lines) && str_ends_with(end($lines), "\u{2026}"))) {
            if (!empty($lines)) {
                $lines[count($lines) - 1] = rtrim($lines[count($lines) - 1]) . "\u{2026}";
            } else {
                $lines = [rtrim(substr($current, 0, $maxchars)) . "\u{2026}"];
            }
        }
        return implode('<br>', $lines);
    }

    /**
     * Collapse the long per-question response rows to one row per attempt,
     * across all uploaded quiz files. Dedupes on (quiz_name, attempt_idx)
     * rather than attempt_idx alone, since attempt_idx is only assigned
     * uniquely within a single quiz.
     *
     * @param array[] $responserows
     * @return array[] one row per attempt, ATTEMPT_FRAME_COLUMNS fields only
     */
    public static function build_quiz_attempt_frame(array $responserows): array {
        $seen = [];
        $out = [];
        foreach ($responserows as $row) {
            $key = $row['quiz_name'] . '|' . $row['attempt_idx'];
            if (isset($seen[$key])) {
                continue;
            }
            $seen[$key] = true;
            $entry = [];
            foreach (self::ATTEMPT_FRAME_COLUMNS as $col) {
                $entry[$col] = $row[$col] ?? null;
            }
            $out[] = $entry;
        }
        return $out;
    }

    /**
     * Builds the compact one-row-per-student-per-quiz summary table.
     * The latest attempt is linked to Moodle's native review page.
     *
     * @param array[] $attemptframe
     * @return array[]
     */
    public static function build_student_quiz_summary(array $attemptframe): array {
        $groups = [];
        foreach ($attemptframe as $row) {
            $key = (string) $row['quiz_name'] . '|' . (string) $row['student_id'];
            $groups[$key][] = $row;
        }

        $summary = [];
        foreach ($groups as $rows) {
            usort($rows, static function (array $a, array $b): int {
                $atime = strtotime((string) ($a['completed_dt'] ?: $a['started_on'])) ?: 0;
                $btime = strtotime((string) ($b['completed_dt'] ?: $b['started_on'])) ?: 0;
                if ($atime !== $btime) {
                    return $atime <=> $btime;
                }
                $anumber = (int) ($a['attempt_number'] ?? 0);
                $bnumber = (int) ($b['attempt_number'] ?? 0);
                if ($anumber !== $bnumber) {
                    return $anumber <=> $bnumber;
                }
                return ((int) ($a['attempt_id'] ?? 0)) <=> ((int) ($b['attempt_id'] ?? 0));
            });

            $first = $rows[0];
            $latest = $rows[count($rows) - 1];
            $reviewurl = new \moodle_url('/mod/quiz/review.php', [
                'attempt' => (int) $latest['attempt_id'],
                'cmid' => (int) $latest['cmid'],
            ]);

            $summary[] = [
                'user' => \html_writer::link($reviewurl, $latest['student_name'], [
                    'target' => '_blank',
                    'rel' => 'noopener',
                ]),
                'quiz_name' => $latest['quiz_name'],
                'attempt_count' => count($rows),
                'first_grade' => $first['overall_grade'],
                'latest_grade' => $latest['overall_grade'],
            ];
        }

        return $summary;
    }

    /**
     * Same formulas as the original per-file Quiz Analysis page, grouped by
     * quiz_name and reading overall_grade instead of a locally normalized
     * grade.
     *
     * @param array[] $attemptframe
     * @param string[] $selectedstats
     * @param array<string, array{facility_index?: float|null, quiz_url?: string}> $quizmetadata
     * @return array[] one row per quiz_name
     */
    public static function compute_quiz_stats(
        array $attemptframe,
        array $selectedstats,
        array $quizmetadata = []
    ): array {
        if (empty($attemptframe)) {
            return [];
        }

        $byquiz = table_helpers::group_by($attemptframe, 'quiz_name');
        // group_by() preserves each quiz's first-appearance order from
        // $attemptframe, which callers build by iterating quizzes in
        // data_fetcher::get_course_stack_quizzes()'s own chronological
        // order — deliberately kept as-is rather than the Python original's
        // pandas groupby("quiz_name") (sort=True default, i.e. alphabetical)
        // so a course-wide chart's quiz axis reads as a timeline instead of
        // "Quiz 10" sorting before "Quiz 2".
        $quiznames = array_keys($byquiz);

        $statsbyquiz = array_fill_keys($quiznames, []);

        if (in_array('student_count', $selectedstats, true)) {
            foreach ($byquiz as $quiz => $rows) {
                $statsbyquiz[$quiz]['student_count'] = count(array_unique(array_map(fn($r) => $r['student_id'], $rows)));
            }
        }

        if (in_array('mean_grade', $selectedstats, true) || in_array('grade_variance', $selectedstats, true)) {
            foreach ($byquiz as $quiz => $rows) {
                $grades = array_map(fn($r) => $r['overall_grade'], $rows);
                if (in_array('mean_grade', $selectedstats, true)) {
                    $statsbyquiz[$quiz]['mean_grade'] = stats::mean($grades);
                }
                if (in_array('grade_variance', $selectedstats, true)) {
                    $statsbyquiz[$quiz]['grade_variance'] = stats::sample_variance($grades);
                }
            }
        }

        if (in_array('mean_highest_grade', $selectedstats, true)) {
            foreach ($byquiz as $quiz => $rows) {
                $maxbystudent = [];
                foreach ($rows as $r) {
                    $sid = $r['student_id'];
                    $maxbystudent[$sid] = isset($maxbystudent[$sid])
                        ? max($maxbystudent[$sid], $r['overall_grade'])
                        : $r['overall_grade'];
                }
                $statsbyquiz[$quiz]['mean_highest_grade'] = stats::mean(array_values($maxbystudent));
            }
        }

        if (in_array('attempt_count', $selectedstats, true)) {
            foreach ($byquiz as $quiz => $rows) {
                $statsbyquiz[$quiz]['attempt_count'] = count($rows);
            }
        }

        if (in_array('attempt_rate', $selectedstats, true)) {
            foreach ($byquiz as $quiz => $rows) {
                $counts = [];
                foreach ($rows as $r) {
                    $counts[$r['student_id']] = ($counts[$r['student_id']] ?? 0) + 1;
                }
                $statsbyquiz[$quiz]['attempt_rate'] = stats::mean(array_values($counts));
            }
        }

        $out = [];
        foreach ($quiznames as $quiz) {
            if (empty($statsbyquiz[$quiz])) {
                continue;
            }
            $row = ['quiz_name' => $quiz];
            if (!empty($quizmetadata[$quiz]['quiz_url'])) {
                $row['quiz_name'] = \html_writer::link(
                    $quizmetadata[$quiz]['quiz_url'],
                    format_string($quiz),
                    ['target' => '_blank', 'rel' => 'noopener']
                );
            }
            foreach ($statsbyquiz[$quiz] as $k => $v) {
                $row[$k] = py_compat::round($v, 2);
            }
            if (array_key_exists('facility_index', $quizmetadata[$quiz] ?? [])) {
                $row['facility_index'] = $quizmetadata[$quiz]['facility_index'];
            }
            $out[] = $row;
        }
        return $out;
    }

    /**
     * Grade distribution per quiz, with an overlaid mean_grade line.
     *
     * @param array[] $attemptframe
     */
    public static function build_boxplot_figure(array $attemptframe, bool $colorblindmode = false): array {
        $quiznames = [];
        foreach ($attemptframe as $row) {
            $quiznames[$row['quiz_name']] = true;
        }
        $quiznames = array_keys($quiznames);
        $palette = chart_helpers::qualitative_colors($colorblindmode, chart_helpers::PALETTE_BOLD);

        $data = [];
        foreach ($quiznames as $i => $quiz) {
            $rows = array_values(array_filter($attemptframe, fn($r) => $r['quiz_name'] === $quiz));
            $data[] = [
                'type' => 'box',
                'y' => array_map(fn($r) => $r['overall_grade'], $rows),
                'x0' => $quiz,
                'name' => (string) $quiz,
                'boxpoints' => 'all',
                'jitter' => 0.3,
                'pointpos' => 0,
                'marker' => ['color' => $palette[$i % count($palette)], 'size' => 4, 'opacity' => 0.6],
            ];
        }

        $accent = $colorblindmode ? self::COLORBLIND_ACCENT : self::DEFAULT_ACCENT;
        // Reuses $quiznames' own (chronological, first-appearance) order
        // rather than a separately alphabetically-sorted copy, so the mean
        // line's markers land under the box they actually summarize instead
        // of drifting to wherever that quiz's name would sort alphabetically.
        $meansquiznames = $quiznames;
        $meansx = [];
        $meansy = [];
        foreach ($meansquiznames as $quiz) {
            $rows = array_values(array_filter($attemptframe, fn($r) => $r['quiz_name'] === $quiz));
            $meansx[] = $quiz;
            $meansy[] = stats::mean(array_map(fn($r) => $r['overall_grade'], $rows));
        }
        $data[] = [
            'type' => 'scatter',
            'x' => $meansx, 'y' => $meansy,
            'mode' => 'lines+markers',
            'name' => chart_helpers::humanize_label('mean_grade'),
            'line' => ['color' => $accent, 'width' => 2],
            'marker' => ['size' => 8, 'color' => $accent],
        ];

        return [
            'data' => $data,
            'layout' => [
                'title' => ['text' => 'Grade Distribution'],
                'template' => 'plotly',
                'xaxis' => ['showticklabels' => false, 'title' => null],
                'yaxis' => ['title' => ['text' => 'Grade']],
            ],
        ];
    }

    /**
     * A small, stable pseudo-random offset per key, in the range
     * [-$amplitude, $amplitude]. Seeded from each row's own key via an MD5
     * content hash (identical byte-for-byte to Python's hashlib.md5 for the
     * same string), so the same student gets the same offset on every
     * rerun and this matches the Python original exactly.
     */
    private static function deterministic_jitter(string $key, float $amplitude, string $salt): float {
        $digest = md5("{$salt}:{$key}");
        $fraction = hexdec(substr($digest, 0, 8)) / 0xFFFFFFFF;
        return ($fraction * 2 - 1) * $amplitude;
    }

    /**
     * Attempts-vs-grade scatter, keyed by quiz_name.
     *
     * @param array[] $attemptframe
     * @return array{plotly_json: array, correlation: float, y_label: string, title: string}|null
     */
    public static function build_scatter_figure(array $attemptframe, string $gradetype, bool $colorblindmode = false): ?array {
        if (empty($attemptframe)) {
            return null;
        }

        // Attempt_count per (quiz_name, student_id).
        $attemptcounts = [];
        foreach ($attemptframe as $r) {
            $key = $r['quiz_name'] . '|' . $r['student_id'];
            $attemptcounts[$key] = ($attemptcounts[$key] ?? 0) + 1;
        }

        // Grade_data per (quiz_name, student_id), per $gradetype.
        $gradesbykey = [];
        foreach ($attemptframe as $r) {
            $key = $r['quiz_name'] . '|' . $r['student_id'];
            $gradesbykey[$key][] = $r['overall_grade'];
        }

        if ($gradetype === 'Highest Grade') {
            $agg = fn($g) => max($g);
            $ylabel = 'Highest Grade';
            $title = 'Attempts vs Highest Grade';
        } else if ($gradetype === 'Minimum Grade') {
            $agg = fn($g) => min($g);
            $ylabel = 'Minimum Grade';
            $title = 'Attempts vs Minimum Grade';
        } else {
            $agg = fn($g) => stats::mean($g);
            $ylabel = 'Average Grade';
            $title = 'Attempts vs Average Grade';
        }

        $merged = [];
        foreach ($gradesbykey as $key => $grades) {
            [$quizname, $studentid] = explode('|', $key, 2);
            $merged[] = [
                'quiz_name' => $quizname,
                'student_id' => $studentid,
                'attempt_count' => $attemptcounts[$key],
                'overall_grade' => $agg($grades),
            ];
        }

        // Correlation from the true, unjittered values.
        $xs = array_map(fn($r) => (float) $r['attempt_count'], $merged);
        $ys = array_map(fn($r) => (float) $r['overall_grade'], $merged);
        $correlation = self::pearson_correlation($xs, $ys);

        // Group sizes for marker-size saturation, keyed on
        // (quiz_name, attempt_count, overall_grade) — matches the Python
        // groupby key exactly (pre-jitter coordinates).
        $groupsizes = [];
        foreach ($merged as $r) {
            $gkey = "{$r['quiz_name']}\x00{$r['attempt_count']}\x00{$r['overall_grade']}";
            $groupsizes[$gkey] = ($groupsizes[$gkey] ?? 0) + 1;
        }

        foreach ($merged as $i => $r) {
            $jitterkey = $r['quiz_name'] . '|' . $r['student_id'];
            $merged[$i]['attempt_count_plot'] = $r['attempt_count'] + self::deterministic_jitter($jitterkey, 0.15, 'x');
            $merged[$i]['overall_grade_plot'] = $r['overall_grade'] + self::deterministic_jitter($jitterkey, 0.15, 'y');
            $gkey = "{$r['quiz_name']}\x00{$r['attempt_count']}\x00{$r['overall_grade']}";
            $size = $groupsizes[$gkey];
            $saturation = (min($size, self::SCATTER_MARKER_SIZE_SATURATES_AT) - 1) / (self::SCATTER_MARKER_SIZE_SATURATES_AT - 1);
            $merged[$i]['_marker_size'] = self::SCATTER_MARKER_SIZE_MIN
                + $saturation * (self::SCATTER_MARKER_SIZE_MAX - self::SCATTER_MARKER_SIZE_MIN);
        }

        // $merged's per-quiz trace order follows its own first-appearance
        // order, which in turn follows $attemptframe's (see
        // compute_quiz_stats()'s identical note) — a course-wide chart's
        // quiz axis/legend should read chronologically, not alphabetically.
        // Within a single quiz's trace, points aren't additionally ordered
        // by student_id — a scatter trace's own point order has no visual
        // effect (it's an unordered cloud of markers).
        $quiznames = [];
        foreach ($merged as $r) {
            $quiznames[$r['quiz_name']] = true;
        }
        $quiznames = array_keys($quiznames);
        $palette = chart_helpers::qualitative_colors($colorblindmode, chart_helpers::PALETTE_SET2);

        $data = [];
        foreach ($quiznames as $i => $quiz) {
            $rows = array_values(array_filter($merged, fn($r) => $r['quiz_name'] === $quiz));
            $data[] = [
                'type' => 'scatter',
                'mode' => 'markers',
                'x' => array_map(fn($r) => $r['attempt_count_plot'], $rows),
                'y' => array_map(fn($r) => $r['overall_grade_plot'], $rows),
                'name' => (string) $quiz,
                'marker' => [
                    'size' => array_map(fn($r) => $r['_marker_size'], $rows),
                    'sizemode' => 'diameter',
                    'color' => $palette[$i % count($palette)],
                    'opacity' => 0.65,
                    'line' => ['width' => 1, 'color' => 'white'],
                ],
                'customdata' => array_map(fn($r) => [$r['attempt_count'], $r['overall_grade']], $rows),
                'hovertemplate' => "Attempts: %{customdata[0]}<br>{$ylabel}: %{customdata[1]}<extra></extra>",
            ];
        }

        return [
            'plotly_json' => [
                'data' => $data,
                'layout' => [
                    'title' => ['text' => $title],
                    'legend' => ['title' => ['text' => 'Quiz']],
                    'xaxis' => ['title' => ['text' => 'No. of Attempts'], 'tickmode' => 'linear', 'dtick' => 1],
                    'yaxis' => ['title' => ['text' => $ylabel]],
                ],
            ],
            'correlation' => $correlation,
            'y_label' => $ylabel,
            'title' => $title,
        ];
    }

    /**
     * Pearson correlation coefficient between $xs and $ys; NAN if either has zero variance.
     */
    private static function pearson_correlation(array $xs, array $ys): float {
        $n = count($xs);
        if ($n < 2) {
            return 0.0;
        }
        $meanx = stats::mean($xs);
        $meany = stats::mean($ys);
        $cov = 0.0;
        $varx = 0.0;
        $vary = 0.0;
        for ($i = 0; $i < $n; $i++) {
            $dx = $xs[$i] - $meanx;
            $dy = $ys[$i] - $meany;
            $cov += $dx * $dy;
            $varx += $dx * $dx;
            $vary += $dy * $dy;
        }
        if ($varx == 0.0 || $vary == 0.0) {
            return NAN;
        }
        return $cov / sqrt($varx * $vary);
    }

    /**
     * Per-quiz values for each selected metric, in quiz_name order, for the line-graph trend chart.
     *
     * @param array[] $attemptframe
     * @param string[] $selectedmetrics
     * @return array[] one row per quiz_name
     */
    public static function build_metric_trend_data(array $attemptframe, array $selectedmetrics): array {
        $byquiz = table_helpers::group_by($attemptframe, 'quiz_name');
        // Chronological (first-appearance) order — see compute_quiz_stats()'s
        // identical note.
        $quiznames = array_keys($byquiz);

        $out = [];
        foreach ($quiznames as $quiz) {
            $rows = $byquiz[$quiz];
            $entry = ['quiz_name' => $quiz];
            if (in_array('student_count', $selectedmetrics, true)) {
                $entry['student_count'] = count(array_unique(array_map(fn($r) => $r['student_id'], $rows)));
            }
            if (in_array('attempt_rate', $selectedmetrics, true)) {
                $counts = [];
                foreach ($rows as $r) {
                    $counts[$r['student_id']] = ($counts[$r['student_id']] ?? 0) + 1;
                }
                $entry['attempt_rate'] = stats::mean(array_values($counts));
            }
            if (in_array('mean_grade', $selectedmetrics, true)) {
                $entry['mean_grade'] = stats::mean(array_map(fn($r) => $r['overall_grade'], $rows));
            }
            if (in_array('grade_variance', $selectedmetrics, true)) {
                $entry['grade_variance'] = stats::sample_variance(array_map(fn($r) => $r['overall_grade'], $rows));
            }
            $out[] = $entry;
        }
        return $out;
    }

    /**
     * One line trace per selected metric across quizzes, for "Line Graph of Various Metrics".
     *
     * @param array[] $trenddata one row per quiz_name, metric fields as
     *        selected by build_metric_trend_data()
     */
    public static function build_line_graph_figure(array $trenddata, bool $colorblindmode = false): array {
        if (empty($trenddata)) {
            return ['data' => [], 'layout' => ['title' => ['text' => 'Line Graph of Various Metrics']]];
        }

        $metricnames = array_values(array_diff(array_keys($trenddata[0]), ['quiz_name']));
        $palette = chart_helpers::qualitative_colors($colorblindmode, chart_helpers::PALETTE_SET1);

        $wrappedlabels = array_map(fn($r) => self::wrap_category_label((string) $r['quiz_name']), $trenddata);

        $data = [];
        foreach ($metricnames as $i => $metric) {
            $data[] = [
                'type' => 'scatter',
                'mode' => 'lines+markers',
                'x' => $wrappedlabels,
                'y' => array_map(fn($r) => $r[$metric], $trenddata),
                'name' => chart_helpers::humanize_label($metric),
                'line' => ['color' => $palette[$i % count($palette)]],
                'marker' => ['color' => $palette[$i % count($palette)]],
            ];
        }

        $ncategories = max(count(array_unique(array_map(fn($r) => $r['quiz_name'], $trenddata))), 1);

        return [
            'data' => $data,
            'layout' => [
                'title' => ['text' => 'Line Graph of Various Metrics'],
                'template' => 'plotly',
                'legend' => ['title' => ['text' => 'Metric']],
                'xaxis' => ['type' => 'category', 'tickangle' => 0, 'tickfont' => ['size' => 10], 'title' => ['text' => 'Quiz']],
                'yaxis' => ['title' => ['text' => 'Value']],
                'width' => max(800, 220 * $ncategories),
            ],
        ];
    }

    /**
     * Per-quiz gaussian KDE (Scott's rule bandwidth — the same method
     * seaborn.kdeplot/scipy.stats.gaussian_kde use) of attempt start dates.
     * Returns null if there's nothing plottable: any row across every quiz
     * missing started_on, or no quiz ends up with a usable (2+ distinct
     * values) date series.
     *
     * @param array[] $attemptframe
     */
    public static function build_engagement_figure(array $attemptframe, bool $colorblindmode = false): ?array {
        if (empty($attemptframe)) {
            return null;
        }
        foreach ($attemptframe as $row) {
            if (empty($row['started_on'])) {
                return null;
            }
        }

        $quiznames = [];
        foreach ($attemptframe as $row) {
            $quiznames[$row['quiz_name']] = true;
        }
        $quiznames = array_keys($quiznames);
        $palette = chart_helpers::qualitative_colors($colorblindmode, chart_helpers::PALETTE_PLOTLY);

        $data = [];
        foreach ($quiznames as $i => $quiz) {
            $rows = array_values(array_filter($attemptframe, fn($r) => $r['quiz_name'] === $quiz));
            if (empty($rows)) {
                continue;
            }
            $datesnumeric = array_map(fn($r) => self::date_to_days((string) $r['started_on']), $rows);

            $kde = self::gaussian_kde_scott($datesnumeric);
            if ($kde === null) {
                continue;
            }
            [$mean, $variance] = $kde;

            $mind = min($datesnumeric);
            $maxd = max($datesnumeric);
            $ngrid = 200;
            $grid = [];
            for ($g = 0; $g < $ngrid; $g++) {
                $grid[] = $mind + ($maxd - $mind) * $g / ($ngrid - 1);
            }
            $density = array_map(fn($x) => self::kde_density($x, $datesnumeric, $variance), $grid);

            $data[] = [
                'type' => 'scatter',
                'mode' => 'lines',
                'x' => array_map(fn($d) => self::days_to_iso($d), $grid),
                'y' => $density,
                'name' => (string) $quiz,
                'fill' => 'tozeroy',
                'line' => ['color' => $palette[$i % count($palette)]],
            ];
        }

        if (empty($data)) {
            return null;
        }

        return [
            'data' => $data,
            'layout' => [
                'title' => ['text' => 'Engagement Over Time'],
                'xaxis' => ['title' => ['text' => 'Date']],
                'yaxis' => ['title' => ['text' => 'Frequency Density']],
            ],
        ];
    }

    /**
     * Scott's rule bandwidth (matching scipy.stats.gaussian_kde's default):
     * factor = n^(-1/5) for 1D data; covariance = sample_variance(data,
     * ddof=1) * factor^2. Returns null if the KDE is singular (fewer than 2
     * points, or zero variance) — scipy raises LinAlgError in that case,
     * which the Python original catches and skips that quiz's trace for.
     *
     * @param float[] $datesnumeric
     * @return array{0: float, 1: float}|null [mean, covariance]
     */
    private static function gaussian_kde_scott(array $datesnumeric): ?array {
        $n = count($datesnumeric);
        if ($n < 2) {
            return null;
        }
        $variance = stats::sample_variance($datesnumeric);
        if ($variance <= 0.0 || !is_finite($variance)) {
            return null;
        }
        $factor = $n ** (-1.0 / 5.0);
        $covariance = $variance * $factor * $factor;
        return [stats::mean($datesnumeric), $covariance];
    }

    /**
     * Gaussian kernel density estimate at $x given $datapoints and a precomputed bandwidth $covariance.
     *
     * @param float[] $datapoints
     */
    private static function kde_density(float $x, array $datapoints, float $covariance): float {
        $n = count($datapoints);
        $norm = 1.0 / sqrt(2 * M_PI * $covariance);
        $sum = 0.0;
        foreach ($datapoints as $xi) {
            $d = $x - $xi;
            $sum += $norm * exp(-($d * $d) / (2 * $covariance));
        }
        return $sum / $n;
    }

    /**
     * Days since the Unix epoch (any fixed epoch works: only relative
     * differences between values matter for KDE bandwidth/shape).
     */
    private static function date_to_days(string $datetimestr): float {
        $ts = strtotime($datetimestr);
        return $ts !== false ? $ts / 86400.0 : 0.0;
    }

    /**
     * Sub-second digits can differ from the Python original's own
     * num2date()-based labels by a few hundred milliseconds — an inherent
     * floating-point precision artifact of matplotlib's date2num() using an
     * epoch of year 1 (a much larger day-count integer part, leaving less
     * float64 precision for the time-of-day fraction) vs. this using the
     * Unix epoch. The underlying density curve these label is identical;
     * only the displayed hover instant can drift by well under a second.
     */
    private static function days_to_iso(float $days): string {
        $ts = (int) round($days * 86400.0);
        return gmdate('Y-m-d\TH:i:s', $ts) . '+00:00';
    }
}
