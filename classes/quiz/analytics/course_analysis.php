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
 * Assembles the Course-Wide Analytics payload — the PHP port's equivalent
 * of analytics-service's app.py::analyze_course() (POST /analyze-course)
 * route.
 *
 * @package local_quizanalytics
 * @copyright  2026 Ernest Ting <eting@caltech.edu>
 * @license    http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */

namespace local_quizanalytics\quiz\analytics;

/**
 * Assembles the Course-Wide Analytics {summary, sections} payload across every STACK quiz in a course.
 */
class course_analysis {
    /** @var string[] Default "Summary of Quiz Stats" columns when the teacher hasn't picked any. */
    const DEFAULT_QUIZ_STATS = [
        'student_count', 'attempt_rate', 'mean_grade', 'grade_variance', 'mean_highest_grade', 'attempt_count',
    ];

    /** @var string[] Default "Line Graph of Various Metrics" series when the teacher hasn't picked any. */
    const DEFAULT_QUIZ_METRICS = ['student_count', 'attempt_rate', 'mean_grade', 'grade_variance'];

    /** @var string Default grade type for the Attempts vs Grades scatter plot. */
    const DEFAULT_GRADE_TYPE = 'Average Grade';

    /**
     * Builds the full Course-Wide Analytics payload: attempt list, quiz stats, boxplot,
     * engagement, scatter, and metric-trend sections across every quiz passed in.
     *
     * @param array<string, array[]> $quizzes quiz_name => records[]
     * @param array<string, array{facility_index?: float|null, quiz_url?: string}> $quizmetadata
     * @return array{summary: array, sections: array[]}
     */
    public static function build_analysis(
        string $coursename,
        array $quizzes,
        bool $colorblindmode = false,
        ?array $selectedstats = null,
        ?array $selectedmetrics = null,
        string $gradetype = self::DEFAULT_GRADE_TYPE,
        bool $anonymize = false,
        array $quizmetadata = [],
        ?callable $progresscallback = null,
        ?array $preparedframes = null
    ): array {
        $reporttiming = function(string $metric, float $started) use ($progresscallback): void {
            if ($progresscallback !== null) {
                $progresscallback($metric, microtime(true) - $started);
            }
        };
        if ($preparedframes !== null) {
            $attemptframe = [];
            foreach ($preparedframes as $frame) {
                if (!empty($frame)) {
                    $attemptframe = array_merge($attemptframe, $frame);
                }
            }
        } else {
            $started = microtime(true);
            $combined = [];
            foreach ($quizzes as $quizname => $records) {
                if (empty($records)) {
                    continue;
                }
                $rows = parser::build_response_rows($records, $quizname, $anonymize);
                if (!empty($rows)) {
                    $combined = array_merge($combined, $rows);
                }
            }

            if (empty($combined)) {
                throw new \InvalidArgumentException('No gradable attempts parsed for any quiz.');
            }
            $reporttiming('Parse response records', $started);

            $started = microtime(true);
            $attemptframe = quiz_metrics::build_quiz_attempt_frame($combined);
            $reporttiming('Build attempt frame', $started);
        }

        if (empty($attemptframe)) {
            throw new \InvalidArgumentException('No prepared gradable attempts available for any quiz.');
        }

        $selectedstats = !empty($selectedstats) ? $selectedstats : self::DEFAULT_QUIZ_STATS;
        $selectedmetrics = !empty($selectedmetrics) ? $selectedmetrics : self::DEFAULT_QUIZ_METRICS;

        $started = microtime(true);
        $statsrows = quiz_metrics::compute_quiz_stats($attemptframe, $selectedstats, $quizmetadata);
        $reporttiming('Compute quiz statistics', $started);

        $sections = [];

        $sections[] = [
            'id' => 'attempt-list',
            'title' => 'Student Quiz Summary',
            'caption' => 'Summary of each student\'s attempts and grades for every STACK quiz in the course.',
            'column_help' => [
                'first_grade' => 'The grade from the student\'s earliest recorded attempt, ordered by completion time.',
                'latest_grade' => 'The grade from the student\'s most recent recorded attempt, ordered by completion time.',
            ],
            'table' => table_helpers::to_table(
                quiz_metrics::build_student_quiz_summary($attemptframe)
            ),
        ];

        $sections[] = [
            'id' => 'quiz-stats',
            'title' => 'Summary of Quiz Stats',
            'caption_html' => 'This table provides a course-wide summary of the selected STACK quizzes. '
                . 'Grade and attempt measures are calculated from processed finished quiz attempts. '
                . 'Mean Question Facility Index is calculated by averaging Moodle\'s question-level '
                . 'Facility Index values for the STACK questions in each quiz. '
                . 'Click a quiz name to inspect its Question Analytics in more detail. '
                . html_writer::link(
                    'https://phpdoc.moodledev.io/4.5/d2/dc2/classquiz__statistics__report.html',
                    'Moodle Quiz Statistics',
                    ['target' => '_blank', 'rel' => 'noopener']
                ) . '.',
            'column_help' => [
                'mean_grade' => 'The average raw mark across processed finished attempts for this quiz. If a student has several finished attempts, each attempt contributes separately.',
                'grade_variance' => 'The sample variance of the raw marks across processed finished attempts. A higher value means the attempt marks are more spread out around the mean.',
                'mean_highest_grade' => 'For each student, the plugin takes their highest processed finished attempt mark for the quiz and then averages these best marks across students. This is calculated independently of Moodle\'s configured quiz grading method.',
                'student_count' => 'The number of unique students represented by the processed finished attempts for this quiz.',
                'attempt_count' => 'The number of processed finished, non-preview attempts included in the analytics.',
                'attempt_rate' => 'The average number of processed finished attempts per student. Attempt Rate = No. of Attempts ÷ Student Count.',
                'facility_index' => 'Mean Question Facility Index = (FI₁ + FI₂ + ... + FIₙ) / n, where FIᵢ is Moodle\'s Facility Index for question i. Moodle calculates Facility Index separately for each question; the plugin takes the simple arithmetic mean of the non-null STACK question Facility Index values in that quiz. These values come from Moodle Quiz Statistics. This is not a Moodle-provided quiz-level Facility Index.',
            ],
            'table' => table_helpers::to_table($statsrows),
        ];

        $started = microtime(true);
        $boxfig = quiz_metrics::build_boxplot_figure($attemptframe, $colorblindmode);
        $reporttiming('Build grade distribution', $started);
        $sections[] = [
            'id' => 'boxplot',
            'title' => 'Quiz Grade Distribution (Box Plot)',
            'caption' => 'Spread of grades per quiz, with mean grade overlay.',
            'charts' => [['id' => 'boxplot-fig', 'title' => null, 'plotly_json' => $boxfig]],
        ];

        $started = microtime(true);
        $engagementfig = quiz_metrics::build_engagement_figure($attemptframe, $colorblindmode);
        $reporttiming('Build engagement timeline', $started);
        if ($engagementfig !== null) {
            $sections[] = [
                'id' => 'engagement',
                'title' => 'Engagement Over Time',
                'caption' => 'Density of quiz attempt start times per quiz, combined across the course.',
                'charts' => [['id' => 'engagement-fig', 'title' => null, 'plotly_json' => $engagementfig]],
            ];
        }

        $started = microtime(true);
        $scattervariants = [];
        foreach (['Highest Grade', 'Average Grade', 'Minimum Grade'] as $scattertype) {
            $scatterresult = quiz_metrics::build_scatter_figure($attemptframe, $scattertype, $colorblindmode);
            if ($scatterresult !== null) {
                $correlationstr = is_nan($scatterresult['correlation']) ? 'nan' : sprintf('%.2f', $scatterresult['correlation']);
                $scattervariants[$scattertype] = [
                    'label' => $scatterresult['y_label'],
                    'caption' => "Correlation between number of attempts and quiz {$scatterresult['y_label']}: r = {$correlationstr}",
                    'plotly_json' => $scatterresult['plotly_json'],
                ];
            }
        }
        if (!empty($scattervariants)) {
            $selectedtype = array_key_exists($gradetype, $scattervariants) ? $gradetype : self::DEFAULT_GRADE_TYPE;
            $scatterresult = $scattervariants[$selectedtype];
            $sections[] = [
                'id' => 'scatter',
                'title' => 'Scatter Plot: Attempts vs Grades',
                'caption' => $scatterresult['caption'],
                'scatter_variants' => $scattervariants,
                'selected_scatter_type' => $selectedtype,
                'charts' => [['id' => 'scatter-fig', 'title' => null, 'plotly_json' => $scatterresult['plotly_json']]],
            ];
        }
        $reporttiming('Build attempts-versus-grades metrics', $started);

        $started = microtime(true);
        $trenddata = quiz_metrics::build_metric_trend_data($attemptframe, $selectedmetrics);
        if (!empty($trenddata)) {
            $trendfig = quiz_metrics::build_line_graph_figure($trenddata, $colorblindmode);
            $sections[] = [
                'id' => 'trend',
            'title' => 'Line Graph of Various Metrics',
                'caption' => 'Trend of selected metrics across quizzes.',
                'table' => table_helpers::to_table($trenddata),
                'charts' => [['id' => 'trend-fig', 'title' => null, 'plotly_json' => $trendfig]],
            ];
        }
        $reporttiming('Build metric trends', $started);

        return [
            'summary' => ['course_name' => $coursename],
            'sections' => $sections,
        ];
    }
}
