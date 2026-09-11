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
        array $quizmetadata = []
    ): array {
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

        $attemptframe = quiz_metrics::build_quiz_attempt_frame($combined);

        $selectedstats = !empty($selectedstats) ? $selectedstats : self::DEFAULT_QUIZ_STATS;
        $selectedmetrics = !empty($selectedmetrics) ? $selectedmetrics : self::DEFAULT_QUIZ_METRICS;

        $statsrows = quiz_metrics::compute_quiz_stats($attemptframe, $selectedstats, $quizmetadata);

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
            'caption' => 'Aggregated statistics per quiz, combined across the course.',
            'column_help' => [
                'grade_variance' => 'Shows how spread out the grades are: a higher value means students received more different grades.',
                'mean_highest_grade' => 'The average of each student\'s best grade on this quiz.',
                'attempt_count' => 'The total number of attempts submitted by all students.',
                'attempt_rate' => 'The average number of attempts per student: total attempts divided by students with attempts.',
                'facility_index' => 'The average percentage of maximum marks earned across the quiz\'s questions, using Moodle\'s per-question values.',
            ],
            'table' => table_helpers::to_table($statsrows),
        ];

        $boxfig = quiz_metrics::build_boxplot_figure($attemptframe, $colorblindmode);
        $sections[] = [
            'id' => 'boxplot',
            'title' => 'Quiz Grade Distribution (Box Plot)',
            'caption' => 'Spread of grades per quiz, with mean grade overlay.',
            'charts' => [['id' => 'boxplot-fig', 'title' => null, 'plotly_json' => $boxfig]],
        ];

        $engagementfig = quiz_metrics::build_engagement_figure($attemptframe, $colorblindmode);
        if ($engagementfig !== null) {
            $sections[] = [
                'id' => 'engagement',
                'title' => 'Engagement Over Time',
                'caption' => 'Density of quiz attempt start times per quiz, combined across the course.',
                'charts' => [['id' => 'engagement-fig', 'title' => null, 'plotly_json' => $engagementfig]],
            ];
        }

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

        return [
            'summary' => ['course_name' => $coursename],
            'sections' => $sections,
        ];
    }
}
