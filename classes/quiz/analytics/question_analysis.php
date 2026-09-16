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
 * Assembles the simplified Question Analytics {summary, snapshot, sections,
 * questions, audit} payload — a single Question Response Overview chart plus
 * a versioned per-question drill-down, backed by a compact Moodle-native
 * "quiz snapshot" (see local_quizanalytics_quiz_data_fetcher::
 * get_quiz_snapshot()) instead of this plugin's own self-computed difficulty/
 * metrics tables.
 *
 * @package local_quizanalytics
 * @copyright  2026 Ernest Ting <eting@caltech.edu>
 * @license    http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */

namespace local_quizanalytics\quiz\analytics;

/**
 * Assembles the simplified Question Analytics {summary, snapshot, sections, questions, audit} payload for one quiz.
 */
class question_analysis {
    /**
     * Remove the expensive per-variant detail from the lecturer-facing
     * payload. The complete result remains in the questionanalysis MUC cache
     * and is retrieved by questionreview.php only for the selected variant.
     *
     * @param array $result
     * @param string $reviewurl
     * @return array
     */
    public static function to_lightweight_review(array $result, string $reviewurl): array {
        foreach ($result['questions'] ?? [] as &$detail) {
            foreach ($detail['versions'] ?? [] as &$version) {
                unset($version['question_text_html'], $version['question_text_raw'], $version['question_text']);
                unset($version['common_responses']);
            }
            unset($version);
        }
        unset($detail);
        $result['question_review_url'] = $reviewurl;
        return $result;
    }

    /**
     * Builds the individual Question Analytics payload: response overview
     * and per-question drill-down.
     *
     * @param array[] $records as returned by
     *        local_quizanalytics_quiz_data_fetcher::get_response_records_for_quiz()
     * @param string $quizname
     * @param bool $colorblindmode
     * @param bool $anonymize
     * @param array|null $snapshot Moodle-backed quiz snapshot from
     *        local_quizanalytics_quiz_data_fetcher::get_quiz_snapshot()
     * @return array {quiz_name, summary, snapshot, sections, questions, audit}
     */
    public static function build_analysis(
        array $records,
        string $quizname,
        bool $colorblindmode = false,
        bool $anonymize = false,
        ?array $snapshot = null,
        ?callable $progresscallback = null
    ): array {
        $responserows = parser::build_response_rows($records, $quizname, $anonymize);

        $pools = parser::get_attempt_pools($responserows);
        $poolb = $pools['pool_b'];

        $questionmetricsrows = question_metrics::compute_question_metrics($responserows);
        $snapshotquestions = array_column($snapshot['students_per_question'] ?? [], 'question');
        $questionstudents = [];
        foreach ($snapshot['students_per_question'] ?? [] as $questionstudent) {
            $questionstudents[$questionstudent['question']] = (int) $questionstudent['students'];
        }
        $responseoverviewrows = response_analysis::compute_response_statuses(
            $responserows,
            $snapshotquestions,
            $questionstudents,
            $snapshot['question_means'] ?? [],
            $snapshot['question_mean_counts'] ?? []
        );

        $sections = [];
        $questionorder = array_map(fn($r) => $r['question'], $questionmetricsrows);

        $facilityrows = array_values(array_filter(
            $snapshot['question_facilities'] ?? [],
            fn($row) => array_key_exists('facility_index', $row)
        ));
        if (!empty($facilityrows)) {
            $sections[] = [
                'id' => 'facility-index',
                'title' => 'Facility Index',
                'caption' => 'Moodle\'s official measure of how easy each question was in practice. Low values highlight questions that may need review; dashed lines mark 30% and 70%.',
                'documentation_link' => [
                    'url' => 'https://docs.moodle.org/404/en/mod/quiz/statistics',
                    'label' => 'Read Moodle\'s Facility Index documentation',
                ],
                'charts' => [[
                    'id' => 'facility-index-fig',
                    'title' => null,
                    'plotly_json' => question_charts::build_facility_index_figure($facilityrows, $colorblindmode),
                ]],
            ];
        }

        // Question Response Overview — replaces the old Question Difficulty
        // Analysis / Question Response Distribution / Student Performance
        // Matrix / Question Metrics sections with a single chart, matching
        // this simplified page's own reduced scope.
        $responsecharts = [];
        if (!empty($responseoverviewrows)) {
            $responsecharts[] = [
                'id' => 'response-status', 'title' => null,
                'plotly_json' => question_charts::build_response_status_figure($responseoverviewrows, $colorblindmode),
            ];
        }
        $sections[] = [
            'id' => 'question-response-overview',
            'title' => 'Question Response Overview',
            'caption' => 'Shows how students responded to each question. Select a question to inspect the responses ' .
                'in more detail.',
            'charts' => $responsecharts,
        ];

        $totalquestions = count($questionorder);
        if ($progresscallback !== null) {
            $progresscallback(0, $totalquestions, '');
        }

        // Per-question detail (drives the PHP question <select>), grouped by
        // instantiated STACK question "version" so randomized variants don't
        // mix their expected answers/wrong-response lists together.
        $questions = [];
        foreach ($questionorder as $questionindex => $q) {
            $versions = question_details::build_versioned_review($poolb, $q);
            foreach ($versions as &$version) {
                // Debug-dump detection stays on the raw (pre-format_text) HTML,
                // matching where this check always ran — format_text()'s HTML
                // purification runs after, on whatever's left once a leaked dump
                // is split off.
                [$cleanraw, ] = latex_utils::split_stack_debug_dump(
                    $version['question_text_raw'] !== '' ? $version['question_text_raw'] : $version['question_text']
                );
                $version['question_text_html'] = latex_utils::render_question_html($cleanraw);
                $version['right_answer_html'] = latex_utils::extract_stack_answer_latex($version['right_answer_text']);
                foreach ($version['common_responses'] as &$commonresponse) {
                    $commonresponse['response'] = latex_utils::extract_stack_answer_latex(
                        (string) $commonresponse['response']
                    );
                }
                unset($commonresponse);
                unset($version['question_text'], $version['question_text_raw'], $version['right_answer_text']);
            }
            unset($version);
            $questions[$q] = [
                'versions' => $versions,
            ];
            if ($progresscallback !== null) {
                $progresscallback($questionindex + 1, $totalquestions, $q);
            }
        }

        return [
            'quiz_name' => $quizname,
            'summary' => [],
            'snapshot' => $snapshot,
            'sections' => $sections,
            'questions' => $questions,
            'audit' => null,
        ];
    }
}
