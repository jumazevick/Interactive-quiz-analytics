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
 * PHP port of analytics-service/analytics/question_details.py.
 *
 * @package local_quizanalytics
 * @copyright  2026 Ernest Ting <eting@caltech.edu>
 * @license    http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */

namespace local_quizanalytics\quiz\analytics;

/**
 * Per-question drill-down: question text, right answer, and wrong-response breakdown.
 */
class question_details {
    /** @var string Placeholder shown when a field wasn't captured for a given attempt. */
    const NOT_AVAILABLE = 'Not available in this export';

    /** @var string Placeholder shown for a genuinely blank (never-answered) response, in place of Moodle's raw internal response summary. */
    const NO_RESPONSE = '(No response)';

    /**
     * Group best-attempt rows by the instantiated STACK question and expected
     * answer, so randomized values are never mixed into one error list.
     *
     * @param array[] $poolbrows
     * @param string $question
     * @return array[]
     */
    public static function build_versioned_review(array $poolbrows, string $question): array {
        $groups = [];
        foreach ($poolbrows as $row) {
            if ($row['question'] !== $question) {
                continue;
            }
            $questiontext = (string) ($row['question_text'] ?? '');
            $questiontextraw = (string) ($row['question_text_raw'] ?? '');
            $rightanswer = (string) ($row['right_answer_text'] ?? '');
            $key = sha1($questiontext . "\0" . $rightanswer);
            if (!isset($groups[$key])) {
                $groups[$key] = [
                    'question_text' => $questiontext,
                    'question_text_raw' => $questiontextraw,
                    'right_answer_text' => $rightanswer,
                    'students' => [],
                    'student_statuses' => [],
                    'sample_attempt_id' => (int) ($row['attempt_id'] ?? 0),
                    'cmid' => (int) ($row['cmid'] ?? 0),
                    'wrong' => [],
                    'status_counts' => [
                        'correct' => 0,
                        'incorrect' => 0,
                        'invalid' => 0,
                        'noresponse' => 0,
                    ],
                ];
            }
            $studentid = (string) ($row['student_id'] ?? '');
            $groups[$key]['students'][$studentid] = true;

            // Pool B is already the best overall quiz attempt per student.
            // Keep a second guard here so each student's status contributes
            // once per displayed variant even if an upstream export contains
            // duplicate rows.
            if (!isset($groups[$key]['student_statuses'][$studentid])) {
                $status = (string) ($row['response_status'] ?? '');
                if (!in_array($status, ['correct', 'incorrect', 'invalid'], true)) {
                    $status = 'noresponse';
                }
                $groups[$key]['student_statuses'][$studentid] = $status;
                $groups[$key]['status_counts'][$status]++;
            }

            if (($row['grade'] ?? null) !== null && (float) $row['grade'] < 1.0) {
                // A genuinely blank response (no ansN: field at all, so
                // parser::build_response_rows() never had an expression to
                // parse) still carries Moodle's own raw response summary in
                // response_text — internal bookkeeping like "Seed: 123456;
                // prt1: !", not anything the student actually typed. Shown
                // as-is that read like a bizarre "common wrong answer";
                // swapped for a plain, honest placeholder instead. A
                // response_status of 'invalid' is left alone here — that's
                // real (if malformed) student input, worth showing.
                $response = ($row['response_status'] ?? '') === 'blank'
                    ? self::NO_RESPONSE
                    : trim((string) ($row['response_text'] ?? ''));
                if ($response !== '') {
                    if (!isset($groups[$key]['wrong'][$response])) {
                        $groups[$key]['wrong'][$response] = [
                            'students' => 0,
                            'sample_attempt_id' => (int) ($row['attempt_id'] ?? 0),
                            'cmid' => (int) ($row['cmid'] ?? 0),
                        ];
                    }
                    $groups[$key]['wrong'][$response]['students']++;
                }
            }
        }

        $versions = [];
        $versionnumber = 1;
        foreach ($groups as $group) {
            uasort($group['wrong'], static function (array $a, array $b): int {
                return $b['students'] <=> $a['students'];
            });
            $common = [];
            foreach ($group['wrong'] as $response => $data) {
                $reviewurl = (new \moodle_url('/mod/quiz/review.php', [
                    'attempt' => $data['sample_attempt_id'],
                    'cmid' => $data['cmid'],
                ]))->out(false);
                $common[] = [
                    'response' => $response,
                    'students' => (int) $data['students'],
                    'review_url' => $reviewurl,
                ];
            }
            $students = count($group['students']);
            $statuscounts = $group['status_counts'];
            $statustotal = array_sum($statuscounts);
            if ($statustotal !== $students) {
                throw new \coding_exception('Question Review response-status counts do not match unique students.');
            }
            $notcorrect = $statuscounts['incorrect'] + $statuscounts['invalid'] + $statuscounts['noresponse'];
            $versions[] = [
                    'label' => 'Variant ' . $versionnumber++,
                'students' => $students,
                'correct' => $statuscounts['correct'],
                'incorrect' => $statuscounts['incorrect'],
                'invalid' => $statuscounts['invalid'],
                'noresponse' => $statuscounts['noresponse'],
                'not_correct' => $notcorrect,
                'not_correct_percent' => $students > 0 ? $notcorrect / $students * 100.0 : 0.0,
                'question_text' => $group['question_text'],
                'question_text_raw' => $group['question_text_raw'],
                'right_answer_text' => $group['right_answer_text'],
                'common_responses' => $common,
                'review_url' => (new \moodle_url('/mod/quiz/review.php', [
                    'attempt' => $group['sample_attempt_id'],
                    'cmid' => $group['cmid'],
                ]))->out(false),
            ];
        }
        return $versions;
    }

    /**
     * Question text / right answer for a question, taken from Pool B (Best
     * Attempt per Student).
     *
     * @param array[] $poolbrows
     * @return array{question_text: string, right_answer_text: string}
     */
    public static function build_question_detail(array $poolbrows, string $question): array {
        $rows = array_values(array_filter($poolbrows, fn($r) => $r['question'] === $question));

        $firstnonempty = function (string $field) use ($rows): string {
            foreach ($rows as $r) {
                $v = $r[$field] ?? null;
                if (is_string($v) && trim($v) !== '') {
                    return $v;
                }
            }
            return self::NOT_AVAILABLE;
        };

        return [
            'question_text' => $firstnonempty('question_text'),
            'right_answer_text' => $firstnonempty('right_answer_text'),
        ];
    }

    /**
     * Best-attempt rows for a question where the student didn't get full
     * credit — lets a teacher scan submitted responses against the right
     * answer to spot common misconceptions.
     *
     * @param array[] $poolbrows
     * @return array{columns: string[], rows: array[]}
     */
    public static function build_error_drilldown(array $poolbrows, string $question): array {
        $columns = ['Student Name', 'Email', 'Submitted Response', 'Right Answer', 'Score', 'Status'];

        $wrong = array_values(array_filter(
            $poolbrows,
            fn($r) => $r['question'] === $question && $r['grade'] !== null && $r['grade'] < 1.0
        ));

        $rows = array_map(fn($r) => [
            $r['student_name'] ?? '',
            $r['student_id'] ?? '',
            $r['response_text'] ?? '',
            $r['right_answer_text'] ?? '',
            $r['grade'] ?? null,
            $r['response_status'] ?? '',
        ], $wrong);

        $common = [];
        foreach ($wrong as $row) {
            $response = trim((string) ($row['response_text'] ?? ''));
            if ($response === '') {
                continue;
            }
            $common[$response] = ($common[$response] ?? 0) + 1;
        }
        arsort($common, SORT_NUMERIC);
        $commonresponses = [];
        foreach ($common as $response => $count) {
            $commonresponses[] = ['response' => $response, 'students' => (int) $count];
        }

        return [
            'columns' => $columns,
            'rows' => $rows,
            'common_responses' => $commonresponses,
        ];
    }
}
