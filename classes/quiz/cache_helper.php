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
 * Cheap SQL fingerprints used to key the cache areas defined in db/caches.php,
 * so a cached result is only ever served while it still matches the current
 * DB state — new attempts, finished attempts, and regrades (which change
 * sumgrades) all naturally bust the cache; a manual per-question mark
 * override without a full regrade does not change sumgrades and so won't
 * bust the cache until the 1-hour TTL — an accepted gap for v1.
 *
 * Deliberately just COUNT/MAX(timefinish)/SUM(sumgrades) rather than hashing
 * every attempt's full response data: the whole point is for this check to
 * be dramatically cheaper than the thing it's guarding (get_response_records()
 * calls question_engine::load_questions_usage_by_activity() once per
 * attempt, which is the actual expensive step).
 *
 * @package local_quizanalytics
 * @copyright  2026 Ernest Ting <eting@caltech.edu>
 * @license    http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */

/**
 * Builds a cheap fingerprint of a set of quizzes' attempts to key the analysis result cache by.
 */
class local_quizanalytics_quiz_cache_helper {
    /**
     * Cheap per-quiz-set fingerprint (attempt count + latest timefinish + grade sum) used as a cache key.
     *
     * @param int[] $quizids
     * @return stdClass {count: int, fingerprint: string}
     */
    protected static function stats_for_quiz_ids(array $quizids): stdClass {
        global $DB;

        if (empty($quizids)) {
            return (object) ['count' => 0, 'fingerprint' => md5('empty')];
        }

        [$insql, $params] = $DB->get_in_or_equal($quizids);
        $row = $DB->get_record_sql(
            "SELECT COUNT(*) AS cnt, MAX(timefinish) AS maxfinish, SUM(sumgrades) AS sumgrades
               FROM {quiz_attempts}
              WHERE quiz $insql AND state = 'finished'",
            $params
        );

        $fingerprint = md5(
            ($row->cnt ?? 0) . '_' . ($row->maxfinish ?? '0') . '_' . ($row->sumgrades ?? '0')
        );

        return (object) ['count' => (int) $row->cnt, 'fingerprint' => $fingerprint];
    }

    /**
     * Fingerprint for one quiz's finished attempts — used by the
     * questionanalysis, solutionprocessmeta, and solutionprocess cache areas.
     *
     * @param stdClass $quiz row from mdl_quiz
     * @return stdClass {count: int, fingerprint: string}
     */
    public static function stats_for_quiz(stdClass $quiz): stdClass {
        return self::stats_for_quiz_ids([$quiz->id]);
    }

    /**
     * Fingerprint across every quiz in a set — used by the
     * quizanalysiscoursewide cache area, where the course-wide result
     * depends on the attempts of every STACK quiz in the course at once.
     *
     * @param stdClass[] $quizzes rows from mdl_quiz (or anything with ->id)
     * @return stdClass {count: int, fingerprint: string}
     */
    public static function stats_for_quizzes(array $quizzes): stdClass {
        return self::stats_for_quiz_ids(array_map(fn($q) => $q->id, $quizzes));
    }

    /** Return cheap attempt fingerprints for several quizzes in one query. */
    public static function stats_for_quizzes_by_id(array $quizzes): array {
        global $DB;
        $ids = array_values(array_unique(array_map(fn($q) => (int) $q->id, $quizzes)));
        if (empty($ids)) {
            return [];
        }
        [$insql, $params] = $DB->get_in_or_equal($ids, SQL_PARAMS_NAMED, 'quizstats');
        $rows = $DB->get_records_sql(
            "SELECT quiz, COUNT(*) AS cnt, MAX(timefinish) AS maxfinish, SUM(sumgrades) AS sumgrades
               FROM {quiz_attempts}
              WHERE quiz $insql AND state = 'finished'
           GROUP BY quiz",
            $params
        );
        $result = [];
        foreach ($ids as $id) {
            $row = $rows[$id] ?? (object) ['cnt' => 0, 'maxfinish' => 0, 'sumgrades' => 0];
            $result[$id] = (object) [
                'count' => (int) $row->cnt,
                'fingerprint' => md5(($row->cnt ?? 0) . '_' . ($row->maxfinish ?? '0') . '_' . ($row->sumgrades ?? '0')),
            ];
        }
        return $result;
    }

    /**
     * Builds a cache key safe for a `simplekeys => true` MUC cache area
     * (alphanumeric-only) from an arbitrary list of key parts.
     *
     * @param mixed ...$parts
     * @return string
     */
    public static function build_key(...$parts): string {
        return md5(implode('|', array_map(
            fn($p) => $p === null ? '' : (string) $p,
            $parts
        )));
    }

    /** Stable cache component for an ordered course-quiz selection. */
    public static function selection_key(array $quizids): string {
        return md5(implode(',', array_map('intval', array_values($quizids))));
    }
}
