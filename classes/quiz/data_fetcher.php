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
 * All database access for the Quiz Analytics half of local_quizanalytics:
 * detecting which quizzes in a
 * course (or a specific quiz) contain qtype_stack questions, and reading
 * finished attempts + responses straight out of the database — reconstructing
 * the same row shape Moodle's own "Responses report" CSV export produces
 * (matching what analytics/parser.py::build_response_rows already expects),
 * so the analytics engine itself needs no changes, only its input source.
 *
 * @package local_quizanalytics
 * @copyright  2026 Ernest Ting <eting@caltech.edu>
 * @license    http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */

defined('MOODLE_INTERNAL') || die();

require_once($CFG->dirroot . '/mod/quiz/locallib.php');
require_once($CFG->dirroot . '/mod/quiz/report/statistics/report.php');
require_once($CFG->dirroot . '/question/engine/lib.php');
require_once($CFG->dirroot . '/local/quizanalytics/classes/quiz/cache_helper.php');

/**
 * Fetches quiz/course attempt records from Moodle's own DB tables for the analytics package to consume.
 */
class local_quizanalytics_quiz_data_fetcher {
    /**
     * How many attempts' question usages get batch-loaded into memory at
     * once inside get_response_records_for_quiz(), rather than the whole
     * quiz's attempts in one go. Purely a memory/round-trip tradeoff, not a
     * correctness one — see that method's own comment for the measurement
     * behind this number.
     */
    const ATTEMPT_BATCH_SIZE = 250;

    /**
     * Cheap existence check: does this course contain at least one quiz with
     * at least one qtype_stack question in one of its slots?
     *
     * Only matches questions added directly to a slot (the normal way STACK
     * questions are added to a quiz). Quizzes that pull STACK questions in
     * only via "random question from category" slots are not detected here —
     * that's a deliberately narrow scope for a fast nav-gating check, not a
     * full quiz-structure walk.
     *
     * @param int $courseid
     * @return bool
     */
    public static function course_has_stack_quiz(int $courseid): bool {
        global $DB;

        $sql = "SELECT 1
                  FROM {quiz} quiz
                  JOIN {course_modules} cm ON cm.instance = quiz.id
                  JOIN {modules} m ON m.id = cm.module AND m.name = 'quiz'
                  JOIN {context} ctx ON ctx.contextlevel = :contextmodule AND ctx.instanceid = cm.id
                  JOIN {quiz_slots} slot ON slot.quizid = quiz.id
                  JOIN {question_references} qr ON qr.usingcontextid = ctx.id
                                                AND qr.component = 'mod_quiz'
                                                AND qr.questionarea = 'slot'
                                                AND qr.itemid = slot.id
                  JOIN {question_bank_entries} qbe ON qbe.id = qr.questionbankentryid
                  JOIN {question_versions} qv ON qv.questionbankentryid = qbe.id
                  JOIN {question} q ON q.id = qv.questionid AND q.qtype = 'stack'
                 WHERE quiz.course = :courseid";

        return $DB->record_exists_sql($sql, [
            'contextmodule' => CONTEXT_MODULE,
            'courseid'      => $courseid,
        ]);
    }

    /**
     * Same matching rule as course_has_stack_quiz(), narrowed to one quiz —
     * used to gate the Analytics link this plugin adds to a quiz's own
     * settings/administration menu (see lib.php's
     * local_quizanalytics_extend_settings_navigation()), which runs on every
     * quiz page view so needs to stay just as cheap.
     *
     * @param int $quizid
     * @return bool
     */
    public static function quiz_has_stack_question(int $quizid): bool {
        global $DB;

        $sql = "SELECT 1
                  FROM {quiz} quiz
                  JOIN {course_modules} cm ON cm.instance = quiz.id
                  JOIN {modules} m ON m.id = cm.module AND m.name = 'quiz'
                  JOIN {context} ctx ON ctx.contextlevel = :contextmodule AND ctx.instanceid = cm.id
                  JOIN {quiz_slots} slot ON slot.quizid = quiz.id
                  JOIN {question_references} qr ON qr.usingcontextid = ctx.id
                                                AND qr.component = 'mod_quiz'
                                                AND qr.questionarea = 'slot'
                                                AND qr.itemid = slot.id
                  JOIN {question_bank_entries} qbe ON qbe.id = qr.questionbankentryid
                  JOIN {question_versions} qv ON qv.questionbankentryid = qbe.id
                  JOIN {question} q ON q.id = qv.questionid AND q.qtype = 'stack'
                 WHERE quiz.id = :quizid";

        return $DB->record_exists_sql($sql, [
            'contextmodule' => CONTEXT_MODULE,
            'quizid'        => $quizid,
        ]);
    }

    /**
     * Lists every quiz in the course that contains at least one qtype_stack
     * question (same matching rule as course_has_stack_quiz()), for the quiz
     * selector on index.php.
     *
     * @param int $courseid
     * @return stdClass[] quiz records (id, name, course), keyed by quiz id
     */
    public static function get_course_stack_quizzes(int $courseid): array {
        global $DB;

        // Ordered chronologically (quiz.timeopen when the teacher set one,
        // falling back to when the activity was added to the course
        // otherwise) rather than alphabetically — a course-wide chart
        // plotting one point per quiz reads as a timeline, so "Quiz 10"
        // sorting before "Quiz 2" left it visibly out of order. Both
        // columns feeding the COALESCE have to appear in the SELECT list
        // (aliased) since this is a SELECT DISTINCT and Postgres requires
        // every ORDER BY expression to be one of the selected columns.
        $sql = "SELECT DISTINCT quiz.id, quiz.name, quiz.course, quiz.sumgrades, quiz.grade,
                       COALESCE(NULLIF(quiz.timeopen, 0), cm.added) AS chronoorder
                  FROM {quiz} quiz
                  JOIN {course_modules} cm ON cm.instance = quiz.id
                  JOIN {modules} m ON m.id = cm.module AND m.name = 'quiz'
                  JOIN {context} ctx ON ctx.contextlevel = :contextmodule AND ctx.instanceid = cm.id
                  JOIN {quiz_slots} slot ON slot.quizid = quiz.id
                  JOIN {question_references} qr ON qr.usingcontextid = ctx.id
                                                AND qr.component = 'mod_quiz'
                                                AND qr.questionarea = 'slot'
                                                AND qr.itemid = slot.id
                  JOIN {question_bank_entries} qbe ON qbe.id = qr.questionbankentryid
                  JOIN {question_versions} qv ON qv.questionbankentryid = qbe.id
                  JOIN {question} q ON q.id = qv.questionid AND q.qtype = 'stack'
                 WHERE quiz.course = :courseid
              ORDER BY chronoorder, quiz.name";

        return $DB->get_records_sql($sql, [
            'contextmodule' => CONTEXT_MODULE,
            'courseid'      => $courseid,
        ]);
    }

    /**
     * Every course id containing at least one STACK quiz (same matching
     * rule as course_has_stack_quiz()), site-wide — used by the cache-
     * warming scheduled task to find which courses to precompute for. Unlike
     * stack_course_helper::get_viewable_courses(), this isn't scoped to any
     * particular user's own enrolments/capabilities, since a scheduled task
     * has no such user to check against.
     *
     * @return int[] course ids
     */
    public static function get_courses_with_stack_quizzes(): array {
        global $DB;

        $sql = "SELECT DISTINCT quiz.course
                  FROM {quiz} quiz
                  JOIN {course_modules} cm ON cm.instance = quiz.id
                  JOIN {modules} m ON m.id = cm.module AND m.name = 'quiz'
                  JOIN {context} ctx ON ctx.contextlevel = :contextmodule AND ctx.instanceid = cm.id
                  JOIN {quiz_slots} slot ON slot.quizid = quiz.id
                  JOIN {question_references} qr ON qr.usingcontextid = ctx.id
                                                AND qr.component = 'mod_quiz'
                                                AND qr.questionarea = 'slot'
                                                AND qr.itemid = slot.id
                  JOIN {question_bank_entries} qbe ON qbe.id = qr.questionbankentryid
                  JOIN {question_versions} qv ON qv.questionbankentryid = qbe.id
                  JOIN {question} q ON q.id = qv.questionid AND q.qtype = 'stack'";

        $courseids = $DB->get_records_sql($sql, ['contextmodule' => CONTEXT_MODULE]);
        return array_map('intval', array_keys($courseids));
    }

    /**
     * Response records for a single quiz — one row per finished attempt,
     * with one set of question_N_text, response_N, right_answer_N (etc.)
     * columns per slot in that attempt.
     *
     * @param stdClass $quiz   row from mdl_quiz
     * @param stdClass $course course record
     * @param bool $includeinprogress include Moodle's in-progress attempts,
     *        not just finished ones — see get_quiz_snapshot() for the
     *        compact summary this plugin already reads straight from SQL
     *        for in-progress/finished counts; this param is only about
     *        whether the full per-attempt response rows built here also
     *        cover in-progress attempts.
     * @return array
     */
    public static function get_response_records_for_quiz(
        stdClass $quiz,
        stdClass $course,
        bool $includeinprogress = false
    ): array {
        global $DB;

        $cm = get_coursemodule_from_instance('quiz', $quiz->id, $course->id, false, MUST_EXIST);
        $records = [];

        // Only finished attempts by default — matches what a teacher would
        // get from the "Responses" report with the default "Finished" state
        // filter. Pass $includeinprogress to also cover in-progress/overdue
        // attempts.
        if ($includeinprogress) {
            $attempts = $DB->get_records_select(
                'quiz_attempts',
                "quiz = :quizid AND state IN ('finished', 'inprogress')",
                ['quizid' => $quiz->id],
                'userid, attempt'
            );
        } else {
            $attempts = $DB->get_records('quiz_attempts', [
                'quiz'  => $quiz->id,
                'state' => 'finished',
            ], 'userid, attempt');
        }

        if (!$attempts) {
            return [];
        }

        // Batch-load user records instead of querying per attempt.
        $userids = array_unique(array_map(fn($a) => $a->userid, $attempts));
        $users = $DB->get_records_list('user', 'id', $userids, '', 'id, firstname, lastname, email');

        // Batch-loading every attempt's question usage in one pass instead of
        // once per attempt (question_engine::load_questions_usage_by_activity(),
        // singular, issues its own set of queries every time it's called — at
        // 500-1000 finished attempts that's 500-1000x the round trips) is
        // still done in batches of self::ATTEMPT_BATCH_SIZE attempts at a
        // time, not one single batch covering the whole quiz. Each batch is
        // still one bulk query via question_engine_data_mapper::
        // load_questions_usages_by_activity() (plural) — the same fix
        // mod_quiz's own "Responses" report uses for this problem (see
        // quiz_first_or_all_responses_table::load_extra_data() in Moodle
        // core, confirmed against a real Moodle 4.5 checkout) — so this is
        // still a handful of bulk round trips per quiz, not hundreds; the
        // batching here is purely to bound *memory*, not round trips.
        // Confirmed directly: a course with several quizzes at 2,000-2,700+
        // attempts each needed close to or over 1GB just to hold one such
        // quiz's whole batch of hydrated question_usage_by_activity objects
        // (question_attempt <-> question_attempt_step reference cycles) at
        // once. Processing and discarding one batch at a time — with an
        // explicit gc_collect_cycles() between batches, the same fix that
        // already solved this same class of problem one level up in
        // get_course_response_records() below — bounds peak memory to
        // roughly one batch's worth regardless of how large the quiz is,
        // letting more parallel workers fit in the same host memory budget.
        $questiontextbyslot = [];

        foreach (array_chunk($attempts, self::ATTEMPT_BATCH_SIZE, true) as $attemptbatch) {
            $uniqueids = array_map(fn($a) => (int) $a->uniqueid, $attemptbatch);
            $datamapper = new question_engine_data_mapper();
            $qubasbyid = $datamapper->load_questions_usages_by_activity(new qubaid_list($uniqueids));

            self::append_response_rows($attemptbatch, $qubasbyid, $users, $quiz, $cm, $questiontextbyslot, $records);

            unset($qubasbyid);
            gc_collect_cycles();
        }

        return $records;
    }

    /**
     * Same as get_response_records_for_quiz(), but backed by a short-lived
     * cache (see db/caches.php's own 'rawrecords' area) keyed on the same
     * attempt-fingerprint pattern this plugin already uses for its
     * *computed* result caches — this one caches the raw fetch itself.
     *
     * Exists specifically to bridge questionanalytics.php (which fetches
     * these records to build the on-screen view) and
     * questionanalyticspdf.php (a separate later request when a visitor
     * clicks "Download PDF" from that same page) — confirmed directly that
     * PDF generation was always re-fetching every record from scratch even
     * though the exact same quiz's exact same records had just been
     * fetched, moments earlier, to render the page the PDF button appears
     * on. Deliberately a short TTL (300s, not the 3600s this plugin's
     * *result* caches use) — this is meant to catch "the same visitor,
     * moments later," not to serve as a long-lived cache for records that
     * are considerably larger than an already-computed result.
     *
     * @param stdClass $quiz
     * @param stdClass $course
     * @param string $fingerprint from local_quizanalytics_quiz_cache_helper::stats_for_quiz($quiz)
     * @return array
     */
    public static function get_response_records_for_quiz_cached(stdClass $quiz, stdClass $course, string $fingerprint): array {
        $cache = \cache::make('local_quizanalytics', 'rawrecords');
        $key = \local_quizanalytics_quiz_cache_helper::build_key($quiz->id, $fingerprint);
        $records = $cache->get($key);
        if ($records === false) {
            $records = self::get_response_records_for_quiz($quiz, $course);
            $cache->set($key, $records);
        }
        return $records;
    }

    /**
     * Build the compact Moodle-backed snapshot shown above an individual quiz
     * analysis. Attempt counts include every Moodle attempt state, while the
     * per-question student counts include finished and in-progress attempts.
     *
     * @param stdClass $quiz
     * @param stdClass $course
     * @return array
     */
    public static function get_quiz_snapshot(stdClass $quiz, stdClass $course): array {
        global $DB;

        $cm = get_coursemodule_from_instance('quiz', $quiz->id, $course->id, false, MUST_EXIST);
        $slots = $DB->get_records_sql(
            "SELECT id, slot
               FROM {quiz_slots}
              WHERE quizid = :quizid AND slot > 0
           ORDER BY slot",
            ['quizid' => $quiz->id]
        );

        $statecounts = $DB->get_records_sql(
            "SELECT state, COUNT(*) AS attemptcount
               FROM {quiz_attempts}
              WHERE quiz = :quizid
           GROUP BY state",
            ['quizid' => $quiz->id]
        );
        $studentswithattempts = (int) $DB->count_records_sql(
            "SELECT COUNT(DISTINCT userid)
               FROM {quiz_attempts}
              WHERE quiz = :quizid",
            ['quizid' => $quiz->id]
        );
        $attempts = ['total' => 0, 'finished' => 0, 'inprogress' => 0, 'other' => 0];
        foreach ($statecounts as $statecount) {
            $count = (int) $statecount->attemptcount;
            $attempts['total'] += $count;
            if ($statecount->state === 'finished') {
                $attempts['finished'] = $count;
            } else if ($statecount->state === 'inprogress') {
                $attempts['inprogress'] = $count;
            } else {
                $attempts['other'] += $count;
            }
        }

        // Match Moodle's Quiz Overview "Overall average" row: average the
        // raw sumgrades for finished attempts with a grade, then rescale it
        // to the quiz grade shown by Moodle.
        $averagerecord = $DB->get_record_sql(
            "SELECT AVG(sumgrades) AS averagegrade, COUNT(sumgrades) AS gradedcount
               FROM {quiz_attempts}
              WHERE quiz = :quizid AND state = :finishedstate AND sumgrades IS NOT NULL",
            ['quizid' => $quiz->id, 'finishedstate' => 'finished']
        );
        $quizaverage = $averagerecord->averagegrade === null
            ? null
            : (float) quiz_rescale_grade((float) $averagerecord->averagegrade, $quiz, false);

        $studentcounts = $DB->get_records_sql(
            "SELECT slot.slot,
                    COUNT(DISTINCT CASE WHEN questionattempt.id IS NOT NULL THEN attempt.userid END) AS studentcount
               FROM {quiz_slots} slot
          LEFT JOIN {quiz_attempts} attempt
                 ON attempt.quiz = slot.quizid
                AND attempt.state IN ('finished', 'inprogress')
          LEFT JOIN {question_attempts} questionattempt
                 ON questionattempt.questionusageid = attempt.uniqueid
                AND questionattempt.slot = slot.slot
              WHERE slot.quizid = :quizid AND slot.slot > 0
           GROUP BY slot.slot
           ORDER BY slot.slot",
            ['quizid' => $quiz->id]
        );
        $studentsbyquestion = [];
        foreach ($studentcounts as $studentcount) {
            $studentsbyquestion[] = [
                'question' => 'Q' . (int) $studentcount->slot,
                'students' => (int) $studentcount->studentcount,
            ];
        }

        // Use the same Moodle Quiz Statistics Facility Index values shown in
        // Quiz -> Results -> Statistics. Do not recompute question means from
        // the plugin's parsed response rows.
        $questionmeans = [];
        $questionmeancounts = [];
        foreach (self::get_course_question_facility_data($course, [$quiz]) as $facilityrow) {
            $questionmeans[$facilityrow['question_number']] = $facilityrow['mark_average'];
            $questionmeancounts[$facilityrow['question_number']] = $facilityrow['mark_average_count'];
        }

        return [
            'question_count' => count($slots),
            'students_with_attempts' => $studentswithattempts,
            'attempts_total' => $attempts['total'],
            'attempts_finished' => $attempts['finished'],
            'attempts_inprogress' => $attempts['inprogress'],
            'attempts_other' => $attempts['other'],
            'quiz_average' => $quizaverage,
            'quiz_average_finished' => (int) $averagerecord->gradedcount,
            'students_per_question' => $studentsbyquestion,
            'question_means' => $questionmeans,
            'question_mean_counts' => $questionmeancounts,
            'quiz_report_url' => (new \moodle_url('/mod/quiz/report.php', [
                'id' => $cm->id,
                'mode' => 'overview',
            ]))->out(false),
            'quiz_responses_url' => (new \moodle_url('/mod/quiz/report.php', [
                'id' => $cm->id,
                'mode' => 'responses',
            ]))->out(false),
        ];
    }

    /**
     * Times a real fetch of just the first $samplesize finished attempts
     * for $quiz, on this host, right now — used by the on-demand pages
     * (index.php/questionanalytics.php) to decide whether a genuinely cold
     * compute is safe to run inline or should defer to a background task,
     * without relying on a fixed attempt-count threshold. A fixed count
     * can't generalize across hosts or question sets: this session's own
     * benchmarking found per-attempt cost varying roughly 10x between a
     * simple randomised question and a genuinely complex one, and it
     * necessarily also depends on the host's own CPU/DB speed — a number
     * picked against one specific machine's measured rate has no reason to
     * hold on a slower shared host or a faster dedicated one. Measuring a
     * small real sample here instead answers the only question that
     * actually matters: on *this* host, with *this* quiz's *actual*
     * questions, right now, how long would the rest of this fetch take.
     *
     * Deliberately redoes this sample's own work as part of the full fetch
     * that follows a "yes, proceed inline" decision, rather than trying to
     * resume/reuse the sampled records — the sample is small and cheap by
     * design (a few dozen attempts against a real course's own average
     * question complexity costs well under a second), so the redundant
     * work is negligible next to either outcome (a full inline fetch, or a
     * background dispatch that was going to redo everything anyway).
     *
     * Default sample size (100) chosen empirically, not arbitrarily:
     * measured directly against a real 2,764-attempt quiz, a 30-attempt
     * sample overestimated the true steady-state rate by ~4.5x (one-time
     * costs — class loading, first-time CAS/compiled-cache population for
     * the first distinct questions encountered — dominate a very small
     * sample), while 100 landed within ~1.4x of it and 250 barely improved
     * on that further. 100 attempts costs roughly a second on this
     * session's own hardware — cheap enough to run synchronously as a
     * probe before deciding, and the remaining ~1.4x overestimate is a
     * conservative bias (more likely to defer to background than strictly
     * necessary), the safe direction to err on for a timeout-avoidance
     * check.
     *
     * @param stdClass $quiz
     * @param stdClass $course
     * @param int $samplesize
     * @return float|null seconds per attempt, or null if the quiz has no
     *         finished attempts to sample at all (nothing to estimate from).
     */
    public static function estimate_seconds_per_attempt(stdClass $quiz, stdClass $course, int $samplesize = 100): ?float {
        global $DB;

        $cm = get_coursemodule_from_instance('quiz', $quiz->id, $course->id, false, MUST_EXIST);

        $sample = $DB->get_records('quiz_attempts', [
            'quiz'  => $quiz->id,
            'state' => 'finished',
        ], 'userid, attempt', '*', 0, max(1, $samplesize));

        if (!$sample) {
            return null;
        }

        $userids = array_unique(array_map(fn($a) => $a->userid, $sample));
        $users = $DB->get_records_list('user', 'id', $userids, '', 'id, firstname, lastname, email');

        $uniqueids = array_map(fn($a) => (int) $a->uniqueid, $sample);
        $datamapper = new question_engine_data_mapper();

        $t0 = microtime(true);
        $qubasbyid = $datamapper->load_questions_usages_by_activity(new qubaid_list($uniqueids));
        $questiontextbyslot = [];
        $samplerecords = [];
        self::append_response_rows($sample, $qubasbyid, $users, $quiz, $cm, $questiontextbyslot, $samplerecords);
        $elapsed = microtime(true) - $t0;

        unset($qubasbyid);
        gc_collect_cycles();

        return $elapsed / count($sample);
    }

    /**
     * Builds and appends one response row per attempt in $attemptbatch to
     * $records — the per-attempt body of get_response_records_for_quiz(),
     * factored out so that method can call it once per attempt batch
     * instead of once for the whole quiz (see the batching comment there).
     *
     * @param stdClass[] $attemptbatch quiz_attempts rows for this batch
     * @param \question_usage_by_activity[] $qubasbyid keyed by uniqueid, this batch only
     * @param stdClass[] $users keyed by userid, every user in the whole quiz (not just this batch)
     * @param stdClass $quiz
     * @param stdClass $cm course module record for $quiz, used to build the review-page link
     * @param array $questiontextbyslot memo keyed by [slot][variant], shared and mutated across every batch of this quiz
     * @param array $records output, appended to in place
     */
    private static function append_response_rows(
        array $attemptbatch,
        array $qubasbyid,
        array $users,
        stdClass $quiz,
        stdClass $cm,
        array &$questiontextbyslot,
        array &$records
    ): void {
        foreach ($attemptbatch as $attempt) {
            $user = $users[$attempt->userid] ?? null;
            if (!$user) {
                continue; // Deleted/suspended user — skip rather than fail the whole report.
            }

            $quba = $qubasbyid[$attempt->uniqueid] ?? null;
            if (!$quba) {
                continue; // No question usage recorded for this attempt — shouldn't happen for a finished attempt.
            }

            $row = [
                'attempt_id'     => $attempt->id,
                'cmid'           => $cm->id,
                'last_name'      => $user->lastname,
                'first_name'     => $user->firstname,
                'email'          => $user->email,
                'state'          => $attempt->state,
                // Passing fixday=false: userdate()'s default (true) strips the
                // leading zero from the day ("2026-6-3" instead of
                // "2026-06-03"), which breaks sections-renderer.js's
                // ISO_DATETIME_RE match for single-digit days — that row
                // then fell back to showing this raw string unformatted
                // while every other row displayed nicely, which is exactly
                // the inconsistent Completed On/Started On formatting
                // reported against this page.
                'started_on'     => userdate($attempt->timestart, '%Y-%m-%d %H:%M:%S', 99, false),
                'completed'      => $attempt->timefinish
                    ? userdate($attempt->timefinish, '%Y-%m-%d %H:%M:%S', 99, false) : '',
                'time_taken_secs' => $attempt->timefinish
                    ? ($attempt->timefinish - $attempt->timestart) : null,
                'grade'          => $attempt->sumgrades,
                'max_grade'      => $quiz->sumgrades,
                'attempt_number' => $attempt->attempt,
            ];

            $qnum = 1;
            foreach ($quba->get_slots() as $slot) {
                // $quba->get_question($slot) is the single most expensive
                // call in this whole method — 4.2ms/call measured directly
                // (vs. under 0.03ms combined for the three calls below),
                // because it instantiates the STACK question fresh for this
                // attempt's specific seed (CAS work), not just a DB read.
                // $question's only use anywhere in this method is as the
                // argument to render_stack_question_text() two lines below,
                // which is itself memoized per slot — so once that memo is
                // warm, instantiating a fresh question object here would be
                // pure waste. Only call get_question() when the memo is
                // actually still empty and its result will really be used.
                //
                // Memoized by (slot, variant), not slot alone: a randomised
                // STACK question can get a different Moodle-assigned seed
                // per attempt (deployed variants, or free randomisation when
                // none are deployed — see qtype_stack_question::
                // start_attempt()), and different seeds genuinely instantiate
                // different CAS text for the same slot. Confirmed directly
                // against a real course's data first (courses 9/10/11/12:
                // zero questions there ever use more than one distinct
                // variant, so that data alone could never have caught this)
                // then against a synthetic randomised question forced onto
                // two different variants for the same slot, which reproduced
                // exactly this bug — the second attempt processed silently
                // inherited the first attempt's instantiated text. Variant
                // count is bounded per slot (deployed variants, or whatever
                // distinct seeds real attempts actually used), so this still
                // gives the original memo's full benefit whenever a slot's
                // attempts share a seed — the common case — and only ever
                // pays for a fresh instantiation when they genuinely differ.
                $variant = $quba->get_variant($slot);
                if (empty($questiontextbyslot[$qnum][$variant])) {
                    $questiontextbyslot[$qnum][$variant] = self::render_stack_question_text($quba->get_question($slot));
                }
                $row["question_{$qnum}_text"] = $questiontextbyslot[$qnum][$variant];

                // Get_response_summary() is the same method the core "Responses"
                // report calls to build its "Response N" column — for STACK
                // questions this includes the ansK/prtK trace the Python parser
                // already knows how to read (parse_response_cell).
                $row["response_{$qnum}"]         = $quba->get_response_summary($slot) ?? '';
                $row["right_answer_{$qnum}"]     = $quba->get_right_answer_summary($slot) ?? '';
                $row["question_{$qnum}_mark"]    = $quba->get_question_mark($slot);
                $row["question_{$qnum}_maxmark"] = $quba->get_question_max_mark($slot);
                // Moodle's own question state (e.g. "todo", "needsgrading")
                // flags a response that isn't really gradeable yet even
                // though it may still carry leftover PRT fractions from an
                // earlier, since-superseded validation — see parser.php's
                // own use of this field. Cheap: reads off the already-loaded
                // $quba, no extra query.
                $row["question_{$qnum}_state"]   = (string) $quba->get_question_attempt($slot)->get_state();

                $qnum++;
            }

            $records[] = $row;
        }
    }

    /**
     * $question->questiontext is the raw, author-written source — for STACK
     * questions that still contains unresolved CAS placeholders (@variable@)
     * and every language's [[lang code='xx']]...[[/lang]] block side by side,
     * none of which get processed until the question is actually rendered
     * through STACK's own CAS-text engine. This renders it the same way
     * STACK's own question renderer would (CAS variables substituted,
     * [[lang]] blocks resolved to only the current language), using the
     * "out of context" processor STACK itself uses for report/summary
     * generation (stack_question::get_question_summary() is the other
     * caller of this exact pattern) since there's no live question_attempt
     * here to render against.
     *
     * Falls back to the raw questiontext for non-STACK questions or if
     * anything above goes wrong — a report showing unresolved placeholders
     * is still more useful than one that fails outright for one bad question.
     *
     * @param question_definition $question
     * @return string
     */
    protected static function render_stack_question_text(question_definition $question): string {
        global $CFG;

        if (!($question instanceof \qtype_stack_question) || empty($question->questiontextinstantiated)) {
            return $question->questiontext;
        }

        try {
            require_once($CFG->dirroot . '/question/type/stack/locallib.php');
            $processor = new \castext2_qa_processor(new \stack_outofcontext_process());
            $rendered = $question->questiontextinstantiated->get_rendered($processor);
            if ($rendered !== null && $rendered !== '') {
                return $rendered;
            }
        } catch (\Throwable $e) {
            debugging('local_quizanalytics (quiz): could not render CAS question text for a STACK question: ' .
                $e->getMessage(), DEBUG_DEVELOPER);
        }

        return $question->questiontext;
    }

    /**
     * Response records for every STACK quiz in the course, grouped by quiz
     * name — the shape local_quizanalytics_quiz_api_client::analyze_course()
     * expects.
     *
     * @param stdClass $course
     * @param stdClass[] $stackquizzes as returned by get_course_stack_quizzes()
     * @return array [quiz_name => records[]]
     */
    public static function get_course_response_records(stdClass $course, array $stackquizzes): array {
        $bycourse = [];
        foreach ($stackquizzes as $quiz) {
            $bycourse[$quiz->name] = self::get_response_records_for_quiz($quiz, $course);

            // get_response_records_for_quiz() batch-loads a question_usage_by_activity
            // per attempt (load_questions_usages_by_activity()) — these hold internal
            // reference cycles (question_attempt <-> question_attempt_step linkage),
            // which only PHP's *cycle* collector (not plain refcounting) can reclaim.
            // Across many quizzes in one course-wide request, that collector's own
            // automatic, threshold-based triggering falls further and further behind
            // as more cyclic garbage piles up, and each of its automatic runs gets
            // more expensive to walk — measured directly on a real 38-quiz/48,445-
            // attempt course, this showed up as a single quiz's own fetch going from
            // ~10s (first few quizzes) to 461s (ninth) with no code-level difference
            // between them. Forcing a clean, predictable collection right after each
            // quiz — while its now-unreferenced QUBA objects are the only large batch
            // of cyclic garbage outstanding — keeps every run cheap instead of one
            // increasingly expensive automatic sweep; same course dropped to 16-17s
            // for the same previously-461s quiz with this in place, and memory usage
            // stayed bounded rather than growing for the rest of the request.
            gc_collect_cycles();
        }
        return $bycourse;
    }

    /**
     * Reads Moodle's Quiz Statistics Facility Index/average mark for each
     * direct STACK question slot in the supplied quizzes — the same figures
     * shown in Quiz -> Results -> Statistics. Moodle calculates and caches
     * missing statistics itself; this only reads the resulting values, never
     * recomputes them from this plugin's own parsed response rows, so the
     * quiz snapshot always agrees with Moodle's own report.
     *
     * @param stdClass $course
     * @param stdClass[] $stackquizzes
     * @return array[]
     */
    public static function get_course_question_facility_data(stdClass $course, array $stackquizzes): array {
        global $DB;

        $rows = [];
        $report = new \quiz_statistics_report();
        foreach ($stackquizzes as $quiz) {
            $cm = get_coursemodule_from_instance('quiz', $quiz->id, $course->id, false, MUST_EXIST);
            $context = \context_module::instance($cm->id);
            $stackslots = $DB->get_records_sql(
                "SELECT DISTINCT slot.id AS slotid, slot.slot, slot.maxmark, q.id AS questionid, q.name AS questionname
                   FROM {quiz_slots} slot
                   JOIN {question_references} qr ON qr.usingcontextid = :contextid
                                                 AND qr.component = 'mod_quiz'
                                                 AND qr.questionarea = 'slot'
                                                 AND qr.itemid = slot.id
                   JOIN {question_bank_entries} qbe ON qbe.id = qr.questionbankentryid
                   JOIN {question_versions} qv ON qv.questionbankentryid = qbe.id
                                                 AND qv.version = (
                                                     SELECT MAX(qvlatest.version)
                                                       FROM {question_versions} qvlatest
                                                      WHERE qvlatest.questionbankentryid = qbe.id
                                                 )
                   JOIN {question} q ON q.id = qv.questionid AND q.qtype = 'stack'
                  WHERE slot.quizid = :quizid
               ORDER BY slot.slot",
                ['contextid' => $context->id, 'quizid' => $quiz->id]
            );

            if (empty($stackslots)) {
                continue;
            }

            // This is the same raw question-average source used by Moodle's
            // Quiz overview report (the values shown in its final average
            // row) — averaged over finished, non-preview attempts only.
            $questions = $report->load_and_initialise_questions_for_calculations($quiz);
            $qubaids = new \qubaid_join(
                '{quiz_attempts} quiza',
                'quiza.uniqueid',
                'quiza.quiz = :quizid AND quiza.preview = 0 AND quiza.state = :finishedstate',
                ['quizid' => (int) $quiz->id, 'finishedstate' => 'finished']
            );
            $averagemarks = (new \question_engine_data_mapper())->load_average_marks(
                $qubaids,
                array_keys($questions)
            );

            $questionstats = $report->calculate_questions_stats_for_question_bank(
                (int) $quiz->id,
                true,
                false
            );
            foreach ($stackslots as $slot) {
                $facility = null;
                $markaverage = null;
                if (isset($averagemarks[(int) $slot->slot]) && (float) $slot->maxmark > 0) {
                    $rawmarkaverage = (float) $averagemarks[(int) $slot->slot]->averagefraction *
                        (float) $slot->maxmark;
                    $markaverage = (float) quiz_rescale_grade($rawmarkaverage, $quiz, false);
                }
                if ($questionstats !== null && $questionstats->has_slot((int) $slot->slot)) {
                    $stat = $questionstats->for_slot((int) $slot->slot);
                    if ($stat->facility !== null) {
                        $facility = (float) $stat->facility * 100.0;
                    }
                }
                $rows[] = [
                    'quiz_name' => $quiz->name,
                    'question_number' => 'Q' . (int) $slot->slot,
                    'question_slot' => (int) $slot->slot,
                    'question_name' => $slot->questionname,
                    'facility_index' => $facility,
                    'mark_average' => $markaverage,
                    'mark_average_count' => isset($averagemarks[(int) $slot->slot])
                        ? (int) $averagemarks[(int) $slot->slot]->numaveraged : 0,
                ];
            }
        }

        return $rows;
    }
}
