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
 * Proactively (re)computes the Quiz Analytics/Question Analytics result
 * caches for every course with STACK quiz activity — the same
 * get_response_records_for_quiz()/analyze()/analyze_course() calls
 * index.php and questionanalytics.php already make on a cache miss, just
 * moved off a real visitor's own HTTP request and onto cron instead.
 *
 * Exists because that cold-cache compute, over a course with several
 * hundred finished attempts, can run long enough that a reverse proxy in
 * front of the site gives up on the browser before PHP finishes
 * (Cloudflare's default ~100s edge timeout, seen in practice as a 524) —
 * raising local_quizanalytics/computetimelimit doesn't help with that
 * specific failure, since it only raises PHP's own execution time limit,
 * not any proxy sitting in front of PHP. Running this on a schedule means
 * a real viewer only ever needs to hit a warm cache, and the expensive
 * first-time compute happens somewhere with no browser timeout waiting on
 * it. See ignore_user_abort() usage in index.php/questionanalytics.php for
 * the complementary fix for whatever narrow window still lands on a cold
 * cache between task runs.
 *
 * Fetches each course's quizzes at most once per run — profiling a real
 * 38-quiz/48,445-attempt course showed the fetch stage (data_fetcher.php)
 * dominating runtime by roughly 10-20x over the entire analysis pipeline
 * combined for the same quiz (see parallel_course_fetcher.php's own
 * docblock for why: it's genuine live CAS/PRT grading Moodle's own
 * question-engine API does on every call, not something this plugin can
 * memoize away). The original version of this task fetched every cold
 * quiz's records once for its own per-quiz cache entry, then fetched all
 * of them *again* for the course-wide entry — this version fetches once,
 * in parallel via parallel_course_fetcher::fetch(), and reuses the same
 * records for both.
 *
 * Scoped to the default view only — colorblind off, anonymize off, the
 * default course-wide grade type — the combination almost every visitor
 * actually lands on (sections_output_helper::resolve_colorblind_mode()/
 * resolve_anonymize_mode() both default false, and are per-user
 * preferences this task has no particular user's session to read). A
 * visitor who's personally toggled colorblind or anonymize mode still
 * computes cold on their own next visit — an accepted gap, not something a
 * site-wide background task can reasonably pre-empt for every user.
 *
 * Solution Process Visualization's own cache areas (solutionprocessmeta/
 * solutionprocess) are deliberately not warmed here: unlike the other two
 * areas, there's no single "default" question/part/student selection to
 * precompute — every combination is its own cache entry, and eagerly
 * computing all of them for every STACK question in every course would be
 * its own much larger background job. Those two still benefit from caching
 * after a teacher's first real look at one specific question/part, same as
 * before this task existed.
 *
 * @package local_quizanalytics
 * @copyright  2026 Ernest Ting <eting@caltech.edu>
 * @license    http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */

namespace local_quizanalytics\task;

use local_quizanalytics\quiz\analytics\course_analysis;

defined('MOODLE_INTERNAL') || die();

require_once($CFG->dirroot . '/local/quizanalytics/classes/quiz/data_fetcher.php');
require_once($CFG->dirroot . '/local/quizanalytics/classes/quiz/api_client.php');
require_once($CFG->dirroot . '/local/quizanalytics/classes/quiz/cache_helper.php');
require_once($CFG->dirroot . '/local/quizanalytics/classes/task/parallel_course_fetcher.php');

/**
 * Scheduled task that keeps the Quiz Analytics/Question Analytics result caches warm.
 */
class warm_analytics_cache extends \core\task\scheduled_task {
    /**
     * Task name shown in Site administration > Server > Scheduled tasks.
     *
     * @return string
     */
    public function get_name(): string {
        return get_string('task:warmanalyticscache', 'local_quizanalytics');
    }

    /**
     * Recomputes and caches the default-view result for every STACK course
     * whose current cache entry is missing or stale, skipping anything
     * that's already warm for its current attempt fingerprint.
     */
    public function execute(): void {
        global $DB;

        // This is CLI/cron-only (never a web request a shared host needs to
        // budget memory for per-request), and a real course's course-wide
        // fetch legitimately needs to hold every one of its quizzes' full
        // response records in memory at once — confirmed directly: a
        // 38-quiz/48,445-attempt course exhausted a 2048M ceiling on one
        // worker that happened to draw several of the largest quizzes.
        // ini_set() here is inherited by every forked child in
        // parallel_course_fetcher.php (ordinary copied process memory, no
        // shared-resource fork hazard the way a DB connection is), so this
        // one call covers the parent's own merge/analyze work and every
        // child's fetch work alike.
        //
        // Deliberately a bounded, admin-configurable value rather than -1
        // (unlimited) — confirmed directly why that's not safe: with no PHP-
        // level ceiling at all, several concurrent workers each drawing a
        // few large quizzes can exceed the *host's* real available memory
        // before PHP's own accounting would ever object, and at that point
        // it's the kernel's OOM killer deciding what to kill, not PHP —
        // observed taking out this task's own worker processes with no
        // catchable error (just gone, SIGKILL), and on a busier host it
        // could just as easily pick MariaDB or the Maxima backend instead,
        // which is a much bigger problem than one course staying stale
        // until the next run. A real PHP "Allowed memory size exhausted"
        // fatal is self-contained and reported; see this task's own
        // parallelworkermemory setting description for sizing guidance.
        $workermemorymb = max(256, (int) (get_config('local_quizanalytics', 'parallelworkermemory') ?: 2048));
        @ini_set('memory_limit', $workermemorymb . 'M');

        $client = new \local_quizanalytics_quiz_api_client();
        $workers = max(1, (int) (get_config('local_quizanalytics', 'parallelworkers') ?: 4));

        foreach (\local_quizanalytics_quiz_data_fetcher::get_courses_with_stack_quizzes() as $courseid) {
            $course = $DB->get_record('course', ['id' => $courseid]);
            if (!$course) {
                continue; // Deleted since get_courses_with_stack_quizzes() ran.
            }

            $stackquizzes = \local_quizanalytics_quiz_data_fetcher::get_course_stack_quizzes($courseid);
            if (empty($stackquizzes)) {
                continue;
            }

            $this->warm_course($client, $course, $stackquizzes, $workers);
        }
    }

    /**
     * Warms whatever's cold (per-quiz and/or course-wide) for one course,
     * fetching every quiz that needs it exactly once.
     *
     * @param \local_quizanalytics_quiz_api_client $client
     * @param \stdClass $course
     * @param \stdClass[] $stackquizzes
     * @param int $workers
     */
    private function warm_course(
        \local_quizanalytics_quiz_api_client $client,
        \stdClass $course,
        array $stackquizzes,
        int $workers
    ): void {
        $quizstats = [];
        $coldquizzes = [];
        foreach ($stackquizzes as $quiz) {
            $stats = \local_quizanalytics_quiz_cache_helper::stats_for_quiz($quiz);
            $quizstats[$quiz->id] = $stats;
            if ($stats->count === 0) {
                continue; // No finished attempts yet — nothing to warm for this quiz.
            }
            $qacache = \cache::make('local_quizanalytics', 'questionanalysis');
            $qakey = \local_quizanalytics_quiz_cache_helper::build_key($quiz->id, $stats->fingerprint, false, false);
            if ($qacache->get($qakey) === false) {
                $coldquizzes[$quiz->id] = $quiz;
            }
        }

        $coursestats = \local_quizanalytics_quiz_cache_helper::stats_for_quizzes($stackquizzes);
        $coursewidecold = false;
        $qwkey = null;
        if ($coursestats->count > 0) {
            $qwcache = \cache::make('local_quizanalytics', 'quizanalysiscoursewide');
            $selectionkey = \local_quizanalytics_quiz_cache_helper::selection_key(
                array_map(fn($quiz) => (int) $quiz->id, $stackquizzes)
            );
            $qwkey = \local_quizanalytics_quiz_cache_helper::build_key(
                'course-ui-v5', $course->id,
                $coursestats->fingerprint,
                $selectionkey,
                course_analysis::DEFAULT_GRADE_TYPE,
                false,
                false
            );
            $coursewidecold = $qwcache->get($qwkey) === false;
        }

        if (empty($coldquizzes) && !$coursewidecold) {
            return; // Everything for this course is already warm.
        }

        // If the course-wide entry needs warming it needs every quiz's
        // records regardless of whether that quiz's own per-quiz entry
        // does; otherwise only fetch the quizzes that are actually cold.
        $quizzestofetch = $coursewidecold ? $stackquizzes : $coldquizzes;

        try {
            $byquiz = \local_quizanalytics\task\parallel_course_fetcher::fetch($course, $quizzestofetch, $workers);
        } catch (\Throwable $e) {
            // mtrace(), not debugging() — this task runs unattended on cron,
            // typically with $CFG->debug at its production default of
            // DEBUG_NONE, under which debugging() is a silent no-op.
            // Without a visible trace here, a course that consistently fails
            // to warm (e.g. a worker that keeps losing the fork/OOM race
            // under real host memory pressure — see parallel_course_fetcher's
            // own docblock) looks like an ordinary "task completed" run in
            // Site administration > Server > Tasks every single time, while
            // silently re-attempting the same full, expensive fetch and
            // never actually caching anything for that course.
            mtrace('local_quizanalytics: could not warm course ' . $course->id . ': ' . $e->getMessage());
            return; // Try again next run rather than caching anything partial.
        }

        foreach ($coldquizzes as $quiz) {
            $records = $byquiz[$quiz->name] ?? [];
            if (empty($records)) {
                continue;
            }
            $result = $client->analyze($quiz->name, $records, false, false);
            if ($result !== null) {
                $qacache = \cache::make('local_quizanalytics', 'questionanalysis');
                $qakey = \local_quizanalytics_quiz_cache_helper::build_key(
                    $quiz->id,
                    $quizstats[$quiz->id]->fingerprint,
                    false,
                    false
                );
                $qacache->set($qakey, $result);
            } else {
                mtrace('local_quizanalytics: analyze() returned no result for quiz ' . $quiz->id
                    . ' (' . $quiz->name . ') in course ' . $course->id);
            }
        }

        if ($coursewidecold) {
            $filtered = array_filter($byquiz, fn($records) => !empty($records));
            $result = $client->analyze_course($course->fullname, $filtered, false, course_analysis::DEFAULT_GRADE_TYPE, false);
            if ($result !== null) {
                $qwcache = \cache::make('local_quizanalytics', 'quizanalysiscoursewide');
                $qwcache->set($qwkey, $result);
            } else {
                mtrace('local_quizanalytics: analyze_course() returned no result for course ' . $course->id);
            }
        }
    }
}
