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
 * Computes and caches one specific Quiz Analytics or Question Analytics
 * view in the background, for a visitor whose own on-demand request hit a
 * cold cache on a quiz/course too large to safely compute inline.
 *
 * warm_analytics_cache (the scheduled task) already proactively warms every
 * STACK course's *default* view (colorblind off, anonymize off, average
 * grade) on a schedule — see that task's own docblock. This task exists for
 * the gap that leaves: index.php/questionanalytics.php's synchronous
 * on-demand path, which still fetches serially (pcntl_fork only works in
 * CLI/cron context — confirmed in this session's own notes), and which any
 * visitor can hit on a genuinely cold view (a course that hasn't been
 * warmed yet, or a colorblind/anonymize/grade-type combination the
 * scheduled task never covers, since it has no specific visitor's
 * preferences to warm for).
 *
 * Measured directly against this plugin's own real, largest test quiz
 * (2,764 finished attempts): a cold on-demand compute already takes
 * ~20s even in the best case this session measured, and profiling earlier
 * this session found the fetch stage scaling roughly linearly with attempt
 * count with no structural reason to expect that to change (the batching in
 * data_fetcher.php's own ATTEMPT_BATCH_SIZE already exists specifically to
 * keep memory, and therefore GC cost, from growing worse than linear as a
 * quiz gets larger — see that constant's own comment). A quiz several times
 * that size — nothing in Moodle stops one existing — can plausibly exceed a
 * reverse proxy's own timeout (Cloudflare's free/Pro default is ~100s)
 * before this plugin's own code ever gets a chance to finish or even to
 * reach the ignore_user_abort(true) guard that already protects a
 * request that does complete despite the browser giving up. Dispatching
 * to run here instead means the visitor's own request returns almost
 * immediately either way, and the expensive compute happens somewhere with
 * no proxy timeout waiting on it — the same reasoning as
 * warm_analytics_cache's own scheduled runs, just triggered by a real
 * visitor's specific cold view instead of a schedule.
 *
 * Deliberately dispatched via
 * \core\task\manager::queue_adhoc_task($task, true) — the second argument
 * deduplicates against an already-queued task with identical classname,
 * component, and custom data, so several visitors hitting the same cold
 * (quiz, colorblind, anonymize) or (course, gradetype, colorblind,
 * anonymize) combination in quick succession queue at most one task
 * between them, not one each. Also composes safely with the scheduled
 * task warming the *default* view on its own schedule: both this task and
 * a cron run computing the same view independently arrive at the same
 * deterministic, fingerprint-keyed result and both merely call
 * cache->set() with it — redundant if they overlap, never incorrect — and
 * this task's own cache->get() check before computing anything means
 * whichever of the two finishes first makes the other's remaining work a
 * no-op.
 *
 * @package local_quizanalytics
 * @copyright  2026 Ernest Ting <eting@caltech.edu>
 * @license    http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */

namespace local_quizanalytics\task;

defined('MOODLE_INTERNAL') || die();

require_once($CFG->dirroot . '/local/quizanalytics/classes/quiz/data_fetcher.php');
require_once($CFG->dirroot . '/local/quizanalytics/classes/quiz/api_client.php');
require_once($CFG->dirroot . '/local/quizanalytics/classes/quiz/cache_helper.php');
require_once($CFG->dirroot . '/local/quizanalytics/classes/quiz/prepared_store.php');
require_once($CFG->dirroot . '/local/quizanalytics/classes/task/parallel_course_fetcher.php');

/**
 * Ad-hoc task: warm one specific quiz or course-wide view.
 */
class warm_single_view_adhoc_task extends \core\task\adhoc_task {
    /** Selection currently being processed by this PHP task. */
    private static string $progressselectionkey = '';
    /**
     * How long a matching queued task can sit unstarted/unfinished before
     * get_queued_age_seconds()'s caller should stop showing the ordinary
     * "come back in a few minutes" notice and show an honest "this looks
     * stuck" one instead. A flat cap rather than a multiple of the
     * background-compute threshold (local_quizanalytics/backgroundthreshold)
     * on purpose — that threshold is an *attempt count*, not a time
     * estimate, and per-attempt cost is genuinely question-complexity- and
     * host-dependent (see that setting's own description), so there's no
     * reliable way to turn it into an expected-runtime figure. Every real
     * full-course cold run measured this session, at any tested scale, has
     * finished well inside this — a task still queued past it on a real
     * site most likely means cron isn't running at all, or the task
     * crashed/was OOM-killed and is sitting in a fail-delay backoff.
     */
    const STALE_SECONDS = 900;

    /**
     * Queues (or, if an identical request is already queued, reuses) a
     * background compute of one quiz's Question Analytics/Solution Process
     * view for the given display options.
     *
     * @param int $quizid
     * @param bool $colorblind
     * @param bool $anonymize
     */
    public static function dispatch_for_quiz(int $quizid, bool $colorblind, bool $anonymize): void {
        self::dispatch([
            'type' => 'quiz',
            'id' => $quizid,
            'colorblind' => $colorblind,
            'anonymize' => $anonymize,
        ]);
    }

    /**
     * Queues (or reuses) a background compute of one quiz's Solution
     * Process Visualization *metadata* only (the question/part/student
     * selector's own options) — not any specific (question, part, student)
     * combination, which is deliberately never proactively warmed here or
     * by the scheduled task; see warm_analytics_cache's own docblock for
     * why (every combination is its own cache entry).
     *
     * @param int $quizid
     * @param bool $anonymize
     */
    public static function dispatch_for_quiz_meta(int $quizid, bool $anonymize): void {
        self::dispatch([
            'type' => 'quizmeta',
            'id' => $quizid,
            'anonymize' => $anonymize,
        ]);
    }

    /**
     * Queues (or reuses) a background compute of one course's Quiz
     * Analytics (course-wide) view for the given display options.
     *
     * @param int $courseid
     * @param string $gradetype
     * @param bool $colorblind
     * @param bool $anonymize
     */
    public static function dispatch_for_course(
        int $courseid, string $gradetype, bool $colorblind, bool $anonymize, ?string $fingerprint = null,
        ?array $quizids = null
    ): void {
        $customdata = [
            'type' => 'course',
            'id' => $courseid,
            'gradetype' => $gradetype,
            'fingerprint' => $fingerprint,
            'colorblind' => $colorblind,
            'anonymize' => $anonymize,
            'quizids' => $quizids === null ? [] : array_map('intval', $quizids),
        ];
        if ($fingerprint !== null) {
            $selectionkey = \local_quizanalytics_quiz_cache_helper::selection_key($customdata['quizids']);
            self::$progressselectionkey = $selectionkey;
            $current = self::get_progress($courseid, $fingerprint, $gradetype, $colorblind, $anonymize, $selectionkey);
            if ($current === false || ($current['status'] ?? '') !== 'running') {
                self::set_progress($courseid, $fingerprint, $gradetype, $colorblind, $anonymize,
                    'queued', 'queued', 0, 0, get_string('progressqueued', 'local_quizanalytics'));
            }
        }
        self::dispatch($customdata);
    }

    /** Return the shared progress-cache key for a course computation. */
    public static function progress_key(
        int $courseid, string $fingerprint, string $gradetype, bool $colorblind, bool $anonymize,
        ?string $selectionkey = null
    ): string {
        $selectionkey = $selectionkey ?? (self::$progressselectionkey ?: 'all');
        return \local_quizanalytics_quiz_cache_helper::build_key(
            'course-progress-v2', $courseid, $fingerprint, $selectionkey, $gradetype, $colorblind, $anonymize
        );
    }

    /** Return the latest progress snapshot, or false when none exists. */
    public static function get_progress(
        int $courseid, string $fingerprint, string $gradetype, bool $colorblind, bool $anonymize,
        ?string $selectionkey = null
    ) {
        $cache = \cache::make('local_quizanalytics', 'analyticsprogress');
        return $cache->get(self::progress_key($courseid, $fingerprint, $gradetype, $colorblind, $anonymize, $selectionkey));
    }

    /** Write one small progress snapshot; this never performs analytics work. */
    private static function set_progress(
        int $courseid, string $fingerprint, string $gradetype, bool $colorblind, bool $anonymize,
        string $status, string $stage, int $completed, int $total, string $message, array $details = []
    ): void {
        $cache = \cache::make('local_quizanalytics', 'analyticsprogress');
        $key = self::progress_key($courseid, $fingerprint, $gradetype, $colorblind, $anonymize);
        $previous = $cache->get($key);
        $started = is_array($previous) && !empty($previous['started']) ? $previous['started'] : time();
        $previousdetails = is_array($previous) && !empty($previous['details']) ? $previous['details'] : [];
        $percent = $stage === 'complete' ? 100 : ($stage === 'queued' ? 0 : 5);
        if ($stage === 'processing' && $total > 0) {
            $percent = 10 + (int) round(65 * ($completed / $total));
        } else if ($stage === 'analyzing') {
            $percent = 80;
        } else if ($stage === 'saving') {
            $percent = 95;
        }
        $cache->set($key, [
            'status' => $status,
            'stage' => $stage,
            'completed' => $completed,
            'total' => $total,
            'percent' => min(100, max(0, $percent)),
            'message' => $message,
            'started' => $started,
            'updated' => time(),
            'elapsed' => max(0, time() - $started),
            'details' => array_merge($previousdetails, $details),
        ]);
    }

    /**
     * @param array $customdata
     */
    private static function dispatch(array $customdata): void {
        $task = new self();
        $task->set_custom_data((object) $customdata);
        \core\task\manager::queue_adhoc_task($task, true);
    }

    /**
     * How long (in seconds) a task matching $customdata has been sitting in
     * the adhoc task queue, or null if no matching task is currently
     * queued — either it's never been dispatched, or it already completed
     * and was removed (Moodle deletes an adhoc task's own row on success).
     * Used by index.php/questionanalytics.php to tell a genuinely stuck
     * dispatch (see STALE_SECONDS) apart from one that's just still
     * working normally.
     *
     * Re-derives \core\task\manager::get_queued_adhoc_task_record()'s own
     * matching logic directly (classname + component + exact customdata)
     * since that method is protected — there's no public "peek without
     * dispatching" API to call instead.
     *
     * @param array $customdata must exactly match what the corresponding
     *        dispatch_for_*() call above would pass — same keys, same
     *        value types (bool/int/string, matching json_encode() output).
     * @return int|null
     */
    public static function get_queued_age_seconds(array $customdata): ?int {
        global $DB;

        $task = new self();
        $task->set_custom_data((object) $customdata);
        $encoded = $task->get_custom_data_as_string();

        $sql = 'classname = ? AND component = ? AND '
            . $DB->sql_compare_text('customdata', \core_text::strlen($encoded) + 1) . ' = ?';
        $record = $DB->get_record_select(
            'task_adhoc',
            $sql,
            [\core\task\manager::get_canonical_class_name($task), $task->get_component(), $encoded],
            'timecreated'
        );

        if (!$record) {
            return null;
        }
        return time() - (int) $record->timecreated;
    }

    /**
     * @return string
     */
    public function get_name(): string {
        return get_string('task:warmsingleview', 'local_quizanalytics');
    }

    public function execute(): void {
        // A course-wide view can need every one of a course's quizzes'
        // full response records in memory at once (see
        // get_course_response_records()) — the same reasoning
        // warm_analytics_cache::execute() already applies to its own
        // scheduled run applies here too, since this task can be asked to
        // compute the exact same course-wide work, just for a specific
        // visitor's own (gradetype, colorblind, anonymize) combination
        // instead of the default view. Confirmed directly: without this,
        // warming a real 38-quiz course from this task hit PHP's default
        // 512M CLI memory_limit and died silently (a genuine "Allowed
        // memory size exhausted" fatal, but one that happened to produce no
        // visible output at this specific margin) rather than the fatal
        // error someone debugging the resulting stuck task_adhoc row would
        // expect to see.
        $workermemorymb = max(256, (int) (get_config('local_quizanalytics', 'parallelworkermemory') ?: 2048));
        @ini_set('memory_limit', $workermemorymb . 'M');

        $data = $this->get_custom_data();
        $client = new \local_quizanalytics_quiz_api_client();

        switch ($data->type) {
            case 'quiz':
                $this->warm_quiz_view((int) $data->id, (bool) $data->colorblind, (bool) $data->anonymize, $client);
                break;
            case 'quizmeta':
                $this->warm_quiz_meta((int) $data->id, (bool) $data->anonymize, $client);
                break;
            default:
                if (empty($data->quizids)) {
                    $this->prepare_course_view((int) $data->id, $client);
                    break;
                }
                $this->warm_course_view(
                    (int) $data->id,
                    (string) $data->gradetype,
                    (bool) $data->colorblind,
                    (bool) $data->anonymize,
                    $client,
                    isset($data->fingerprint) ? (string) $data->fingerprint : null,
                    isset($data->quizids) ? array_map('intval', (array) $data->quizids) : null
                );
        }
    }

    /**
     * Prepare reusable per-quiz attempt frames for an entire course.
     * Selection is intentionally absent: this job is independent of the
     * lecturer's current checkbox selection and runs at most once per stale
     * quiz snapshot.
     */
    private function prepare_course_view(int $courseid, $client): void {
        global $DB;

        $course = $DB->get_record('course', ['id' => $courseid]);
        if (!$course) {
            return;
        }
        $quizzes = \local_quizanalytics_quiz_data_fetcher::get_course_stack_quizzes($courseid);
        if (empty($quizzes)) {
            return;
        }

        $stale = [];
        $statsbyquiz = \local_quizanalytics_quiz_cache_helper::stats_for_quizzes_by_id($quizzes);
        foreach ($quizzes as $quiz) {
            $stats = $statsbyquiz[(int) $quiz->id];
            $row = \local_quizanalytics_prepared_store::get($courseid, (int) $quiz->id);
            if (!\local_quizanalytics_prepared_store::is_fresh($row, $stats->fingerprint)) {
                $stale[$quiz->id] = ['quiz' => $quiz, 'fingerprint' => $stats->fingerprint];
            }
        }
        if (empty($stale)) {
            return;
        }

        foreach ($stale as $item) {
            \local_quizanalytics_prepared_store::mark_running($courseid, (int) $item['quiz']->id);
        }

        try {
            $byquiz = parallel_course_fetcher::fetch(
                $course, array_map(fn($item) => $item['quiz'], $stale),
                max(1, (int) (get_config('local_quizanalytics', 'parallelworkers') ?: 4))
            );
        } catch (\Throwable $e) {
            foreach ($stale as $item) {
                \local_quizanalytics_prepared_store::mark_failed($courseid, (int) $item['quiz']->id, $e->getMessage());
            }
            mtrace('local_quizanalytics: reusable preparation failed for course ' . $courseid . ': ' . $e->getMessage());
            return;
        }

        $facilitybyquiz = [];
        foreach (\local_quizanalytics_quiz_data_fetcher::get_course_question_facility_data(
            $course, array_map(fn($item) => $item['quiz'], $stale)
        ) as $facilityrow) {
            if ($facilityrow['facility_index'] === null) {
                continue;
            }
            $name = $facilityrow['quiz_name'];
            $facilitybyquiz[$name]['sum'] = ($facilitybyquiz[$name]['sum'] ?? 0.0)
                + (float) $facilityrow['facility_index'];
            $facilitybyquiz[$name]['count'] = ($facilitybyquiz[$name]['count'] ?? 0) + 1;
        }

        foreach ($stale as $item) {
            $quiz = $item['quiz'];
            try {
                $records = $byquiz[$quiz->name] ?? [];
                $rows = \local_quizanalytics\quiz\analytics\parser::build_response_rows($records, $quiz->name, false);
                $frame = \local_quizanalytics\quiz\analytics\quiz_metrics::build_quiz_attempt_frame($rows);
                $facility = $facilitybyquiz[$quiz->name] ?? null;
                \local_quizanalytics_prepared_store::save_success($courseid, (int) $quiz->id, $item['fingerprint'], [
                    'quizid' => (int) $quiz->id,
                    'quiz_name' => $quiz->name,
                    'attemptframe' => $frame,
                    'facility_index' => $facility
                        ? round($facility['sum'] / $facility['count'], 2) : null,
                ]);
            } catch (\Throwable $e) {
                \local_quizanalytics_prepared_store::mark_failed($courseid, (int) $quiz->id, $e->getMessage());
                mtrace('local_quizanalytics: quiz preparation failed for quiz ' . $quiz->id . ': ' . $e->getMessage());
            }
        }
    }

    /**
     * @param int $quizid
     * @param bool $anonymize
     * @param \local_quizanalytics_quiz_api_client $client
     */
    private function warm_quiz_meta(int $quizid, bool $anonymize, $client): void {
        global $DB;

        $quiz = $DB->get_record('quiz', ['id' => $quizid]);
        if (!$quiz) {
            return;
        }
        $course = $DB->get_record('course', ['id' => $quiz->course]);
        if (!$course) {
            return;
        }

        $stats = \local_quizanalytics_quiz_cache_helper::stats_for_quiz($quiz);
        if ($stats->count === 0) {
            return;
        }

        $cache = \cache::make('local_quizanalytics', 'solutionprocessmeta');
        $key = \local_quizanalytics_quiz_cache_helper::build_key($quiz->id, $stats->fingerprint, $anonymize);
        if ($cache->get($key) !== false) {
            return;
        }

        $records = \local_quizanalytics_quiz_data_fetcher::get_response_records_for_quiz($quiz, $course);
        $meta = $client->solution_process_meta($quiz->name, $records, $anonymize);
        if ($meta !== null) {
            $cache->set($key, $meta);
        }
    }

    /**
     * @param int $quizid
     * @param bool $colorblind
     * @param bool $anonymize
     * @param \local_quizanalytics_quiz_api_client $client
     */
    private function warm_quiz_view(int $quizid, bool $colorblind, bool $anonymize, $client): void {
        global $DB;

        $quiz = $DB->get_record('quiz', ['id' => $quizid]);
        if (!$quiz) {
            return; // Deleted since this task was queued.
        }
        $course = $DB->get_record('course', ['id' => $quiz->course]);
        if (!$course) {
            return;
        }

        $stats = \local_quizanalytics_quiz_cache_helper::stats_for_quiz($quiz);
        if ($stats->count === 0) {
            return;
        }

        $cache = \cache::make('local_quizanalytics', 'questionanalysis');
        $key = \local_quizanalytics_quiz_cache_helper::build_key($quiz->id, $stats->fingerprint, $colorblind, $anonymize);
        if ($cache->get($key) !== false) {
            return; // Already warmed — cron or another dispatch beat this task to it.
        }

        $records = \local_quizanalytics_quiz_data_fetcher::get_response_records_for_quiz($quiz, $course);
        $result = $client->analyze($quiz->name, $records, $colorblind, $anonymize);
        if ($result !== null) {
            $cache->set($key, $result);
        }
    }

    /**
     * @param int $courseid
     * @param string $gradetype
     * @param bool $colorblind
     * @param bool $anonymize
     * @param \local_quizanalytics_quiz_api_client $client
     */
    private function warm_course_view(
        int $courseid, string $gradetype, bool $colorblind, bool $anonymize, $client, ?string $fingerprint = null,
        ?array $quizids = null
    ): void {
        global $DB;

        $course = $DB->get_record('course', ['id' => $courseid]);
        if (!$course) {
            return;
        }
        $stackquizzes = \local_quizanalytics_quiz_data_fetcher::get_course_stack_quizzes($courseid);
        if (empty($stackquizzes)) {
            return;
        }
        if ($quizids !== null && !empty($quizids)) {
            $wanted = array_flip(array_map('intval', $quizids));
            $stackquizzes = array_filter($stackquizzes, fn($quiz) => isset($wanted[(int) $quiz->id]));
        }
        $selectionkey = \local_quizanalytics_quiz_cache_helper::selection_key(
            array_map(fn($quiz) => (int) $quiz->id, $stackquizzes)
        );
        self::$progressselectionkey = $selectionkey;

        $coursestats = \local_quizanalytics_quiz_cache_helper::stats_for_quizzes($stackquizzes);
        if ($coursestats->count === 0) {
            return;
        }

        $cache = \cache::make('local_quizanalytics', 'quizanalysiscoursewide');
        $fingerprint = $coursestats->fingerprint;
        $key = \local_quizanalytics_quiz_cache_helper::build_key(
            'course-ui-v5', $courseid, $coursestats->fingerprint, $selectionkey, $gradetype, $colorblind, $anonymize
        );
        $existing = $cache->get($key);
        if ($existing !== false) {
            // Keep the stable fallback populated even when this exact
            // fingerprint was already warmed by another path.
            $latestkey = \local_quizanalytics_quiz_cache_helper::build_key(
                'course-ui-latest-v2', $courseid, $selectionkey, $gradetype, $colorblind, $anonymize
            );
            $cache->set($latestkey, $existing);
            self::set_progress($courseid, $fingerprint, $gradetype, $colorblind, $anonymize,
                'complete', 'complete', count($stackquizzes), count($stackquizzes),
                get_string('progresscomplete', 'local_quizanalytics'));
            return;
        }

        self::set_progress($courseid, $fingerprint, $gradetype, $colorblind, $anonymize,
            'running', 'preparing', 0, count($stackquizzes),
            get_string('progresspreparing', 'local_quizanalytics'));

        // Same forked-worker-pool fetch warm_analytics_cache's own
        // warm_course() uses, not the plain serial
        // get_course_response_records() — this runs in CLI/cron context
        // just the same (adhoc tasks execute via admin/cli/adhoc_task.php,
        // the same runner scheduled tasks use), so there's no reason this
        // background compute should be slower or more memory-concentrated
        // than the equivalent scheduled-task run of the same course.
        $workers = max(1, (int) (get_config('local_quizanalytics', 'parallelworkers') ?: 4));
        $fetchstarted = microtime(true);
        $slowestitem = ['name' => '', 'seconds' => 0.0];
        try {
            $progresscallback = function(int $completed, int $total, array $itemdetails = []) use (
                $courseid, $fingerprint, $gradetype, $colorblind, $anonymize, &$slowestitem
            ): void {
                $itemseconds = (float) ($itemdetails['seconds'] ?? 0.0);
                if ($itemseconds > $slowestitem['seconds']) {
                    $slowestitem = [
                        'name' => (string) ($itemdetails['item'] ?? 'worker chunk'),
                        'seconds' => $itemseconds,
                    ];
                }
                self::set_progress($courseid, $fingerprint, $gradetype, $colorblind, $anonymize,
                    'running', 'processing', $completed, $total,
                    get_string('progressprocessing', 'local_quizanalytics', (object) [
                        'completed' => $completed, 'total' => $total,
                    ]), [
                        'last_item' => $itemdetails['item'] ?? 'worker chunk',
                        'last_item_seconds' => $itemseconds,
                        'last_item_records' => (int) ($itemdetails['records'] ?? 0),
                        'slowest_item' => $slowestitem,
                    ]);
            };
            $byquiz = parallel_course_fetcher::fetch($course, $stackquizzes, $workers, $progresscallback);
        } catch (\Throwable $e) {
            self::set_progress($courseid, $fingerprint, $gradetype, $colorblind, $anonymize,
                'failed', 'failed', 0, count($stackquizzes), get_string('progressfailed', 'local_quizanalytics'));
            mtrace('local_quizanalytics: warm_single_view_adhoc_task could not fetch course '
                . $courseid . ': ' . $e->getMessage());
            return; // Leave the cache cold — a future dispatch or cron run can retry.
        }
        $byquiz = array_filter($byquiz, fn($records) => !empty($records));
        $fetchtime = round(microtime(true) - $fetchstarted, 2);
        self::set_progress($courseid, $fingerprint, $gradetype, $colorblind, $anonymize,
            'running', 'analyzing', count($stackquizzes), count($stackquizzes),
            get_string('progressanalyzing', 'local_quizanalytics'), [
                'timings' => ['response_fetch_seconds' => $fetchtime],
                'slowest_item' => $slowestitem,
            ]);
        $analysisstarted = microtime(true);
        $analysiscallback = function(string $metric, float $seconds) use (
            $courseid, $fingerprint, $gradetype, $colorblind, $anonymize
        ): void {
            self::set_progress($courseid, $fingerprint, $gradetype, $colorblind, $anonymize,
                'running', 'analyzing', 0, 0,
                get_string('progressanalyzing', 'local_quizanalytics'), [
                    'current_metric' => $metric,
                    'current_metric_seconds' => round($seconds, 2),
                ]);
        };
        $result = $client->analyze_course(
            $course->fullname, $byquiz, $colorblind, $gradetype, $anonymize, [], $analysiscallback
        );
        $analysistime = round(microtime(true) - $analysisstarted, 2);
        self::set_progress($courseid, $fingerprint, $gradetype, $colorblind, $anonymize,
            'running', 'analyzing', count($stackquizzes), count($stackquizzes),
            get_string('progressanalyzing', 'local_quizanalytics'), [
                'timings' => ['response_fetch_seconds' => $fetchtime, 'analytics_seconds' => $analysistime],
            ]);
        if ($result !== null) {
            self::set_progress($courseid, $fingerprint, $gradetype, $colorblind, $anonymize,
                'running', 'saving', count($stackquizzes), count($stackquizzes),
                get_string('progresssaving', 'local_quizanalytics'));
            $cache->set($key, $result);
            $latestkey = \local_quizanalytics_quiz_cache_helper::build_key(
                'course-ui-latest-v2', $courseid, $selectionkey, $gradetype, $colorblind, $anonymize
            );
            $cache->set($latestkey, $result);
            self::set_progress($courseid, $fingerprint, $gradetype, $colorblind, $anonymize,
                'complete', 'complete', count($stackquizzes), count($stackquizzes),
                get_string('progresscomplete', 'local_quizanalytics'));
        } else {
            self::set_progress($courseid, $fingerprint, $gradetype, $colorblind, $anonymize,
                'failed', 'failed', count($stackquizzes), count($stackquizzes),
                get_string('progressfailed', 'local_quizanalytics'));
        }
    }
}
