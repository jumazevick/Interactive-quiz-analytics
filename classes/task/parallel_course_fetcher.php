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
 * Fans a course's STACK quizzes' response-record fetch out across forked
 * worker processes, CLI/cron context only.
 *
 * Why this exists: profiling a real 38-quiz/48,445-attempt course showed
 * the fetch stage (data_fetcher.php's get_response_records_for_quiz())
 * completely dominating runtime — ~10-20x the entire analysis pipeline
 * combined for the same quiz. The reason isn't a query or algorithm this
 * plugin controls: Moodle's own qtype_stack calls get_response_summary()
 * -> summarise_response() -> get_prt_result(), which re-runs the PRT's
 * actual CAS/Maxima grading logic live, every time, for every attempt —
 * confirmed directly against question/type/stack/question.php's
 * validate_cache()/get_prt_result(), which only memoizes repeat calls
 * *within one already-hydrated question object for the same response*, not
 * across the many separate question objects this plugin's batch fetch
 * creates (one per attempt). There's no "read the already-graded result
 * without recomputing" path through the standard question-engine API.
 *
 * What IS true, though: those CAS calls are I/O-bound (waiting on the
 * Maxima backend), and this site's goemaxima worker pool
 * (docker-compose.yml's GOEMAXIMA_QUEUE_LEN) sits almost entirely idle
 * during a fetch, since the whole plugin makes these calls strictly
 * sequentially, one attempt at a time, in one PHP process. Running several
 * quizzes' fetches concurrently in separate forked processes lets that
 * pool actually do concurrent work, without touching a single line of the
 * fetch logic itself (get_response_records_for_quiz() is called completely
 * unmodified in every child).
 *
 * pcntl_fork() only works in CLI/cron context (Apache/mod_php can't fork),
 * so this is deliberately only ever called from the warm_analytics_cache
 * scheduled task — the synchronous on-demand path (index.php,
 * questionanalytics.php) is untouched and keeps fetching serially on a
 * cold-cache request, same as before this existed.
 *
 * @package local_quizanalytics
 * @copyright  2026 Ernest Ting <eting@caltech.edu>
 * @license    http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */

namespace local_quizanalytics\task;

defined('MOODLE_INTERNAL') || die();

require_once($CFG->dirroot . '/local/quizanalytics/classes/quiz/data_fetcher.php');
require_once($CFG->dirroot . '/local/quizanalytics/classes/quiz/cache_helper.php');

/**
 * Forked-worker-pool fetch coordinator for one course's STACK quizzes.
 */
class parallel_course_fetcher {
    /**
     * Fetches response records for every given quiz in $course, running up
     * to $workers quizzes' worth of fetch work concurrently in separate
     * forked processes when possible.
     *
     * Falls back to a single-process sequential fetch (identical to what
     * this method's forked path would produce, just slower) whenever
     * forking isn't available or wouldn't help: no pcntl, $workers <= 1,
     * or only one quiz to fetch. That fallback path is exactly
     * data_fetcher.php's own get_course_response_records() — the same,
     * already-correct, already-tested code this whole class exists to
     * parallelize, never a reimplementation of it.
     *
     * @param \stdClass $course
     * @param \stdClass[] $stackquizzes quiz records to fetch, keyed by quiz id
     * @param int $workers maximum number of concurrent forked processes
     * @param callable|null $progresscallback optional callback receiving
     *        ($completed, $total, $details) in the parent process
     * @return array [quiz_name => records[]]
     * @throws \Exception if any worker failed — the caller should treat
     *         that as "could not warm this course this run", not cache a
     *         partial/incomplete result.
     */
    public static function fetch(\stdClass $course, array $stackquizzes, int $workers, ?callable $progresscallback = null): array {
        if ($workers <= 1 || count($stackquizzes) <= 1 || !function_exists('pcntl_fork')) {
            return \local_quizanalytics_quiz_data_fetcher::get_course_response_records(
                $course, $stackquizzes, $progresscallback
            );
        }

        $chunks = array_values(array_filter(
            self::balance_chunks($stackquizzes, $workers),
            fn($chunk) => !empty($chunk)
        ));
        if (count($chunks) <= 1) {
            // Balancing collapsed everything into one bucket (e.g. only one
            // quiz has any attempts) — forking for a single chunk of work
            // only adds overhead, so run it inline instead.
            return \local_quizanalytics_quiz_data_fetcher::get_course_response_records(
                $course, $stackquizzes, $progresscallback
            );
        }

        $tmpdir = make_temp_directory('local_quizanalytics_parallel');
        $children = []; // pid => tmpfile path.
        $anyfailed = false;

        // Disposing the parent's own connection *before* forking, not just
        // reconnecting each child afterward, matters: fork() duplicates the
        // process's open file descriptors, so every child briefly holds a
        // second handle onto the exact same underlying OS socket the parent
        // is still using. A child reconnecting doesn't remove that risk by
        // itself. Tried removing this step entirely at one point this
        // session, reasoning that the child's own SIGKILL-based exit (see
        // that branch's own comment) would avoid ever disposing the child's
        // inherited copy — measured directly that this did *not* hold at
        // real 50-quiz scale: the parent's own $DB (confirmed same
        // spl_object_id() before and after, ruling out some other code
        // reassigning it) started throwing "MySQL server has gone away"
        // immediately after the very first fork/reconnect round regardless,
        // and a targeted A/B check (workers=1, no forking at all, on the
        // exact same course) confirmed the connection stays healthy without
        // forking — so this is specifically a forking-related corruption,
        // just not fully explained by the destructor/shutdown-function
        // theory that motivated removing this step. Restored it since it's
        // the one thing empirically proven, across many runs this session,
        // to actually prevent the corruption — even though the exact
        // OS/TCP-level mechanism isn't fully pinned down. Disposing here
        // means nobody has the connection open at the instant fork()
        // actually runs, so there's nothing left to share or corrupt; the
        // parent reconnects its own fresh copy once every child has
        // exited, below.
        //
        // Known, accepted remaining rough edge from reconnecting to a new
        // object here rather than the original: \core\task\manager (when
        // this runs under real cron, not a standalone script) acquires a
        // lock in the *parent* process before this task's own execute()
        // ever runs, holding a direct reference to the $DB object in use at
        // that time to release it later — disposing that object here and
        // swapping to a genuinely different reconnected instance leaves
        // that lock's own reference pointing at a dead connection, so its
        // later release() can fail ("Call to a member function
        // real_escape_string() on null"), which can show this task as
        // "failed" in Moodle's task log despite the actual fetch/cache work
        // above having completed and been cached correctly. Prioritized
        // the connection-safety fix over this one, since a corrupted
        // connection risks real data-fetch failures for whatever a visitor
        // or the next scheduled task does next, where this is a
        // log-accuracy issue on work that already genuinely succeeded.
        global $DB;
        $DB->dispose();
        $parentreconnected = false;

        foreach ($chunks as $index => $chunk) {
            $tmpfile = $tmpdir . '/chunk_' . $index . '_' . uniqid('', true) . '.dat';
            $pid = pcntl_fork();

            if ($pid === -1) {
                // Fork failed (e.g. process limit) — run this chunk inline
                // in the parent rather than losing its data. The parent's
                // own connection was just disposed above, so it needs
                // reconnecting to do this chunk's DB work.
                self::reconnect_db();
                $parentreconnected = true;
                self::run_chunk_and_write($course, $chunk, $tmpfile);
                if (!self::read_and_delete($tmpfile, $partial)) {
                    $anyfailed = true;
                } else {
                    $byquiz = ($byquiz ?? []) + $partial;
                    if ($progresscallback !== null) {
                        $progresscallback(count($byquiz), count($stackquizzes), [
                            'item' => 'worker chunk', 'seconds' => null, 'records' => array_sum(array_map('count', $partial)),
                        ]);
                    }
                }
                continue;
            }

            if ($pid === 0) {
                // Child process: fork() gave this process its own private
                // copy of memory, so reassigning $DB here can never affect
                // the parent's. The connection this process inherited was
                // already disposed (in the parent, before forking), so
                // there's no live shared socket left to worry about either
                // way — reconnect_db() just needs to give this process its
                // own real, working connection to do its actual DB work.
                $exitcode = 0;
                try {
                    self::reconnect_db();
                    self::run_chunk_and_write($course, $chunk, $tmpfile);
                } catch (\Throwable $e) {
                    debugging(
                        'local_quizanalytics: parallel fetch worker failed: '
                            . get_class($e) . ': ' . $e->getMessage(),
                        DEBUG_DEVELOPER
                    );
                    $exitcode = 1;
                }
                // Not a plain exit($exitcode) — confirmed directly running
                // under real Moodle cron (this class is only ever reached
                // from CLI/cron context in the first place): \core\task\
                // manager acquires real, DB-connection-backed locks (a
                // per-task lock, plus the site-wide "core_cron" lock) in the
                // *parent* process before this task's own execute() ever
                // runs, each registered with its own auto-release shutdown
                // function. fork() duplicates that registration into every
                // child along with everything else in memory, so a plain
                // exit() here runs PHP's normal shutdown sequence and fires
                // those inherited handlers too — trying to release a lock
                // this child never acquired, through a $DB reference this
                // child has already disposed/reconnected out from under.
                // Observed directly: "Exception ignored in shutdown
                // function core\lock\mysql_lock_factory::auto_release: Call
                // to a member function real_escape_string() on null",
                // alongside the child itself exiting with status 1 despite
                // run_chunk_and_write() above having already completed and
                // flushed its result to $tmpfile successfully. Killing this
                // process directly skips PHP's shutdown sequence entirely
                // (no shutdown functions, inherited or otherwise, ever run)
                // — safe here specifically because run_chunk_and_write()'s
                // own finally block has already closed and flushed the temp
                // file before this point, so there is nothing left for a
                // normal shutdown to correctly finish anyway.
                if (function_exists('posix_kill') && function_exists('posix_getpid')) {
                    posix_kill(posix_getpid(), SIGKILL);
                }
                exit($exitcode); // Fallback if posix isn't available.
            }

            // Parent process.
            $children[$pid] = $tmpfile;
        }

        foreach (array_keys($children) as $pid) {
            // Only reaping here now, deliberately not treating the exit
            // status itself as a failure signal: every child, successful or
            // not, now terminates via a self-delivered SIGKILL (see the
            // child branch above for why) — pcntl_wifexited()/wexitstatus()
            // can no longer distinguish "finished cleanly" from "genuinely
            // crashed," since both now look identical from here (killed by
            // signal 9). The real, reliable success signal is whether each
            // child's own $tmpfile is a complete, well-formed result,
            // checked below via read_and_delete() — which already detects
            // a truncated/corrupt file, exactly what a *genuine* kernel OOM
            // kill (as opposed to this class's own intentional one) would
            // actually leave behind.
            $status = 0;
            pcntl_waitpid($pid, $status);
        }

        // The parent needs its own working connection again for whatever
        // the caller does next (analyze()/cache->set() calls both touch the
        // DB) — reconnect once, unless a fork-failure fallback above
        // already did.
        if (!$parentreconnected) {
            self::reconnect_db();
        }

        $byquiz = $byquiz ?? [];
        foreach ($children as $pid => $tmpfile) {
            if (!self::read_and_delete($tmpfile, $partial)) {
                // mtrace(), not debugging() — see warm_analytics_cache.php's
                // catch block for why: this task runs unattended on cron
                // with $CFG->debug typically at DEBUG_NONE in production,
                // where debugging() never surfaces anywhere. A missing or
                // truncated result file from a worker that no longer exists
                // is consistent with the kernel OOM killer taking it out
                // mid-write (no PHP-catchable exception at all in that
                // case) — not certain, since a worker can fail other ways
                // too, but worth surfacing as the likely first thing to
                // check.
                mtrace('local_quizanalytics: parallel fetch worker pid ' . $pid . ' for course ' . $course->id
                    . ' produced no usable result (missing/truncated ' . $tmpfile . ') — '
                    . 'possibly killed by the kernel OOM killer; consider lowering parallelworkers '
                    . 'or raising parallelworkermemory');
                $anyfailed = true;
                continue;
            }
            $byquiz += $partial;
            if ($progresscallback !== null) {
                $progresscallback(count($byquiz), count($stackquizzes), [
                    'item' => 'worker chunk', 'seconds' => null, 'records' => array_sum(array_map('count', $partial)),
                ]);
            }
        }

        if ($anyfailed) {
            throw new \Exception(
                'local_quizanalytics: one or more parallel fetch workers failed for course '
                . $course->id . '; not caching a partial result this run.'
            );
        }

        return $byquiz;
    }

    /**
     * Fetches one chunk's quizzes using the existing, unmodified per-quiz
     * fetch — get_response_records_for_quiz(), never
     * get_course_response_records() — and streams each quiz's result to the
     * temp file as soon as that quiz is done, instead of accumulating the
     * whole chunk in memory and serializing it all at once at the end.
     *
     * This matters at real scale: get_course_response_records() already
     * frees each quiz's own internal working state as it goes
     * (gc_collect_cycles() in data_fetcher.php), but it still holds every
     * quiz's *finished* records array in memory simultaneously until the
     * whole chunk is done — for a worker that draws several of a course's
     * largest quizzes, that combined total is what was exhausting a
     * generous, otherwise-plenty-for-one-quiz memory_limit. Writing (and
     * then unsetting) each quiz's records right after fetching it bounds
     * peak memory to roughly one quiz's worth, not a whole chunk's.
     *
     * Format: a sequence of [4-byte big-endian length][serialized
     * ['quizname' => records[]]] pairs, back to back — not one blob or
     * newline-delimited JSON, since response text can itself contain
     * arbitrary bytes including newlines.
     *
     * @param \stdClass $course
     * @param \stdClass[] $chunk
     * @param string $tmpfile
     */
    private static function run_chunk_and_write(\stdClass $course, array $chunk, string $tmpfile): void {
        $handle = fopen($tmpfile, 'wb');
        if ($handle === false) {
            throw new \Exception('local_quizanalytics: could not open worker result file for writing: ' . $tmpfile);
        }
        try {
            foreach ($chunk as $quiz) {
                $records = \local_quizanalytics_quiz_data_fetcher::get_response_records_for_quiz($quiz, $course);
                $blob = serialize([$quiz->name => $records]);
                fwrite($handle, pack('N', strlen($blob)));
                fwrite($handle, $blob);
                unset($records, $blob);
                gc_collect_cycles();
                // Moodle's own core_questiondata cache accumulates every
                // distinct question this process ever loads for the rest of
                // the process's life -- a reasonable assumption for a
                // short-lived web request, not a long CLI worker that can
                // touch hundreds of distinct STACK questions across many
                // quizzes. Measured directly: without this, a worker's
                // memory climbed continuously and never stopped, eventually
                // exhausting even a 2048M ceiling; with it, memory plateaus
                // once, staying flat instead of growing further. Purging
                // just re-fetches from the DB on the next distinct question
                // this process happens to touch — correct, just marginally
                // slower than an in-memory hit would be, which is a fair
                // trade against unbounded growth.
                \cache_helper::purge_by_definition('core', 'questiondata');
            }
        } finally {
            fclose($handle);
        }
    }

    /**
     * Reads, decodes, and deletes a worker's result file — the reverse of
     * run_chunk_and_write()'s length-prefixed-records format.
     *
     * @param string $tmpfile
     * @param array|null $out set to the decoded [quiz_name => records[]] array on success
     * @return bool false if the file was missing/unreadable/corrupt
     */
    private static function read_and_delete(string $tmpfile, ?array &$out): bool {
        if (!is_readable($tmpfile)) {
            $out = null;
            return false;
        }

        $handle = fopen($tmpfile, 'rb');
        if ($handle === false) {
            @unlink($tmpfile);
            $out = null;
            return false;
        }

        $merged = [];
        $corrupt = false;
        while (!feof($handle)) {
            $lengthbytes = fread($handle, 4);
            if ($lengthbytes === '' || $lengthbytes === false) {
                break; // Clean end of file.
            }
            if (strlen($lengthbytes) !== 4) {
                $corrupt = true;
                break;
            }
            $unpacked = unpack('Nlength', $lengthbytes);
            $length = $unpacked['length'];
            $blob = $length > 0 ? fread($handle, $length) : '';
            if ($blob === false || strlen($blob) !== $length) {
                $corrupt = true;
                break;
            }
            $decoded = @unserialize($blob, ['allowed_classes' => true]);
            if (!is_array($decoded)) {
                $corrupt = true;
                break;
            }
            $merged += $decoded; // Quiz names are unique per course — safe to merge.
        }
        fclose($handle);
        @unlink($tmpfile);

        if ($corrupt) {
            $out = null;
            return false;
        }
        $out = $merged;
        return true;
    }

    /**
     * Opens a brand new database connection and rebinds the global $DB to
     * it, without ever touching the connection $DB pointed at before the
     * fork. Same driver/credentials setup_DB() itself uses
     * (lib/dmllib.php), just building a second, independent instance
     * instead of reusing the process-inherited one.
     */
    private static function reconnect_db(): void {
        global $CFG, $DB;
        $freshdb = \moodle_database::get_driver_instance($CFG->dbtype, $CFG->dblibrary, false);
        $freshdb->connect($CFG->dbhost, $CFG->dbuser, $CFG->dbpass, $CFG->dbname, $CFG->prefix, $CFG->dboptions);
        $DB = $freshdb;
    }

    /**
     * Greedily distributes quizzes across $workers buckets balanced by
     * finished-attempt count (a direct proxy for CAS/PRT-evaluation cost —
     * confirmed by profiling that fetch time scales with attempt count),
     * not plain quiz count — this course's own quizzes range from 41 to
     * 2,764 finished attempts each, so an even quiz-count split would leave
     * some workers with far more real work than others.
     *
     * @param \stdClass[] $quizzes
     * @param int $workers
     * @return array[] $workers buckets, each an array of quiz records
     */
    private static function balance_chunks(array $quizzes, int $workers): array {
        $withcounts = array_map(function ($quiz) {
            $stats = \local_quizanalytics_quiz_cache_helper::stats_for_quiz($quiz);
            return ['quiz' => $quiz, 'count' => $stats->count];
        }, array_values($quizzes));

        usort($withcounts, fn($a, $b) => $b['count'] <=> $a['count']);

        $buckets = array_fill(0, $workers, []);
        $bucketloads = array_fill(0, $workers, 0);
        foreach ($withcounts as $item) {
            $lightest = array_search(min($bucketloads), $bucketloads, true);
            $buckets[$lightest][] = $item['quiz'];
            $bucketloads[$lightest] += $item['count'];
        }
        return $buckets;
    }
}
