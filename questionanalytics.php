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
 * The Question Analytics section: a single STACK quiz's Question Analytics
 * or Solution Process Visualization (?view=question|solutionprocess picks
 * which, defaulting to Question Analytics), picked via the required quiz
 * selector at the top of the page. Split out of what used to be
 * index.php's own ?quizid=... drill-down branch — Quiz Analytics
 * (index.php) is now the course-wide comparison only, and this is where
 * per-quiz analytics live instead.
 *
 * Reached from the "Analytics" link this plugin adds to each STACK quiz's
 * own settings menu (lib.php's
 * local_quizanalytics_extend_settings_navigation(), which links
 * straight here with &quizid= already set), from the "Section:" selector
 * at the top of every page in this plugin, or directly via
 * /local/quizanalytics/questionanalytics.php?id=<courseid>.
 *
 * @package local_quizanalytics
 * @copyright  2026 Ernest Ting <eting@caltech.edu>
 * @license    http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */

require_once(__DIR__ . '/../../config.php');
require_once($CFG->dirroot . '/local/quizanalytics/classes/quiz/data_fetcher.php');
require_once($CFG->dirroot . '/local/quizanalytics/classes/quiz/api_client.php');
require_once($CFG->dirroot . '/local/quizanalytics/classes/quiz/cache_helper.php');
require_once($CFG->dirroot . '/local/quizanalytics/classes/quiz/prepared_store.php');
require_once($CFG->dirroot . '/local/quizanalytics/classes/section_selector.php');

use local_quizanalytics\quiz\output\sections_output_helper;
use local_quizanalytics\stack\local\stack_course_helper;

$courseid = required_param('id', PARAM_INT);
$quizid   = optional_param('quizid', 0, PARAM_INT);

$course = $DB->get_record('course', ['id' => $courseid], '*', MUST_EXIST);

require_login($course);
$context = context_course::instance($course->id);
require_capability('local/quizanalytics:view', $context);

$PAGE->set_url('/local/quizanalytics/questionanalytics.php', ['id' => $courseid, 'quizid' => $quizid]);
$PAGE->set_pagelayout('report');
$PAGE->set_context($context);
$PAGE->set_title($course->shortname . ': ' . get_string('pagetitle', 'local_quizanalytics'));
$PAGE->set_heading($course->fullname);

$stackquizzes = local_quizanalytics_quiz_data_fetcher::get_course_stack_quizzes($course->id);

echo $OUTPUT->header();
echo $OUTPUT->heading(get_string('pagemaintitle', 'local_quizanalytics'));
echo local_quizanalytics_section_selector::render($courseid, 'question');

// Course, then Quiz, then View, then toggles — one selector per row in this
// fixed order on every section of this plugin, so switching sections
// doesn't reshuffle where each control sits.
$viewablecourses = stack_course_helper::get_viewable_courses();
if (count($viewablecourses) > 1) {
    $courseoptions = [];
    foreach ($viewablecourses as $viewablecourse) {
        $courseoptions[$viewablecourse->id] = format_string($viewablecourse->fullname);
    }
    $courseselector = new single_select(
        new moodle_url('/local/quizanalytics/questionanalytics.php'),
        'id',
        $courseoptions,
        $courseid,
        null
    );
    $courseselector->label = get_string('courseselectorlabel', 'local_quizanalytics');
    echo html_writer::div($OUTPUT->render($courseselector), 'mb-3');
}

if (empty($stackquizzes)) {
    echo $OUTPUT->notification(get_string('nostackquizzes', 'local_quizanalytics'), 'notifymessage');
    echo $OUTPUT->footer();
    exit;
}

// This section's whole purpose is per-quiz drill-down, so — unlike the old
// combined index.php, where 0 meant "show the course-wide view instead" —
// landing here with no quiz picked defaults to the first STACK quiz in the
// course rather than showing an empty "pick one" state.
if ($quizid === 0 || !array_key_exists($quizid, $stackquizzes)) {
    $quizid = (int) array_key_first($stackquizzes);
}

$selectoptions = [];
foreach ($stackquizzes as $quiz) {
    $selectoptions[$quiz->id] = $quiz->name;
}

$quizselector = new single_select(
    new moodle_url('/local/quizanalytics/questionanalytics.php', ['id' => $courseid]),
    'quizid',
    $selectoptions,
    $quizid,
    null
);
// Same "Quiz:" label as Model Analytics/Diagnostics Analytics' own quiz
// selectors — this used to say "View a single quiz's analytics" instead.
$quizselector->label = get_string('quizselectorlabel', 'local_quizanalytics');
echo html_writer::div($OUTPUT->render($quizselector), 'mb-3');

$selectedquiz = $stackquizzes[$quizid];

// The "View:" sub-selector used to also offer Solution Process
// Visualization here. Question Analytics is currently the only
// teacher-facing individual-quiz workflow — the previous Solution Process
// view remains in the codebase below (for possible redesign) but is
// intentionally not exposed via a selector any more. Still read from
// ?view= (defaulting to 'question') rather than hardcoded, so it stays
// reachable by direct URL as documented in README.md/this file's own
// docblock — only the on-screen <select> is gone, not the URL param.
$view = optional_param('view', 'question', PARAM_ALPHA);
if ($view !== 'question' && $view !== 'solutionprocess') {
    $view = 'question';
}
$PAGE->url->param('view', $view);

$colorblind = sections_output_helper::resolve_colorblind_mode();
$anonymize = sections_output_helper::resolve_anonymize_mode();
echo sections_output_helper::render_options_toggles($colorblind, $anonymize);

$client = new local_quizanalytics_quiz_api_client();

echo $OUTPUT->heading($selectedquiz->name, 3, 'main mt-4 mb-3');

// Cheap fingerprint first — a cache hit below skips the expensive
// per-attempt DB fetch entirely. $records is fetched lazily (at most once)
// since either view below might independently need it on its own cache
// miss. $snapshot is a compact, Moodle-native summary (attempt counts,
// overall average, per-question Facility Index) read straight from SQL —
// cheap enough to compute on every request, unlike the full per-attempt
// $records fetch below.
$snapshot = local_quizanalytics_quiz_data_fetcher::get_quiz_snapshot($selectedquiz, $course);
$stats = local_quizanalytics_quiz_cache_helper::stats_for_quiz($selectedquiz);
$progressurl = (new moodle_url('/local/quizanalytics/progress.php', [
    'id' => $courseid,
    'fingerprint' => $stats->fingerprint,
    'gradetype' => 'Question Analytics',
    'colorblind' => (int) $colorblind,
    'anonymize' => (int) $anonymize,
]))->out(false);
if ($stats->count === 0) {
    echo $OUTPUT->notification(get_string('noattempts', 'local_quizanalytics'), 'notifymessage');
    echo $OUTPUT->footer();
    exit;
}

$records = null;
$fetchrecords = function () use (&$records, $selectedquiz, $course, $stats): array {
    return $records ??= local_quizanalytics_quiz_data_fetcher::get_response_records_for_quiz_cached(
        $selectedquiz,
        $course,
        $stats->fingerprint
    );
};

if ($view === 'question') {
    $qacache = cache::make('local_quizanalytics', 'questionanalysis');
    $qakey = local_quizanalytics_quiz_cache_helper::build_key(
        $selectedquiz->id,
        $stats->fingerprint,
        md5(json_encode($snapshot)),
        $colorblind,
        $anonymize
    );
    $result = $qacache->get($qakey);
    $showingstale = false;
    if ($result === false) {
        $prepared = \local_quizanalytics_prepared_store::get($courseid, $quizid, 'question');
        $preparedpayload = \local_quizanalytics_prepared_store::decode($prepared);
        if ($preparedpayload !== null) {
            if (\local_quizanalytics_prepared_store::is_fresh($prepared, $stats->fingerprint)) {
                $result = $preparedpayload;
                $qacache->set($qakey, $result);
            } else {
                $result = $preparedpayload;
                $showingstale = true;
                \local_quizanalytics\task\warm_single_view_adhoc_task::dispatch_next_question_quiz_for_course($courseid);
            }
        }
    }
    if ($result === false) {
        // Times a real, small sample fetch for this quiz on this host —
        // see estimate_seconds_per_attempt()'s own comment for why a fixed
        // attempt-count threshold doesn't generalize across question
        // complexity or host speed the way an actual measured rate does.
        $samplerate = local_quizanalytics_quiz_data_fetcher::estimate_seconds_per_attempt($selectedquiz, $course);
        $estimatedseconds = $samplerate !== null ? $samplerate * $stats->count : 0.0;
        if (sections_output_helper::should_defer_to_background($estimatedseconds)) {
            // This quiz is large enough that a cold compute risks outliving
            // a reverse proxy's own timeout before ignore_user_abort(true)
            // below would even get a chance to help — see
            // warm_single_view_adhoc_task's own docblock. Hand it to a
            // background task and let the visitor come back to a warm
            // cache instead of blocking this request on it.
            \local_quizanalytics\task\warm_single_view_adhoc_task::dispatch_for_quiz($selectedquiz->id, $colorblind, $anonymize);
            $age = \local_quizanalytics\task\warm_single_view_adhoc_task::get_queued_age_seconds([
                'type' => 'quiz', 'id' => $selectedquiz->id, 'colorblind' => $colorblind, 'anonymize' => $anonymize,
            ]);
            sections_output_helper::render_generating_in_background_notice($age, $progressurl);
            echo $OUTPUT->footer();
            exit;
        }
        // See index.php's own comment on this same pattern: finish the
        // compute even if the browser/reverse proxy gives up first, so the
        // cache actually ends up warm for the next viewer instead of every
        // request redoing the same expensive work from scratch.
        sections_output_helper::flush_computing_notice();
        $previousabort = ignore_user_abort(true);
        $result = $client->analyze($selectedquiz->name, $fetchrecords(), $colorblind, $anonymize, $snapshot);
        if ($result !== null) {
            $qacache->set($qakey, $result);
        }
        ignore_user_abort($previousabort);
    }

    if (is_array($result)) {
        $reviewurl = (new moodle_url('/local/quizanalytics/questionreview.php', [
            'id' => $courseid,
            'quizid' => $quizid,
            'colorblind' => (int) $colorblind,
            'anonymize' => (int) $anonymize,
        ]))->out(false);
        $result = \local_quizanalytics\quiz\analytics\question_analysis::to_lightweight_review($result, $reviewurl);
    }

    if ($result === null) {
        echo $OUTPUT->notification(get_string('servererror', 'local_quizanalytics'), 'notifyproblem');
        echo $OUTPUT->footer();
        exit;
    }

    if ($showingstale) {
        echo $OUTPUT->notification(get_string('showingstale', 'local_quizanalytics'), 'notifymessage');
    }

    echo sections_output_helper::render_containers('qa');
    echo sections_output_helper::render_vendor_and_payload('qa', $result);
    echo sections_output_helper::render_hide_loading_notice();

    echo $OUTPUT->heading(get_string('generatepdfheading', 'local_quizanalytics'), 3, 'main mt-4 mb-3');
    echo sections_output_helper::render_pdf_form(
        new moodle_url('/local/quizanalytics/questionanalyticspdf.php'),
        [
            'id' => $courseid,
            'kind' => 'question',
            'quizid' => $quizid,
            'colorblind' => $colorblind ? 1 : 0,
            'anonymize' => $anonymize ? 1 : 0,
        ],
        $client->report_sections('question'),
        get_string('downloadpdfbutton', 'local_quizanalytics'),
        'qa-pdf',
        'qa'
    );
} else {
    // Solution Process Visualization.
    $metacache = cache::make('local_quizanalytics', 'solutionprocessmeta');
    $metakey = local_quizanalytics_quiz_cache_helper::build_key($selectedquiz->id, $stats->fingerprint, $anonymize);
    $meta = $metacache->get($metakey);
    if ($meta === false) {
        $samplerate = local_quizanalytics_quiz_data_fetcher::estimate_seconds_per_attempt($selectedquiz, $course);
        $estimatedseconds = $samplerate !== null ? $samplerate * $stats->count : 0.0;
        if (sections_output_helper::should_defer_to_background($estimatedseconds)) {
            \local_quizanalytics\task\warm_single_view_adhoc_task::dispatch_for_quiz_meta($selectedquiz->id, $anonymize);
            $age = \local_quizanalytics\task\warm_single_view_adhoc_task::get_queued_age_seconds([
                'type' => 'quizmeta', 'id' => $selectedquiz->id, 'anonymize' => $anonymize,
            ]);
            sections_output_helper::render_generating_in_background_notice($age);
            echo $OUTPUT->footer();
            exit;
        }
        sections_output_helper::flush_computing_notice();
        $previousabort = ignore_user_abort(true);
        $meta = $client->solution_process_meta($selectedquiz->name, $fetchrecords(), $anonymize);
        if ($meta !== null) {
            $metacache->set($metakey, $meta);
        }
        ignore_user_abort($previousabort);
    }

    if ($meta === null) {
        echo $OUTPUT->notification(get_string('servererror', 'local_quizanalytics'), 'notifyproblem');
        echo $OUTPUT->footer();
        exit;
    }
    if (empty($meta['questions'])) {
        echo $OUTPUT->notification(get_string('nostackquestions', 'local_quizanalytics'), 'notifymessage');
        echo $OUTPUT->footer();
        exit;
    }

    // Resolve the current question/part/student selection, validating
    // every GET param against what meta() actually returned rather than
    // trusting the request.
    $questionnames = array_column($meta['questions'], 'name');
    $spvquestion = optional_param('spvquestion', $questionnames[0], PARAM_RAW);
    if (!in_array($spvquestion, $questionnames, true)) {
        $spvquestion = $questionnames[0];
    }

    $spvpartsforquestion = 1;
    foreach ($meta['questions'] as $q) {
        if ($q['name'] === $spvquestion) {
            $spvpartsforquestion = max(1, (int) $q['parts']);
            break;
        }
    }
    $spvpart = optional_param('spvpart', 1, PARAM_INT);
    if ($spvpart < 1 || $spvpart > $spvpartsforquestion) {
        $spvpart = 1;
    }

    $spvstudentid = optional_param('spvstudent', '', PARAM_RAW);
    if ($spvstudentid !== '') {
        $spvvalidstudent = false;
        foreach ($meta['students'] as $s) {
            if ($s['id'] === $spvstudentid) {
                $spvvalidstudent = true;
                break;
            }
        }
        if (!$spvvalidstudent) {
            $spvstudentid = '';
        }
    }

    $PAGE->url->params([
        'spvquestion' => $spvquestion,
        'spvpart'     => $spvpart,
        'spvstudent'  => $spvstudentid,
    ]);

    echo sections_output_helper::render_solutionprocess_selector_form(
        $PAGE->url,
        $meta,
        $spvquestion,
        $spvpartsforquestion,
        $spvpart,
        $spvstudentid
    );

    $resultcache = cache::make('local_quizanalytics', 'solutionprocess');
    $resultkey = local_quizanalytics_quiz_cache_helper::build_key(
        $selectedquiz->id,
        $stats->fingerprint,
        $spvquestion,
        $spvpart,
        $spvstudentid,
        $colorblind,
        $anonymize
    );
    $result = $resultcache->get($resultkey);
    if ($result === false) {
        // Solution Process Visualization (tree edit distance, 3D figures,
        // network graphs) is the most expensive of this plugin's views —
        // the one this pattern matters most for.
        sections_output_helper::flush_computing_notice();
        $previousabort = ignore_user_abort(true);
        $result = $client->solution_process_analyze(
            $selectedquiz->name,
            $fetchrecords(),
            $spvquestion,
            $spvpart,
            $spvstudentid !== '' ? $spvstudentid : null,
            $colorblind,
            $anonymize
        );
        if ($result !== null) {
            $resultcache->set($resultkey, $result);
        }
        ignore_user_abort($previousabort);
    }

    if ($result === null) {
        echo $OUTPUT->notification(get_string('servererror', 'local_quizanalytics'), 'notifyproblem');
        echo $OUTPUT->footer();
        exit;
    }

    echo sections_output_helper::render_containers('spv');
    echo sections_output_helper::render_vendor_and_payload('spv', $result);
    echo sections_output_helper::render_hide_loading_notice();

    echo $OUTPUT->heading(get_string('generatepdfheading', 'local_quizanalytics'), 3, 'main mt-4 mb-3');
    echo sections_output_helper::render_pdf_form(
        new moodle_url('/local/quizanalytics/questionanalyticspdf.php'),
        [
            'id'          => $courseid,
            'kind'        => 'solutionprocess',
            'quizid'      => $quizid,
            'spvquestion' => $spvquestion,
            'spvpart'     => $spvpart,
            'colorblind'  => $colorblind ? 1 : 0,
            'anonymize'   => $anonymize ? 1 : 0,
        ],
        $client->report_sections('solutionprocess'),
        get_string('downloadpdfbutton', 'local_quizanalytics'),
        'spv-pdf',
        'spv'
    );
}

echo $OUTPUT->footer();
