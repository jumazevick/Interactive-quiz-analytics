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
 * The Model Analytics section (ported from the standalone
 * local_stackanalytics plugin this merges): Model 1 (student risk &
 * behavior) and Model 2 (question/PRT quality) — the two ML-oriented models
 * from the architecture doc's own structure. The non-ML Diagnostics
 * Dashboard used to live on this same page behind a View: selector; it now
 * has its own diagnosticsanalytics.php, since it doesn't share Model 1/2's
 * "trained model" framing at all (it's direct calculations, not indicator
 * readings — see that file's own docblock). Both models ship disabled by
 * default (alpha stage — see db/analytics.php), so what this page shows is
 * each model's *live indicator readings*, not a trained model's
 * predictions; those, once an administrator enables and trains a model,
 * live in Moodle's own Site Administration > Analytics > Insights instead.
 *
 * Reached from the "Section:" selector at the top of every page in this
 * plugin, or directly via
 * /local/quizanalytics/modelanalytics.php?id=<courseid>.
 *
 * Renamed from models.php when Diagnostics split into its own section.
 *
 * @package local_quizanalytics
 * @copyright  2026 Ernest Ting <eting@caltech.edu>
 * @license    http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */

require_once(__DIR__ . '/../../config.php');
require_once($CFG->dirroot . '/local/quizanalytics/classes/section_selector.php');

use local_quizanalytics\stack\local\stack_course_helper;
use local_quizanalytics\stack\analytics\report\model1_report;
use local_quizanalytics\stack\analytics\report\model2_report;
use local_quizanalytics\stack\output\dashboard_renderer;

$courseid = required_param('id', PARAM_INT);
$course = $DB->get_record('course', ['id' => $courseid], '*', MUST_EXIST);

require_login($course);
$context = context_course::instance($course->id);
require_capability('local/quizanalytics:view', $context);

$PAGE->set_url('/local/quizanalytics/modelanalytics.php', ['id' => $courseid]);
$PAGE->set_pagelayout('report');
$PAGE->set_context($context);
$PAGE->set_title($course->shortname . ': ' . get_string('dashboardtitle', 'local_quizanalytics'));
$PAGE->set_heading($course->fullname);

echo $OUTPUT->header();
echo $OUTPUT->heading(get_string('pagemaintitle', 'local_quizanalytics'));
echo local_quizanalytics_section_selector::render($courseid, 'models');

// One discreet, collapsed-by-default <details> (no alert-info/alert-warning
// colouring — a page-load-time reference note shouldn't visually compete
// with the actual selectors/toggles below it) combining what used to be two
// separate colored boxes: the "About Model 1 and Model 2" intro and the
// "Responsible use" callout. Every section's own version of this uses the
// same shape, positioned the same way, right after the section selector.
echo html_writer::tag(
    'details',
    html_writer::tag('summary', get_string('modelpageintrosummary', 'local_quizanalytics'))
        . html_writer::div(get_string('modelpageintro', 'local_quizanalytics'), 'mt-2')
        . html_writer::tag('p', html_writer::tag('strong', get_string('responsibleusesummary', 'local_quizanalytics')), ['class' => 'mt-3 mb-1'])
        . html_writer::div(get_string('responsibleusecallout', 'local_quizanalytics')),
    ['class' => 'mb-3']
);

// Course, then Quiz (where applicable), then View, then toggles — one
// selector per row in this fixed order on every section of this plugin, so
// switching sections doesn't reshuffle where each control sits.
$viewablecourses = stack_course_helper::get_viewable_courses();
if (count($viewablecourses) > 1) {
    $courseoptions = [];
    foreach ($viewablecourses as $viewablecourse) {
        $courseoptions[$viewablecourse->id] = format_string($viewablecourse->fullname);
    }
    $courseselector = new single_select(
        new moodle_url('/local/quizanalytics/modelanalytics.php'),
        'id',
        $courseoptions,
        $courseid,
        null
    );
    $courseselector->label = get_string('courseselectorlabel', 'local_quizanalytics');
    echo html_writer::div($OUTPUT->render($courseselector), 'mb-3');
}

$slots = stack_course_helper::get_course_stack_slots($courseid);

if (empty($slots)) {
    echo $OUTPUT->notification(get_string('errornostackactivity', 'local_quizanalytics'), 'notifymessage');
    echo $OUTPUT->footer();
    exit;
}

// One section rendered per page load, not both at once — the previous
// anchor-link "Jump to section" nav still rendered (and queried) everything
// on every load, which is what actually made the page unusably long on a
// course with many students/questions.
$view = optional_param('view', 'model1', PARAM_ALPHANUM);
if (!in_array($view, ['model1', 'model2'], true)) {
    $view = 'model1';
}

// Narrows Model 2 (per-question) to one quiz at a time — Model 1 isn't
// quiz-scoped (its indicators are per-student), so this selector only shows
// on that view. Resolved and rendered here, in the Quiz: row, ahead of the
// View: selector below it — previously this sat after View: instead,
// inside Model 2's own branch further down.
$quizid = optional_param('quizid', 0, PARAM_INT);
if ($view === 'model2') {
    $quiznames = $DB->get_records_menu('quiz', ['course' => $courseid], '', 'id, name');
    $slotsperquiz = [];
    foreach ($slots as $slot) {
        $slotsperquiz[(int) $slot->quizid] = ($slotsperquiz[(int) $slot->quizid] ?? 0) + 1;
    }
    if (count($slotsperquiz) > 1) {
        // Each option names both the quiz and how many STACK questions it
        // has, so the list reads as "this quiz, with this much STACK
        // content" rather than a bare, unfamiliar-looking name.
        $quizoptions = [];
        foreach ($slotsperquiz as $slotquizid => $questioncount) {
            $quizname = format_string($quiznames[$slotquizid] ?? get_string('unknownquiz', 'local_quizanalytics'));
            $quizoptions[$slotquizid] = get_string('quizoptionlabel', 'local_quizanalytics', (object) [
                'name' => $quizname,
                'count' => $questioncount,
            ]);
        }
        asort($quizoptions);
        $quizselector = new single_select(
            new moodle_url('/local/quizanalytics/modelanalytics.php', ['id' => $courseid, 'view' => $view]),
            'quizid',
            $quizoptions,
            $quizid,
            [0 => get_string('allquizzes', 'local_quizanalytics')]
        );
        $quizselector->label = get_string('quizselectorlabel', 'local_quizanalytics');
        echo html_writer::div($OUTPUT->render($quizselector), 'mb-3');
    } else {
        $quizid = 0; // Only one quiz in this course — nothing to filter.
    }
}

$viewoptions = [
    'model1' => get_string('model1heading', 'local_quizanalytics'),
    'model2' => get_string('model2heading', 'local_quizanalytics'),
];
$viewselector = new single_select(
    new moodle_url('/local/quizanalytics/modelanalytics.php', ['id' => $courseid]),
    'view',
    $viewoptions,
    $view,
    null
);
$viewselector->label = get_string('viewselectorlabel', 'local_quizanalytics');
echo html_writer::div($OUTPUT->render($viewselector), 'mb-3');

// Shares its user preference and lang string with Quiz Analytics's own
// anonymize toggle (classes/quiz/output/sections_output_helper.php) — one
// teacher preference for anonymization across every section of this
// plugin, not a separate on/off switch per page. Only meaningful on Model 1
// (Model 2's table is per-question, with no student names to anonymize).
$anonymize = (bool) optional_param('anonymize', 0, PARAM_INT);

if ($view === 'model1') {
    echo html_writer::start_tag('form', [
        'method' => 'get', 'action' => $PAGE->url->out_omit_querystring(), 'class' => 'mb-3',
    ]);
    foreach ($PAGE->url->params() as $name => $value) {
        if ($name === 'anonymize') {
            continue;
        }
        echo html_writer::empty_tag('input', ['type' => 'hidden', 'name' => $name, 'value' => $value]);
    }
    // An unchecked checkbox submits nothing at all, so this hidden 0 (kept
    // first, overridden by the checkbox's own '1' only when it's actually
    // checked) is what makes unchecking the box and submitting genuinely
    // clear the preference, not just leave it at its last value.
    echo html_writer::empty_tag('input', ['type' => 'hidden', 'name' => 'anonymize', 'value' => '0']);
    $anonymizeattrs = ['type' => 'checkbox', 'name' => 'anonymize', 'value' => '1', 'id' => 'ma-anonymize-toggle'];
    if ($anonymize) {
        $anonymizeattrs['checked'] = 'checked';
    }
    echo html_writer::empty_tag('input', $anonymizeattrs);
    echo ' ' . html_writer::label(get_string('anonymizemode', 'local_quizanalytics'), 'ma-anonymize-toggle');
    // Same "Apply" label as Quiz/Question Analytics' own toggle row
    // (sections_output_helper::render_options_toggles()) — this used to say
    // "View" here instead, for what's functionally the same action.
    echo ' ' . html_writer::empty_tag('input', [
        'type' => 'submit',
        'value' => get_string('apply', 'moodle'),
        'class' => 'btn btn-secondary btn-sm ml-2',
    ]);
    echo html_writer::end_tag('form');
}

if ($view === 'model1') {
    echo $OUTPUT->heading(get_string('model1heading', 'local_quizanalytics'), 3);
    echo html_writer::tag('p', get_string('model1intro', 'local_quizanalytics'));
    echo dashboard_renderer::render_model1_about();

    dashboard_renderer::flush_computing_notice();
    echo dashboard_renderer::render_model1_table(model1_report::build($courseid), $anonymize);
    echo dashboard_renderer::render_hide_loading_notice();
} else {
    echo $OUTPUT->heading(get_string('model2heading', 'local_quizanalytics'), 3);
    echo html_writer::tag('p', get_string('model2intro', 'local_quizanalytics'));
    echo dashboard_renderer::render_model2_about();
    dashboard_renderer::flush_computing_notice();
    echo dashboard_renderer::render_model2_table(model2_report::build($courseid, $quizid !== 0 ? $quizid : null));
    echo dashboard_renderer::render_hide_loading_notice();
}

echo dashboard_renderer::render_pdf_form(
    new moodle_url('/local/quizanalytics/modelanalyticspdf.php'),
    $courseid,
    $quizid !== 0 ? $quizid : null,
    $anonymize,
    [
        'model1' => 'model1heading',
        'model2' => 'model2heading',
    ]
);

echo $OUTPUT->footer();
