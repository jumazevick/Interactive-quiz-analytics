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
 * The Diagnostics Analytics section: the non-ML Diagnostics Dashboard
 * (seed-bias ANOVA and bloated-PRT-tree coverage), computed directly from
 * attempt data rather than through the Analytics API's ML machinery — these
 * are not predictions the way Model 1/Model 2's indicator readings are
 * framed, just direct calculations, so this doesn't share Model Analytics's
 * "trained model" framing at all. Used to live on models.php behind a
 * View: selector; split into its own section so it reads as what it is —
 * statistical reports, not a model — rather than a third option alongside
 * two actual models.
 *
 * Reached from the "Section:" selector at the top of every page in this
 * plugin, or directly via
 * /local/quizanalytics/diagnosticsanalytics.php?id=<courseid>.
 *
 * @package local_quizanalytics
 * @copyright  2026 Ernest Ting <eting@caltech.edu>
 * @license    http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */

require_once(__DIR__ . '/../../config.php');
require_once($CFG->dirroot . '/local/quizanalytics/classes/section_selector.php');

use local_quizanalytics\stack\local\stack_course_helper;
use local_quizanalytics\stack\diagnostics\concept_dependency_report;
use local_quizanalytics\stack\analytics\report\diagnostics_report;
use local_quizanalytics\stack\output\dashboard_renderer;

$courseid = required_param('id', PARAM_INT);
$course = $DB->get_record('course', ['id' => $courseid], '*', MUST_EXIST);

require_login($course);
$context = context_course::instance($course->id);
require_capability('local/quizanalytics:view', $context);

$PAGE->set_url('/local/quizanalytics/diagnosticsanalytics.php', ['id' => $courseid]);
$PAGE->set_pagelayout('report');
$PAGE->set_context($context);
$PAGE->set_title($course->shortname . ': ' . get_string('dashboardtitle', 'local_quizanalytics'));
$PAGE->set_heading($course->fullname);

echo $OUTPUT->header();
echo $OUTPUT->heading(get_string('pagemaintitle', 'local_quizanalytics'));
echo local_quizanalytics_section_selector::render($courseid, 'diagnostics');

// One discreet, collapsed-by-default <details> combining what used to be
// two separate colored boxes — the "About the Diagnostics Dashboard" intro
// and the "Responsible use" callout — same shape and position as every
// other section's own version of this.
echo html_writer::tag(
    'details',
    html_writer::tag('summary', get_string('diagnosticspageintrosummary', 'local_quizanalytics'))
        . html_writer::div(get_string('diagnosticspageintro', 'local_quizanalytics'), 'mt-2')
        . html_writer::tag('p', html_writer::tag('strong', get_string('responsibleusesummary', 'local_quizanalytics')), ['class' => 'mt-3 mb-1'])
        . html_writer::div(get_string('responsibleusecallout', 'local_quizanalytics')),
    ['class' => 'mb-3']
);

// Course, then Quiz — one selector per row in this fixed order on every
// section of this plugin, so switching sections doesn't reshuffle where
// each control sits.
$viewablecourses = stack_course_helper::get_viewable_courses();
if (count($viewablecourses) > 1) {
    $courseoptions = [];
    foreach ($viewablecourses as $viewablecourse) {
        $courseoptions[$viewablecourse->id] = format_string($viewablecourse->fullname);
    }
    $courseselector = new single_select(
        new moodle_url('/local/quizanalytics/diagnosticsanalytics.php'),
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

// Per-question, like Model 2 — narrows the report to one quiz at a time.
$quizid = optional_param('quizid', 0, PARAM_INT);

// Anonymization is request-controlled only. An old stored preference must
// never silently change the display of a new report.
$anonymize = (bool) optional_param('anonymize', 0, PARAM_INT);

$quiznames = $DB->get_records_menu('quiz', ['course' => $courseid], '', 'id, name');
$slotsperquiz = [];
foreach ($slots as $slot) {
    $slotsperquiz[(int) $slot->quizid] = ($slotsperquiz[(int) $slot->quizid] ?? 0) + 1;
}
if (count($slotsperquiz) > 1) {
    // Each option names both the quiz and how many STACK questions it has,
    // so the list reads as "this quiz, with this much STACK content" rather
    // than a bare, unfamiliar-looking name.
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
        new moodle_url('/local/quizanalytics/diagnosticsanalytics.php', ['id' => $courseid]),
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

echo $OUTPUT->heading(get_string('diagnosticsheading', 'local_quizanalytics'), 3);

$diagnosticsintrobody = html_writer::tag('p', get_string('diagnosticsintro', 'local_quizanalytics'));
if (!concept_dependency_report::is_available()) {
    $diagnosticsintrobody .= html_writer::tag(
        'p',
        get_string('conceptdependencynote', 'local_quizanalytics'),
        ['class' => 'text-muted small mb-0']
    );
}
echo html_writer::tag(
    'details',
    html_writer::tag('summary', get_string('diagnosticsintrosummary', 'local_quizanalytics'))
        . html_writer::div($diagnosticsintrobody, 'mt-2'),
    ['class' => 'mb-3']
);

dashboard_renderer::flush_computing_notice();
echo dashboard_renderer::render_diagnostics_section(diagnostics_report::build($courseid, $quizid !== 0 ? $quizid : null));
echo dashboard_renderer::render_hide_loading_notice();

echo dashboard_renderer::render_pdf_form(
    new moodle_url('/local/quizanalytics/diagnosticsanalyticspdf.php'),
    $courseid,
    $quizid !== 0 ? $quizid : null,
    $anonymize,
    [
        'diagnostics' => 'diagnosticsheading',
    ]
);

echo $OUTPUT->footer();
