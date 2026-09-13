<?php
// This file is part of Moodle - http://moodle.org/

define('AJAX_SCRIPT', true);
require_once(__DIR__ . '/../../config.php');
require_once($CFG->dirroot . '/local/quizanalytics/classes/task/warm_single_view_adhoc_task.php');

$courseid = required_param('id', PARAM_INT);
$fingerprint = required_param('fingerprint', PARAM_ALPHANUM);
$gradetype = optional_param('gradetype', 'Average Grade', PARAM_TEXT);
$colorblind = optional_param('colorblind', 0, PARAM_BOOL);
$anonymize = optional_param('anonymize', 0, PARAM_BOOL);
$quizidsparam = optional_param('quizids', '', PARAM_RAW);
$quizids = $quizidsparam === '' ? [] : array_values(array_unique(array_filter(
    array_map('intval', explode(',', $quizidsparam))
)));
$selectionkey = local_quizanalytics_quiz_cache_helper::selection_key($quizids);

$course = $DB->get_record('course', ['id' => $courseid], '*', MUST_EXIST);
require_login($course);
$context = context_course::instance($courseid);
require_capability('local/quizanalytics:view', $context);

$progress = \local_quizanalytics\task\warm_single_view_adhoc_task::get_progress(
    $courseid, $fingerprint, $gradetype, $colorblind, $anonymize, $selectionkey
);

if ($progress === false) {
    $progress = [
        'status' => 'queued',
        'stage' => 'queued',
        'completed' => 0,
        'total' => 0,
        'percent' => 0,
        'message' => get_string('progressqueued', 'local_quizanalytics'),
        'updated' => time(),
    ];
}

header('Content-Type: application/json; charset=utf-8');
echo json_encode($progress, JSON_THROW_ON_ERROR);
