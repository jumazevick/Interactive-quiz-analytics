<?php
define('AJAX_SCRIPT', true);
require_once(__DIR__ . '/../../config.php');
require_once($CFG->dirroot . '/local/quizanalytics/classes/quiz/data_fetcher.php');
require_once($CFG->dirroot . '/local/quizanalytics/classes/quiz/cache_helper.php');

$courseid = required_param('id', PARAM_INT);
$quizid = required_param('quizid', PARAM_INT);
$question = required_param('question', PARAM_RAW_TRIMMED);
$variant = required_param('variant', PARAM_INT);

$course = $DB->get_record('course', ['id' => $courseid], '*', MUST_EXIST);
require_login($course);
$context = context_course::instance($courseid);
require_capability('local/quizanalytics:view', $context);

$quiz = $DB->get_record('quiz', ['id' => $quizid, 'course' => $courseid], '*', MUST_EXIST);
$colorblind = (bool) optional_param('colorblind', 0, PARAM_INT);
$anonymize = (bool) optional_param('anonymize', 0, PARAM_INT);
$stats = local_quizanalytics_quiz_cache_helper::stats_for_quiz($quiz);
$snapshot = local_quizanalytics_quiz_data_fetcher::get_quiz_snapshot($quiz, $course);
$key = local_quizanalytics_quiz_cache_helper::build_key(
    $quizid, $stats->fingerprint, md5(json_encode($snapshot)), $colorblind, $anonymize
);
$cache = cache::make('local_quizanalytics', 'questionanalysis');
$result = $cache->get($key);

$version = null;
if (is_array($result) && isset($result['questions'][$question]['versions'][$variant])) {
    $version = $result['questions'][$question]['versions'][$variant];
}
if (!is_array($version)) {
    http_response_code(404);
    echo json_encode(['error' => 'Question variant is not available.']);
    exit;
}

header('Content-Type: application/json; charset=utf-8');
echo json_encode($version, JSON_HEX_TAG | JSON_HEX_AMP | JSON_HEX_APOS | JSON_HEX_QUOT);
