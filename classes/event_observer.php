<?php
// This file is part of Moodle - http://moodle.org/

defined('MOODLE_INTERNAL') || die();

/** Marks affected reusable quiz preparation data stale after quiz activity. */
class local_quizanalytics_event_observer {
    public static function attempt_submitted(\mod_quiz\event\attempt_submitted $event): void {
        global $DB;

        $attempt = $event->get_record_snapshot('quiz_attempts', $event->objectid);
        if (!$attempt || empty($attempt->quiz)) {
            return;
        }
        $quiz = $DB->get_record('quiz', ['id' => $attempt->quiz], 'id,course');
        if (!$quiz) {
            return;
        }
        require_once(__DIR__ . '/quiz/prepared_store.php');
        require_once(__DIR__ . '/task/warm_single_view_adhoc_task.php');
        local_quizanalytics_prepared_store::mark_stale((int) $quiz->course, (int) $quiz->id);
        \local_quizanalytics\task\warm_single_view_adhoc_task::dispatch_for_course(
            (int) $quiz->course,
            \local_quizanalytics\quiz\analytics\course_analysis::DEFAULT_GRADE_TYPE,
            false,
            false
        );
    }
}
