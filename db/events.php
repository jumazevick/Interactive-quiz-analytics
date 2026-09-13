<?php
// This file is part of Moodle - http://moodle.org/

defined('MOODLE_INTERNAL') || die();

$observers = [
    [
        'eventname' => '\\mod_quiz\\event\\attempt_submitted',
        'callback' => 'local_quizanalytics_event_observer::attempt_submitted',
        'includefile' => '/local/quizanalytics/classes/event_observer.php',
        'internal' => true,
        'priority' => 100,
    ],
];
