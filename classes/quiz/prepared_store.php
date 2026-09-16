<?php
// This file is part of Moodle - http://moodle.org/

defined('MOODLE_INTERNAL') || die();

/** Durable store for compact, reusable per-quiz analytics preparation data. */
class local_quizanalytics_prepared_store {
    private const TABLE = 'local_quizanalytics_prepared';

    public static function get_course(int $courseid, ?string $datatype = null): array {
        global $DB;
        $select = ['courseid' => $courseid];
        if ($datatype !== null) {
            $select['datatype'] = $datatype;
        }
        $records = $DB->get_records(self::TABLE, $select, 'quizid ASC');
        $byquiz = [];
        foreach ($records as $record) {
            $byquiz[(int) $record->quizid . ':' . ($record->datatype ?? 'course')] = $record;
        }
        return $byquiz;
    }

    public static function get(int $courseid, int $quizid, string $datatype = 'course'): ?stdClass {
        global $DB;
        $row = $DB->get_record(self::TABLE, [
            'courseid' => $courseid, 'quizid' => $quizid, 'datatype' => $datatype,
        ]);
        return $row ?: null;
    }

    public static function mark_running(int $courseid, int $quizid, string $datatype = 'course'): void {
        global $DB;
        $now = time();
        $row = self::get($courseid, $quizid, $datatype);
        if ($row) {
            $DB->update_record(self::TABLE, (object) [
                'id' => $row->id, 'status' => 'running', 'timemodified' => $now, 'lasterror' => null,
            ]);
            return;
        }
        $DB->insert_record(self::TABLE, (object) [
            'courseid' => $courseid, 'quizid' => $quizid, 'datatype' => $datatype,
            'fingerprint' => '', 'status' => 'running',
            'payload' => '', 'lastsuccess' => 0, 'timemodified' => $now, 'lasterror' => null,
        ]);
    }

    public static function save_success(
        int $courseid, int $quizid, string $fingerprint, array $payload, string $datatype = 'course'
    ): void {
        global $DB;
        $row = self::get($courseid, $quizid, $datatype);
        $data = (object) [
            'courseid' => $courseid, 'quizid' => $quizid, 'datatype' => $datatype,
            'fingerprint' => $fingerprint,
            'status' => 'complete', 'payload' => json_encode($payload, JSON_THROW_ON_ERROR),
            'lastsuccess' => time(), 'timemodified' => time(), 'lasterror' => null,
        ];
        if ($row) {
            $data->id = $row->id;
            $DB->update_record(self::TABLE, $data);
        } else {
            $DB->insert_record(self::TABLE, $data);
        }
    }

    public static function mark_failed(int $courseid, int $quizid, string $message, string $datatype = 'course'): void {
        global $DB;
        $row = self::get($courseid, $quizid, $datatype);
        if ($row) {
            // Keep a previously successful payload available as stale data;
            // the next page load can still render it while this quiz retries.
            $status = ($row->status === 'complete' && $row->payload !== '')
                ? 'complete' : 'failed';
            $DB->update_record(self::TABLE, (object) [
                'id' => $row->id, 'status' => $status, 'timemodified' => time(),
                'lasterror' => substr($message, 0, 65535),
            ]);
        }
    }

    public static function mark_stale(int $courseid, int $quizid, string $datatype = 'question'): void {
        global $DB;
        $row = self::get($courseid, $quizid, $datatype);
        if ($row) {
            $DB->update_record(self::TABLE, (object) [
                'id' => $row->id, 'status' => 'stale', 'timemodified' => time(),
            ]);
        } else {
            $DB->insert_record(self::TABLE, (object) [
                'courseid' => $courseid, 'quizid' => $quizid, 'datatype' => $datatype,
                'fingerprint' => '', 'status' => 'stale', 'payload' => '',
                'lastsuccess' => 0, 'timemodified' => time(), 'lasterror' => null,
            ]);
        }
    }

    public static function is_fresh(?stdClass $row, string $fingerprint): bool {
        return $row !== null && $row->status === 'complete'
            && $row->fingerprint === $fingerprint && $row->payload !== '';
    }

    public static function decode(?stdClass $row): ?array {
        if (!$row || $row->payload === '') {
            return null;
        }
        $payload = json_decode($row->payload, true);
        return is_array($payload) ? $payload : null;
    }
}
