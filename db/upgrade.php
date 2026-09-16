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
 * @package local_quizanalytics
 * @copyright  2026 Ernest Ting <eting@caltech.edu>
 * @license    http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */

defined('MOODLE_INTERNAL') || die();

function local_quizanalytics_create_prepared_table(): void {
    global $DB;
    $dbman = $DB->get_manager();
    $table = new xmldb_table('local_quizanalytics_prepared');
    $table->add_field('id', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, XMLDB_SEQUENCE, null);
    $table->add_field('courseid', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, null);
    $table->add_field('quizid', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, null);
    $table->add_field('datatype', XMLDB_TYPE_CHAR, '20', null, XMLDB_NOTNULL, null, 'course');
    $table->add_field('fingerprint', XMLDB_TYPE_CHAR, '32', null, XMLDB_NOTNULL, null, '');
    $table->add_field('status', XMLDB_TYPE_CHAR, '20', null, XMLDB_NOTNULL, null, 'stale');
    $table->add_field('payload', XMLDB_TYPE_TEXT, null, null, XMLDB_NOTNULL, null, null);
    $table->add_field('lastsuccess', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, 0);
    $table->add_field('timemodified', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, 0);
    $table->add_field('lasterror', XMLDB_TYPE_TEXT, null, null, false, null, null);
    $table->add_key('primary', XMLDB_KEY_PRIMARY, ['id']);
    $table->add_key('coursequiztype', XMLDB_KEY_UNIQUE, ['courseid', 'quizid', 'datatype']);
    $table->add_index('course', XMLDB_INDEX_NOTUNIQUE, ['courseid']);
    if (!$dbman->table_exists($table)) {
        $dbman->create_table($table);
    }
}

/**
 * @param int $oldversion
 * @return bool
 */
function xmldb_local_quizanalytics_upgrade($oldversion) {
    if ($oldversion < 2026082002) {
        // Resource-based auto-detection (classes/task/resource_detector.php)
        // is new as of this version — a fresh install picks it up via
        // db/install.php, but a site upgrading from before this version
        // existed never had that run. Only set parallelworkers here if
        // detection has genuinely never run on this site before (the
        // resourcedetectionrun marker, also set by install.php): an admin
        // upgrading from an earlier version may well have already tuned
        // parallelworkers by hand against their own real host, and an
        // upgrade silently overwriting a deliberate manual setting would
        // be a real regression, not a helpful default.
        if (!get_config('local_quizanalytics', 'resourcedetectionrun')) {
            require_once(__DIR__ . '/../classes/task/resource_detector.php');

            $workermemorymb = (int) (get_config('local_quizanalytics', 'parallelworkermemory') ?: 2048);
            $recommendation = \local_quizanalytics\task\resource_detector::recommend_parallel_workers($workermemorymb);

            set_config('parallelworkers', $recommendation['workers'], 'local_quizanalytics');
        }
        set_config('resourcedetectionrun', time(), 'local_quizanalytics');

        upgrade_plugin_savepoint(true, 2026082002, 'local', 'quizanalytics');
    }
    if ($oldversion < 2026091400) {
        require_once(__DIR__ . '/install.php');
        local_quizanalytics_create_prepared_table();
        upgrade_plugin_savepoint(true, 2026091400, 'local', 'quizanalytics');
    }
    if ($oldversion < 2026091401) {
        upgrade_plugin_savepoint(true, 2026091401, 'local', 'quizanalytics');
    }
    if ($oldversion < 2026091600) {
        global $DB;
        $dbman = $DB->get_manager();
        $table = new xmldb_table('local_quizanalytics_prepared');
        $datatype = new xmldb_field('datatype', XMLDB_TYPE_CHAR, '20', null, XMLDB_NOTNULL, null, 'course', 'quizid');
        if ($dbman->table_exists($table) && !$dbman->field_exists($table, $datatype)) {
            $dbman->add_field($table, $datatype);
            $DB->set_field('local_quizanalytics_prepared', 'datatype', 'course', []);
        }
        $oldkey = new xmldb_key('coursequiz', XMLDB_KEY_UNIQUE, ['courseid', 'quizid']);
        $newkey = new xmldb_key('coursequiztype', XMLDB_KEY_UNIQUE, ['courseid', 'quizid', 'datatype']);
        if ($dbman->table_exists($table)) {
            if ($dbman->find_key_name($table, $oldkey)) {
                $dbman->drop_key($table, $oldkey);
            }
            if (!$dbman->find_key_name($table, $newkey)) {
                $dbman->add_key($table, $newkey);
            }
        }
        upgrade_plugin_savepoint(true, 2026091600, 'local', 'quizanalytics');
    }

    return true;
}
