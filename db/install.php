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
 * Runs once, on a fresh install only (see db/upgrade.php for existing
 * sites upgrading into this feature for the first time).
 *
 * @package local_quizanalytics
 * @copyright  2026 Ernest Ting <eting@caltech.edu>
 * @license    http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */

defined('MOODLE_INTERNAL') || die();

/**
 * Sets local_quizanalytics/parallelworkers from this server's own detected
 * CPU/RAM, so a fresh install gets a sane, host-appropriate value in Site
 * administration immediately — not just correct-but-invisible adaptive
 * runtime behaviour an admin would only discover by reading code. See
 * classes/task/resource_detector.php's own docblock for the detection
 * logic and why it's sized the way it is; see local_quizanalytics_upgrade()
 * for the same logic applied to a site upgrading from before this existed.
 */
function xmldb_local_quizanalytics_install() {
    local_quizanalytics_create_prepared_table();
    require_once(__DIR__ . '/../classes/task/resource_detector.php');

    $workermemorymb = (int) (get_config('local_quizanalytics', 'parallelworkermemory') ?: 2048);
    $recommendation = \local_quizanalytics\task\resource_detector::recommend_parallel_workers($workermemorymb);

    set_config('parallelworkers', $recommendation['workers'], 'local_quizanalytics');
    set_config('resourcedetectionrun', time(), 'local_quizanalytics');
}

function local_quizanalytics_create_prepared_table(): void {
    global $DB;
    $dbman = $DB->get_manager();
    $table = new xmldb_table('local_quizanalytics_prepared');
    $table->add_field('id', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, XMLDB_SEQUENCE, null);
    $table->add_field('courseid', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, null);
    $table->add_field('quizid', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, null);
    $table->add_field('fingerprint', XMLDB_TYPE_CHAR, '32', null, XMLDB_NOTNULL, null, '');
    $table->add_field('status', XMLDB_TYPE_CHAR, '20', null, XMLDB_NOTNULL, null, 'stale');
    $table->add_field('payload', XMLDB_TYPE_TEXT, null, null, XMLDB_NOTNULL, null, null);
    $table->add_field('lastsuccess', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, 0);
    $table->add_field('timemodified', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, 0);
    $table->add_field('lasterror', XMLDB_TYPE_TEXT, null, null, false, null, null);
    $table->add_key('primary', XMLDB_KEY_PRIMARY, ['id']);
    $table->add_key('coursequiz', XMLDB_KEY_UNIQUE, ['courseid', 'quizid']);
    $table->add_index('course', XMLDB_INDEX_NOTUNIQUE, ['courseid']);
    if (!$dbman->table_exists($table)) {
        $dbman->create_table($table);
    }
}
