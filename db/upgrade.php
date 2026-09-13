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

/**
 * @param int $oldversion
 * @return bool
 */
function xmldb_local_quizanalytics_upgrade($oldversion) {
    if ($oldversion < 2026091400) {
        require_once(__DIR__ . '/install.php');
        local_quizanalytics_create_prepared_table();
        upgrade_plugin_savepoint(true, 2026091400, 'local', 'quizanalytics');
    }
    if ($oldversion < 2026091401) {
        upgrade_plugin_savepoint(true, 2026091401, 'local', 'quizanalytics');
    }
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

    return true;
}
