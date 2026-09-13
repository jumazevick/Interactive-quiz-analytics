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
 * Library functions for local_quizanalytics.
 *
 * HOW THIS GETS THE "Analytics" TAB ONTO THE SECONDARY NAV BAR
 * -------------------------------------------------------------
 * Confirmed against a real Moodle 4.5 core checkout while building the two
 * source plugins this merges (both used this exact mechanism):
 *
 *   1. public/lib/classes/navigation/settings_navigation.php calls
 *      get_plugins_with_function('extend_navigation_course', 'lib.php') for
 *      every installed plugin, so a lib.php function named exactly
 *      local_quizanalytics_extend_navigation_course($navigation,
 *      $course, $context) gets called automatically with $navigation set to
 *      the course's "courseadmin" node.
 *
 *   2. public/lib/classes/navigation/views/secondary.php::load_course_navigation()
 *      then promotes any child key not in core's "expected" list onto the
 *      course's secondary nav bar (or its "More" overflow) automatically.
 *
 * ONE nav entry, not several — this merge's whole point is that a teacher
 * sees a single "Analytics" tab, landing on the Quiz Analytics section
 * (index.php), with the "Section:" selector at the top of every page one
 * click away from every other section
 * (classes/section_selector.php::SECTION_PAGES).
 *
 * @package local_quizanalytics
 * @copyright  2026 Ernest Ting <eting@caltech.edu>
 * @license    http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */

defined('MOODLE_INTERNAL') || die();

/**
 * Adds a single "Analytics" link to a course's own administration
 * navigation, landing on the Quiz Analytics section (index.php) — the
 * "Section:" selector on that page reaches every other section from there.
 *
 * @param navigation_node $navigation the course admin node
 * @param stdClass $course
 * @param context_course $context
 */
function local_quizanalytics_extend_navigation_course($navigation, $course, $context) {
    if (!has_capability('local/quizanalytics:view', $context)) {
        return;
    }

    $url = new moodle_url('/local/quizanalytics/index.php', ['id' => $course->id]);
    $navigation->add(
        get_string('pluginname', 'local_quizanalytics'),
        $url,
        navigation_node::TYPE_SETTING,
        null,
        'quizanalyticscourse',
        new pix_icon('i/report', '')
    );
}

/**
 * Adds an "Analytics" link to a STACK quiz's own settings/administration
 * menu, jumping straight to the Question Analytics page pre-scoped to that
 * quiz (questionanalytics.php?id=<courseid>&quizid=<quizid>) — ported from
 * the original local_quizanalytics, which had this; the other source
 * plugin, local_stackanalytics, never did (Model 2/Diagnostics are
 * course-wide with an in-page quiz filter, not naturally reached from a
 * single quiz's own menu).
 *
 * Called by core for every settings-navigation build, at every context
 * level (lib/navigationlib.php's load_local_plugin_settings() calls every
 * local_*_extend_settings_navigation() unconditionally) — so this returns
 * immediately for anything that isn't a quiz's own page.
 *
 * The link is added as a child of the 'modulesettings' node specifically
 * (found via $settingsnav->find(), the same lookup mod_quiz's own
 * secondary-nav view uses to decide what belongs in the quiz page's "More"
 * dropdown), not appended to $settingsnav directly — a plain
 * $settingsnav->add() would attach it as a top-level sibling of
 * 'modulesettings' instead, which mod_quiz's dropdown-building code never
 * looks at.
 *
 * @param \settings_navigation $settingsnav
 * @param \context $context the current page's context
 */
function local_quizanalytics_extend_settings_navigation($settingsnav, $context) {
    if ($context->contextlevel != CONTEXT_MODULE) {
        return;
    }

    $cm = get_coursemodule_from_id('quiz', $context->instanceid, 0, false, IGNORE_MISSING);
    if (!$cm) {
        return; // Not a quiz module.
    }

    $coursecontext = context_course::instance($cm->course);
    if (!has_capability('local/quizanalytics:view', $coursecontext)) {
        return;
    }

    $url = new moodle_url('/local/quizanalytics/questionanalytics.php', [
        'id'     => $cm->course,
        'quizid' => $cm->instance,
    ]);

    $modulenode = $settingsnav->find('modulesettings', navigation_node::TYPE_SETTING);
    $parent = $modulenode ?: $settingsnav;
    $parent->add(
        get_string('pluginname', 'local_quizanalytics'),
        $url,
        navigation_node::TYPE_SETTING,
        null,
        'quizanalyticsquiz',
        new pix_icon('i/report', '')
    );
}
