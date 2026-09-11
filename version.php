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
 * Version details for the STACK q-type Analytics plugin.
 *
 * @package    local_quizanalytics
 * @copyright  2026 Ernest Ting <eting@caltech.edu>
 * @license    http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */

defined('MOODLE_INTERNAL') || die();

$plugin->component = 'local_quizanalytics'; // Must match the folder this unzips into:
                                                  // local/quizanalytics on a Moodle install. This
                                                  // is the original, standalone local_quizanalytics
                                                  // plugin's own frankenstyle name — this merged
                                                  // plugin replaces that listing rather than the
                                                  // other source plugin's (local_stackanalytics),
                                                  // uploaded as a new version of it. This repo's own
                                                  // root IS that folder's contents (no
                                                  // local/quizanalytics/ nesting in the repo itself),
                                                  // so a plain "Download ZIP" of the repo has
                                                  // version.php sitting directly inside the single
                                                  // top-level wrapper folder, which is what the Moodle
                                                  // plugin uploader requires to detect the frankenstyle
                                                  // component, plugin type, and required core version.
$plugin->version   = 2026091100;                 // YYYYMMDDXX — bump this every time you push an update.
$plugin->requires  = 2022041900;                 // Moodle 4.0.0 — matches both source plugins' own
                                                  // requirement (the analyser/target/indicator base
                                                  // classes the original local_stackanalytics used are
                                                  // present since Moodle 3.4 and stable through 4.x/5.x).
$plugin->maturity  = MATURITY_STABLE;            // Bumped from MATURITY_ALPHA: this plugin has now been
                                                  // through extensive real-course testing (see CHANGELOG.md
                                                  // 2.4.0-2.4.1's stress testing) and several rounds of real-
                                                  // world bug fixes. Also affects whether Moodle sites see
                                                  // this as an available update at all — a site's own Update
                                                  // notifications maturity filter (Site administration >
                                                  // Server > Update notifications) commonly excludes alpha
                                                  // releases by default.
$plugin->release   = '3.0.4';

// This plugin is the merger of two previously-standalone plugins:
// local_quizanalytics (course-wide/question/solution-process STACK response
// analytics, PDF export) and local_stackanalytics (the Analytics API's
// Model 1/Model 2 + the non-ML Diagnostics Dashboard). Both are STACK
// (qtype_stack) specific and depend on mod_quiz as the vehicle for STACK
// question attempts. ANY_VERSION for qtype_stack since nothing here calls
// an API added in a particular STACK release — both source plugins read
// finished attempts through mod_quiz's own question engine, and qtype_stack
// is only touched for question text rendering (castext2_qa_processor) and
// PRT tree definitions (the long-standing, stable mdl_qtype_stack_prts /
// mdl_qtype_stack_prt_nodes tables), never a version-specific PHP API.
$plugin->dependencies = [
    'mod_quiz' => 2022041900,
    'qtype_stack' => ANY_VERSION,
];
