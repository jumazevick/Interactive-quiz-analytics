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
 * The top-level "Section:" selector shared by every page-level entry point
 * in this plugin — one nav entry, switching between each section's whole
 * page tree via a plain GET-reload, same convention every other selector
 * in this plugin already uses, no JS required.
 *
 * Started as a 2-way switch (Quiz Analytics / Model & Diagnostics
 * Analytics) when this plugin first merged local_quizanalytics and
 * local_stackanalytics; Quiz Analytics's own per-quiz drill-down later
 * split into its own Question Analytics section, and Model & Diagnostics
 * Analytics split the same way into Model Analytics and Diagnostics
 * Analytics — four independently-reachable sections in total now, though
 * Diagnostics Analytics is currently omitted from the rendered switcher
 * (see render()) while its page/code stays fully intact and reachable
 * directly by URL.
 *
 * Deliberately global-namespace and outside classes/quiz/ or classes/stack/
 * — this belongs to neither product specifically, only to the merged
 * plugin's own top-level page structure.
 *
 * @package local_quizanalytics
 * @copyright  2026 Ernest Ting <eting@caltech.edu>
 * @license    http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */

defined('MOODLE_INTERNAL') || die();

/**
 * Renders the "Section:" selector shown at the top of every section's own page.
 */
class local_quizanalytics_section_selector {
    /** @var array<string, string> section key => page-level entry point, relative to this plugin's own root. */
    const SECTION_PAGES = [
        'quiz' => 'index.php',
        'question' => 'questionanalytics.php',
        'models' => 'modelanalytics.php',
        'diagnostics' => 'diagnosticsanalytics.php',
    ];

    /**
     * Renders the "Section:" switcher, bolding whichever section is currently active.
     *
     * @param int $courseid
     * @param string $current one of self::SECTION_PAGES's keys
     * @return string
     */
    public static function render(int $courseid, string $current): string {
        $options = [
            'quiz' => get_string('sectionquiz', 'local_quizanalytics'),
            'question' => get_string('sectionquestion', 'local_quizanalytics'),
            'models' => get_string('sectionmodels', 'local_quizanalytics'),
            'diagnostics' => get_string('sectiondiagnostics', 'local_quizanalytics'),
        ];

        // Model Analytics and Diagnostics Analytics are intentionally not
        // offered here — pending redesigns, same as Question Analytics's own
        // Solution Process view (see questionanalytics.php). Their pages and
        // code remain reachable by direct URL; this only removes them from
        // this switcher's visible links.
        unset($options['models']);
        unset($options['diagnostics']);

        // Rendered as plain links rather than a form+select — there's no
        // per-section state to carry across the switch (unlike the quiz/view
        // selectors nested inside each section, which do need a GET form to
        // submit their own params), so a link is the simpler, more honestly-GET
        // choice here.
        $links = [];
        foreach ($options as $section => $label) {
            $url = new \moodle_url('/local/quizanalytics/' . self::SECTION_PAGES[$section], ['id' => $courseid]);
            if ($section === $current) {
                $links[] = \html_writer::tag('strong', $label);
            } else {
                $links[] = \html_writer::link($url, $label);
            }
        }

        return \html_writer::tag(
            'p',
            \html_writer::tag('span', get_string('sectionselectorlabel', 'local_quizanalytics'), ['class' => 'mr-2'])
                . implode(' · ', $links),
            ['class' => 'mb-3']
        );
    }
}
