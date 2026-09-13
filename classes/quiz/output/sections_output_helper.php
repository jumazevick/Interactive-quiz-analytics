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
 * Shared rendering scaffolding for the {summary, sections} JSON contract,
 * used across every view local_quizanalytics renders: course-wide
 * comparison, per-quiz Question Analytics, and per-quiz Solution Process
 * Visualization.
 *
 * Deliberately NOT using $PAGE->requires->js()/->css() for the vendored
 * libraries (Plotly, KaTeX): that path routes every file through
 * /lib/javascript.php, which re-minifies it with core_minify
 * (MatthiasMullie\Minify\JS) even though these are already minified — this
 * was confirmed to corrupt the ~3.6MB Plotly bundle (a real syntax error in
 * the re-minified output, verified with `node --check`), producing
 * "Uncaught SyntaxError" in the browser. Echoing raw <script src>/<link>
 * tags serves the plugin files directly, unmodified. KaTeX's CSS is
 * similarly kept off Moodle's CSS pipeline since it references its fonts
 * via relative url(...) paths that a combining/rewriting step could break
 * the same way.
 *
 * Namespaced, autoloaded class — no MOODLE_INTERNAL guard needed (that
 * convention is for classic directly-included files; this is loaded via
 * Moodle's PSR-4 autoloader from classes/output/sections_output_helper.php).
 *
 * Deliberately NOT named "*_renderer": these are plain static string-building
 * helpers with no $this->page/$this->output (unlike a real Moodle renderer
 * extending renderer_base) - naming it that way would trip moodle-cs's
 * ForbiddenGlobalUseSniff, which assumes any "*_renderer" class is a genuine
 * renderer and bans "global $PAGE" there in favor of $this->page.
 *
 * @package local_quizanalytics
 * @copyright  2026 Ernest Ting <eting@caltech.edu>
 * @license    http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */

namespace local_quizanalytics\quiz\output;

/**
 * Static HTML-building helpers for the page scaffolding shared across every view.
 */
class sections_output_helper {
    /**
     * DOM id of the "this may take a while" notice — shared with
     * dashboard_renderer's own identical flush_computing_notice(), so
     * render_hide_loading_notice() below can find and remove either one by
     * the same id regardless of which section rendered it.
     */
    const LOADING_NOTICE_ID = 'lqa-loading-notice';

    /**
     * Echoes the empty container divs a payload gets rendered into.
     * Callers must use a unique $prefix per page when more than one
     * independent result is rendered on the same page.
     *
     * @param string $prefix DOM id prefix, e.g. "qa", "spv", "qw".
     * @return string
     */
    public static function render_containers(string $prefix): string {
        $html  = \html_writer::div('', '', ['id' => $prefix . '-summary']);
        $html .= \html_writer::div('', '', ['id' => $prefix . '-sections']);
        return $html;
    }

    /**
     * Echoes a "this may take a while" notice and pushes it to the browser
     * immediately, for a caller about to run a cold-cache compute that can
     * take anywhere from seconds to several minutes on a large course.
     * PHP doesn't send anything to the browser until either its output
     * buffer fills or the script ends, so without this a visitor just sees
     * a blank/loading tab for the whole wait with no indication anything
     * is happening — this at minimum gets the page shell and this notice
     * out before the expensive part starts. Purely cosmetic: doesn't change
     * what gets computed, cached, or how long it takes. Shares its wording
     * (largecoursenotice) with dashboard_renderer's own identical method,
     * used by Model/Diagnostics Analytics — one consistent "this may take a
     * while" message across every section instead of a different one here.
     */
    public static function flush_computing_notice(): void {
        echo \html_writer::div(
            get_string('largecoursenotice', 'local_quizanalytics'),
            'alert alert-info',
            ['id' => self::LOADING_NOTICE_ID]
        );
        if (ob_get_level() > 0) {
            @ob_flush();
        }
        flush();
    }

    /**
     * Default time budget (seconds) for a cold on-demand fetch — used only
     * when the admin hasn't set local_quizanalytics/backgroundtimebudget.
     *
     * A fixed *attempt-count* threshold was tried first and replaced: measured
     * directly against real data, per-attempt cost varies roughly 10x between
     * a simple randomised question and a genuinely complex one (a synthetic
     * simple-question quiz measured ~10x cheaper per attempt than this
     * plugin's own real, largest test quiz), and obviously also depends on
     * the host's own CPU/DB speed — a count picked against one specific
     * machine's measured rate has no reason to hold on a slower shared host
     * or a faster dedicated one. should_defer_to_background() now takes a
     * real, sampled time estimate for *this* quiz on *this* host instead
     * (see data_fetcher.php's own estimate_seconds_per_attempt()), so this
     * constant is a genuine time budget, not a proxy for one. 20s leaves
     * real margin under a 100s reverse-proxy timeout even after adding the
     * analyze() stage on top (measured well under it on every quiz tested
     * this session) and after accounting for the sampling estimate's own
     * built-in conservative bias (~1.4x over a 100-attempt sample, measured
     * directly — see estimate_seconds_per_attempt()'s own comment).
     */
    const DEFAULT_BACKGROUND_TIME_BUDGET_SECONDS = 20;

    /**
     * Whether a cold compute estimated to take $estimatedseconds is long
     * enough that index.php/questionanalytics.php should dispatch it to a
     * background task (see warm_single_view_adhoc_task) instead of
     * computing it inline on this request — the synchronous on-demand path
     * has no forking (pcntl_fork only works in CLI/cron context) and no
     * upper bound on how large a single quiz or course can be, so unlike
     * the scheduled task's own cold-compute cost, this one really can
     * exceed a reverse proxy's timeout before ignore_user_abort(true) even
     * gets a chance to help.
     *
     * @param float $estimatedseconds from data_fetcher.php's
     *        estimate_seconds_per_attempt(), multiplied by the view's own
     *        real attempt count — 0.0 (never defer) if estimation wasn't
     *        possible (e.g. genuinely zero attempts to sample from).
     * @return bool
     */
    public static function should_defer_to_background(float $estimatedseconds): bool {
        $budget = (int) (get_config('local_quizanalytics', 'backgroundtimebudget') ?: self::DEFAULT_BACKGROUND_TIME_BUDGET_SECONDS);
        return $budget > 0 && $estimatedseconds > $budget;
    }

    /**
     * Echoes the notice shown instead of blocking when
     * should_defer_to_background() is true — the compute has been handed
     * to a background task rather than started on this request, so unlike
     * flush_computing_notice() (which precedes an inline compute that
     * really will finish on this same request) there is nothing to wait
     * for here; the visitor needs to come back to this page later.
     *
     * Distinguishes a task still working normally from one that's been
     * queued suspiciously long (see warm_single_view_adhoc_task::
     * STALE_SECONDS/get_queued_age_seconds()) — the same calm "come back
     * in a few minutes" notice shown indefinitely for a task that's
     * actually stuck (no cron running at all, a crashed worker, one stuck
     * in a fail-delay retry loop) would leave an admin with nothing to go
     * on beyond "the page never finishes loading."
     *
     * @param int|null $queuedagesecs from
     *        warm_single_view_adhoc_task::get_queued_age_seconds() for the
     *        exact view being requested — null if that couldn't be
     *        determined (falls back to the ordinary notice rather than
     *        risking a false "stuck" warning).
     */
    public static function render_generating_in_background_notice(?int $queuedagesecs = null): void {
        if ($queuedagesecs !== null && $queuedagesecs > \local_quizanalytics\task\warm_single_view_adhoc_task::STALE_SECONDS) {
            // No auto-refresh here: the task is genuinely stuck (or cron
            // isn't running at all), so repeatedly reloading only repeats
            // the same "still not done" result — the admin needs to see
            // and act on this troubleshooting message, not have it vanish
            // into a reload every 20 seconds.
            echo \html_writer::div(
                get_string('generatingstale', 'local_quizanalytics', format_time($queuedagesecs)),
                'alert alert-warning'
            );
            return;
        }
        // Auto-refreshing via a plain <meta> tag (not JS) matches this
        // plugin's own established "plain GET-reload, no JS needed"
        // convention elsewhere (native <details>, plain forms) — and
        // previously this page just sat here forever until a visitor
        // manually reloaded it themselves, which is exactly what a visitor
        // reported having to do "sometimes" and not reliably remembering
        // to. A <meta http-equiv="refresh"> tag works correctly even
        // placed in the body (long-standing, reliable browser behaviour),
        // so no change to header output earlier in the response is needed
        // here. 20s matches the wording in the notice text itself.
        echo \html_writer::empty_tag('meta', ['http-equiv' => 'refresh', 'content' => '20']);
        echo \html_writer::start_div('alert alert-info d-flex align-items-center');
        echo \html_writer::tag(
            'span',
            '',
            [
                'class' => 'lqa-spinner mr-2',
                'style' => 'display:inline-block;width:1rem;height:1rem;border:2px solid currentColor;'
                    . 'border-right-color:transparent;border-radius:50%;animation:lqa-spin 0.75s linear infinite;',
            ]
        );
        echo \html_writer::tag('style', '@keyframes lqa-spin { to { transform: rotate(360deg); } }');
        echo \html_writer::tag('span', get_string('generatinginbackground', 'local_quizanalytics'));
        echo \html_writer::end_div();
    }

    /**
     * Echoes the vendored-library <script>/<link> tags, the JSON payload as
     * an inline script variable, and the shared sections-renderer.js — in
     * that guaranteed document order, so plain synchronous script execution
     * order does the right thing with no event-listener ordering games.
     *
     * @param string $prefix Must match the prefix passed to render_containers().
     * @param array $result The API response to render (summary/sections or
     *        the legacy summary/figures shape).
     * @param bool $includevendor Set false to skip re-emitting the vendored
     *        library tags when multiple payloads are rendered on one page
     *        (only the first render_vendor_and_payload() call on a page
     *        needs $includevendor = true).
     * @return string
     */
    public static function render_vendor_and_payload(string $prefix, array $result, bool $includevendor = true): string {
        $html = '';

        if ($includevendor) {
            $plotlyurl = new \moodle_url('/local/quizanalytics/js/vendor/plotly.min.js');
            $katexcssurl = new \moodle_url('/local/quizanalytics/js/vendor/katex/katex.min.css');
            $katexjsurl = new \moodle_url('/local/quizanalytics/js/vendor/katex/katex.min.js');
            $katexautorenderurl = new \moodle_url('/local/quizanalytics/js/vendor/katex/contrib/auto-render.min.js');
            $sectionsrendererurl = new \moodle_url('/local/quizanalytics/js/vendor-shared/sections-renderer.js');

            $html .= \html_writer::empty_tag('link', ['rel' => 'stylesheet', 'href' => $katexcssurl->out(false)]);
            $html .= \html_writer::tag('script', '', ['src' => $plotlyurl->out(false)]);
            $html .= \html_writer::tag('script', '', ['src' => $katexjsurl->out(false)]);
            $html .= \html_writer::tag('script', '', ['src' => $katexautorenderurl->out(false)]);
            $html .= \html_writer::tag('script', '', ['src' => $sectionsrendererurl->out(false)]);
        }

        $safejson = json_encode($result, JSON_HEX_TAG | JSON_HEX_AMP | JSON_HEX_APOS | JSON_HEX_QUOT);
        $html .= \html_writer::tag(
            'script',
            "QuizAnalyticsRenderer.render(" . json_encode($prefix) . ", {$safejson});"
        );

        return $html;
    }

    /**
     * A tiny inline script removing the "may take a while" notice
     * (LOADING_NOTICE_ID above) from the page — call this right after
     * echoing the real results, never before. A <script> tag runs
     * synchronously the instant the browser's HTML parser reaches it, so by
     * the time this one runs, everything printed before it — the notice,
     * and whatever real results this call follows — is already sitting in
     * the DOM. No event listener or load-state tracking needed for that:
     * the notice exists to reassure a visitor mid-wait, not to linger once
     * there's something to look at instead. Safe to call even when no
     * notice was actually shown (a cache hit that skipped
     * flush_computing_notice() entirely) — getElementById() then finds
     * nothing and the optional chaining below is a no-op.
     *
     * @return string
     */
    public static function render_hide_loading_notice(): string {
        return \html_writer::tag(
            'script',
            'document.getElementById(' . json_encode(self::LOADING_NOTICE_ID) . ')?.remove();'
        );
    }

    /**
     * Reads (and, if a new value was submitted, persists) the colorblind
     * display preference. Shared across every view so a teacher's choice is
     * consistent everywhere rather than set per-page.
     *
     * @return bool
     */
    public static function resolve_colorblind_mode(): bool {
        $param = optional_param('colorblind', null, PARAM_INT);
        if ($param !== null) {
            \set_user_preference('local_quizanalytics_colorblind', (bool) $param);
            return (bool) $param;
        }
        // Note: Moodle's getter is the plural get_user_preferences(), despite
        // the setter being the singular set_user_preference() above — a real
        // asymmetry in core's own API, not a typo.
        return (bool) \get_user_preferences('local_quizanalytics_colorblind', false);
    }

    /**
     * Reads the anonymize flag from the current request.
     *
     * Anonymization is deliberately not persisted: the default must always
     * be real student labels, and anonymization is enabled only when the
     * current form submission explicitly sends anonymize=1.
     *
     * @return bool
     */
    public static function resolve_anonymize_mode(): bool {
        return (bool) optional_param('anonymize', 0, PARAM_INT);
    }

    /**
     * A small GET-reload checkbox form for the colorblind mode and anonymize
     * student data toggles, preserving every other current query parameter
     * (quiz id, question/part selectors, etc.) so every distinct view stays
     * a cacheable URL.
     *
     * @return string
     */
    public static function render_options_toggles(bool $colorblind, bool $anonymize): string {
        global $PAGE;

        $html = \html_writer::start_tag('form', [
            'method' => 'get',
            'action' => $PAGE->url->out_omit_querystring(),
            'class'  => 'mb-4',
        ]);
        foreach ($PAGE->url->params() as $name => $value) {
            if ($name === 'colorblind' || $name === 'anonymize') {
                continue;
            }
            $html .= \html_writer::empty_tag('input', ['type' => 'hidden', 'name' => $name, 'value' => $value]);
        }
        // An unchecked HTML checkbox submits nothing at all, which would
        // leave the param null and fall through to the stored (possibly
        // still "true") preference — this hidden 0, submitted first, is
        // overridden by the checkbox's own value only when it's actually
        // checked (PHP keeps the last of two same-named query params), so
        // unchecking the box and submitting genuinely clears it.
        $html .= \html_writer::empty_tag('input', ['type' => 'hidden', 'name' => 'colorblind', 'value' => '0']);
        $colorblindattrs = ['type' => 'checkbox', 'name' => 'colorblind', 'value' => '1', 'id' => 'qa-colorblind-toggle'];
        if ($colorblind) {
            $colorblindattrs['checked'] = 'checked';
        }
        $html .= \html_writer::empty_tag('input', $colorblindattrs);
        $html .= ' ' . \html_writer::label(\get_string('colorblindmode', 'local_quizanalytics'), 'qa-colorblind-toggle');

        $html .= \html_writer::empty_tag('input', ['type' => 'hidden', 'name' => 'anonymize', 'value' => '0']);
        $anonymizeattrs = [
            'type' => 'checkbox', 'name' => 'anonymize', 'value' => '1', 'id' => 'qa-anonymize-toggle', 'class' => 'ml-3',
        ];
        if ($anonymize) {
            $anonymizeattrs['checked'] = 'checked';
        }
        $html .= \html_writer::empty_tag('input', $anonymizeattrs);
        $html .= ' ' . \html_writer::label(\get_string('anonymizemode', 'local_quizanalytics'), 'qa-anonymize-toggle');

        $html .= ' ' . \html_writer::empty_tag('input', [
            'type' => 'submit', 'value' => \get_string('apply', 'moodle'), 'class' => 'btn btn-secondary btn-sm',
        ]);
        $html .= \html_writer::end_tag('form');
        return $html;
    }

    /**
     * The per-quiz "View:" selector between Question Analytics and Solution
     * Process Visualization — a plain GET-reload, like everything else here,
     * so only the selected view's data ever gets fetched/computed (picking a
     * view is a page reload with &view=..., not a client-side tab swap that
     * would need both already computed).
     *
     * @param string $current 'question' or 'solutionprocess'
     * @return string
     */
    public static function render_view_selector_form(string $current): string {
        global $PAGE;

        $options = [
            'question'        => \get_string('viewquestionanalytics', 'local_quizanalytics'),
            'solutionprocess' => \get_string('viewsolutionprocess', 'local_quizanalytics'),
        ];

        $html = \html_writer::start_tag('form', [
            'method' => 'get', 'action' => $PAGE->url->out_omit_querystring(), 'class' => 'mb-4',
        ]);
        foreach ($PAGE->url->params() as $name => $value) {
            if ($name === 'view') {
                continue;
            }
            $html .= \html_writer::empty_tag('input', ['type' => 'hidden', 'name' => $name, 'value' => $value]);
        }
        $html .= \html_writer::label(\get_string('viewselectorlabel', 'local_quizanalytics'), 'qa-view-select');
        $html .= ' ' . \html_writer::select($options, 'view', $current, false, ['id' => 'qa-view-select']);
        $html .= ' ' . \html_writer::empty_tag('input', [
            'type' => 'submit', 'value' => \get_string('gobutton', 'local_quizanalytics'), 'class' => 'btn btn-secondary',
        ]);
        $html .= \html_writer::end_tag('form');
        return $html;
    }

    /**
     * The Solution Process Visualization question/part/student selector — a
     * plain GET-reload form, matching the "view" selector above.
     *
     * @param \moodle_url $url
     * @param array $meta {questions: [{name, parts}], students: [{id, name}]}
     * @param string $question
     * @param int $partsforquestion
     * @param int $part
     * @param string $studentid
     * @return string
     */
    public static function render_solutionprocess_selector_form(
        \moodle_url $url,
        array $meta,
        string $question,
        int $partsforquestion,
        int $part,
        string $studentid
    ): string {
        $ownparams = ['spvquestion', 'spvpart', 'spvstudent'];

        $html = \html_writer::start_tag('form', [
            'method' => 'get', 'action' => $url->out_omit_querystring(), 'class' => 'mb-4',
        ]);
        foreach ($url->params() as $name => $value) {
            if (in_array($name, $ownparams, true)) {
                continue;
            }
            $html .= \html_writer::empty_tag('input', ['type' => 'hidden', 'name' => $name, 'value' => $value]);
        }

        $questionoptions = [];
        foreach ($meta['questions'] as $q) {
            $questionoptions[$q['name']] = $q['name'];
        }
        $html .= \html_writer::label(\get_string('selectquestion', 'local_quizanalytics'), 'spv-question-select');
        $html .= ' ' . \html_writer::select($questionoptions, 'spvquestion', $question, false, ['id' => 'spv-question-select']);
        $html .= ' ';

        $partoptions = [];
        for ($i = 1; $i <= $partsforquestion; $i++) {
            $partoptions[$i] = $i;
        }
        $html .= \html_writer::label(\get_string('selectpart', 'local_quizanalytics'), 'spv-part-select');
        $html .= ' ' . \html_writer::select($partoptions, 'spvpart', $part, false, ['id' => 'spv-part-select']);
        $html .= ' ';

        $studentoptions = ['' => \get_string('selectstudentnone', 'local_quizanalytics')];
        foreach ($meta['students'] as $s) {
            $studentoptions[$s['id']] = $s['name'];
        }
        $html .= \html_writer::label(\get_string('selectstudent', 'local_quizanalytics'), 'spv-student-select');
        $html .= ' ' . \html_writer::select($studentoptions, 'spvstudent', $studentid, false, ['id' => 'spv-student-select']);
        $html .= ' ';

        $html .= \html_writer::empty_tag('input', [
            'type' => 'submit', 'value' => \get_string('gobutton', 'local_quizanalytics'), 'class' => 'btn btn-secondary',
        ]);
        $html .= \html_writer::end_tag('form');
        return $html;
    }

    /**
     * A "Generate PDF Report" form: one checkbox per available section (all
     * checked by default, matching Streamlit's export-everything default),
     * plus whatever hidden params the target pdf.php needs to re-derive the
     * quiz/course/selection server-side. Deliberately GET, like every other
     * form in this plugin, so each distinct export stays a plain link.
     *
     * If $sections is empty (e.g. the report-sections lookup itself
     * failed), the form still renders with just the submit button — pdf.php
     * treats a request with no `sections[]` at all as "export every section",
     * so this degrades gracefully rather than blocking the download.
     *
     * @param \moodle_url $action    The pdf.php endpoint to submit to.
     * @param array $hiddenparams    name => value pairs pdf.php needs (id, colorblind, etc).
     * @param array<string, string> $sections Section id => localized checkbox
     *        label, from local_quizanalytics_quiz_api_client::report_sections($kind).
     *        The id (never the label) is what's posted back as each
     *        checkbox's value, so the posted selection stays meaningful
     *        regardless of the site's language.
     * @param string $buttonlabel
     * @param string $idprefix       Unique per rendered form, so checkbox ids never collide
     *        when this is called more than once on one page.
     * @param string $chartprefix    The DOM id prefix charts on this page were rendered
     *        with (render_vendor_and_payload()'s own $prefix — "qa"/"spv"/"qw") — read by
     *        sections-renderer.js's PDF submit handler to know which "{chartprefix}-
     *        chart-*" containers on the page to capture via Plotly.toImage() before
     *        this form actually submits. A plain POST form (not fetch/AJAX): the
     *        browser handles the PDF download response the same way it always has,
     *        the JS submit handler just fills in the chart_images field first.
     * @return string
     */
    public static function render_pdf_form(
        \moodle_url $action,
        array $hiddenparams,
        array $sections,
        string $buttonlabel,
        string $idprefix,
        string $chartprefix
    ): string {
        $html = \html_writer::start_tag('form', [
            'method' => 'post', 'action' => $action->out(false), 'class' => 'mb-3 qa-pdf-form',
            'data-chart-prefix' => $chartprefix,
        ]);
        foreach ($hiddenparams as $name => $value) {
            $html .= \html_writer::empty_tag('input', ['type' => 'hidden', 'name' => $name, 'value' => $value]);
        }
        // Populated by sections-renderer.js immediately before the form
        // actually submits — see setupPdfForm() there. Never trusted as
        // structured data server-side beyond "is this a plausible image":
        // pdf.php only ever uses these bytes as-is to embed in the
        // requesting user's own PDF, never to drive any computation.
        $html .= \html_writer::empty_tag('input', ['type' => 'hidden', 'name' => 'chart_images', 'value' => '']);
        foreach ($sections as $sectionid => $label) {
            $id = $idprefix . '-' . md5($sectionid);
            $html .= \html_writer::empty_tag('input', [
                'type' => 'checkbox', 'name' => 'sections[]', 'value' => $sectionid, 'id' => $id, 'checked' => 'checked',
            ]);
            $html .= ' ' . \html_writer::label($label, $id) . \html_writer::empty_tag('br');
        }
        $html .= \html_writer::tag('p', \get_string('pdfdownloadnotice', 'local_quizanalytics'), ['class' => 'text-muted mt-2 mb-0']);
        $html .= \html_writer::empty_tag('input', [
            'type' => 'submit', 'value' => $buttonlabel, 'class' => 'btn btn-primary mt-2',
        ]);
        $html .= \html_writer::end_tag('form');
        return $html;
    }
}
