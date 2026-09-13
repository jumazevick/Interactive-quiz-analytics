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
 * English language strings for local_quizanalytics.
 *
 * The full merged string set (the original local_quizanalytics's +
 * local_stackanalytics's, minus the handful of overlapping keys
 * reconciled into one value).
 *
 * @package local_quizanalytics
 * @copyright  2026 Ernest Ting <eting@caltech.edu>
 * @license    http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */

defined('MOODLE_INTERNAL') || die();

$string['pluginname'] = 'STACK q-type Analytics';
$string['quizanalytics:view'] = 'View STACK quiz analytics, models, and diagnostics';
$string['sectionselectorlabel'] = 'Section:';
// The on-screen H1 heading every section's own page shows, kept as its own
// string (distinct from 'pluginname', used for the plugin's own
// admin-facing/nav-link name, and 'dashboardtitle', used for the
// Model & Diagnostics PDF's own title) — previously each section showed a
// different heading text switching depending which one you were on.
$string['pagemaintitle'] = 'STACK q-type Analytics';
$string['sectionquiz'] = 'Quiz Analytics';
$string['sectionquestion'] = 'Question Analytics';
$string['sectionmodels'] = 'Model Analytics';
$string['sectiondiagnostics'] = 'Diagnostics Analytics';
$string['privacy:metadata'] = 'The STACK q-type Analytics plugin does not store any personal data of its own. It reads finished quiz attempts, question responses, grades, and log events directly from Moodle\'s own database (mod_quiz, the question engine, grade_grades, and logstore_standard_log) at request/calculation time, all of which are already covered by their own privacy providers.';

// Quiz Analytics section (index.php, quizanalyticspdf.php — ported from the
// standalone local_quizanalytics plugin this merges). Values unchanged from
// that plugin's own lang file; only the two strings superseded by this
// plugin's unified pluginname/capability (its own 'pluginname' and
// 'quizanalytics:view') were dropped here.
$string['anonymizemode'] = 'Anonymize student data';
$string['anonymizedstudent'] = 'Student {$a}';
$string['cachedef_questionanalysis'] = 'The Question Analytics result for one quiz.';
$string['cachedef_quizanalysiscoursewide'] = 'The course-wide Quiz Analysis result for one course.';
$string['cachedef_solutionprocess'] = 'The Solution Process Visualization result for one quiz/question/part/student selection.';
$string['cachedef_solutionprocessmeta'] = 'The question/part/student lists used to populate the Solution Process Visualization selector form, for one quiz.';
$string['colorblindmode'] = 'Colorblind mode';
$string['computetimelimit']      = 'Computation time limit (seconds)';
$string['computetimelimit_desc'] = 'Raises PHP\'s own execution time limit before the heaviest analytics computations (course-wide analysis, and any PDF export). These run in-process rather than calling a separate service, so a course with many STACK quizzes/students may need longer than PHP\'s normal max_execution_time allows. 0 leaves PHP\'s own default in place.';
$string['coursewideheading']    = 'Course-Wide Analytics';
$string['downloadpdfbutton']  = 'Download PDF';
$string['generatepdfheading'] = 'Generate PDF report';
$string['gobutton']             = 'View';
$string['gradetypeaverage']     = 'Average grade';
$string['gradetypehighest']     = 'Highest grade';
$string['gradetypelabel']       = 'Compare attempts against:';
$string['gradetypeminimum']     = 'Minimum grade';
$string['loaderror']            = 'Analytics returned an unexpected response.';
$string['noattempts']           = 'No finished attempts yet for this quiz. Analytics will appear once at least one student has completed it.';
$string['nocourseattempts']     = 'None of this course\'s STACK quizzes have finished attempts yet.';
$string['nostackquestions']     = 'This quiz has no STACK questions to visualize.';
$string['nostackquizzes']       = 'This course has no STACK quizzes yet, or none have finished attempts.';
$string['pagetitle']            = 'Quiz analytics';
$string['pdfchartunavailable']  = '{$a}: chart image unavailable (not captured from the page).';
$string['pdfdownloadnotice']    = 'Generating this PDF may take a while for a large course. Please wait for the download to finish.';
$string['pdferror']           = 'The PDF report could not be generated. Contact your Moodle administrator.';
$string['pdfnosections']        = 'No sections were selected for this report.';
$string['pdfquizsubtitle']        = 'Combined across every STACK quiz in the course';
$string['pdfsectionattemptlist']       = '1. Student Quiz Summary';
$string['pdfsectionboxplot']           = '3. Quiz Grade Distribution (Box Plot)';
$string['pdfsectioncrossattempt']      = 'Cross-Attempt Comparison';
$string['pdfsectionengagement']        = '4. Engagement Over Time';
$string['pdfsectionnetworkfeatures']   = 'Network Features per Node';
$string['pdfsectionprtdistance3d']     = 'PRT-Distance 3D Chart';
$string['pdfsectionquestiondetails']        = 'Question Review';
$string['pdfsectionquestiondetailscaption'] = 'Inspect the question, expected answer, and common response patterns to understand where students had difficulty.';
$string['pdfsectionquizstats']         = '2. Summary of Quiz Stats';
$string['pdfsectionresponseoverview'] = 'Question Response Overview';
$string['pdfsectionscatter']           = '5. Scatter Plot: Attempts vs Grades';
$string['pdfsectionsummary']        = 'Quiz Snapshot';
$string['pdfsectionsummarycaption'] = 'Participation and summary statistics';
$string['pdfsectiontransitiongraph']   = 'Class-Wide Transition Graph';
$string['pdfsectiontreeeditdistance3d'] = 'Tree Edit Distance 3D Chart';
$string['pdfsectiontrend']             = '6. Line Graph of Various Metrics';
$string['pdfsolutionprocesssubtitle'] = '{$a->question}, part {$a->part}';
$string['pdftitlequestion']       = '{$a}: Question Analytics';
$string['pdftitlequiz']           = '{$a}: Quiz Analysis';
$string['pdftitlesolutionprocess'] = '{$a}: Solution Process Visualization';
$string['pdftruncatedrows']       = 'Showing the first {$a->shown} of {$a->total} rows.';
$string['quizselectoption']     = 'All STACK quizzes (course-wide view)';
$string['selectpart']           = 'Part';
$string['selectquestion']       = 'Question';
$string['selectstudent']        = 'Student drill-down';
$string['selectstudentnone']    = 'None';
$string['servererror']          = 'Analytics could not be computed for this quiz. Contact your Moodle administrator.';
$string['viewquestionanalytics'] = 'Question Analytics';
$string['viewsolutionprocess']  = 'Solution Process Visualization';

// Model Analytics + Diagnostics Analytics sections (modelanalytics.php,
// modelanalyticspdf.php, diagnosticsanalytics.php, diagnosticsanalyticspdf.php
// — ported from the standalone local_stackanalytics plugin this merges,
// which combined them into one page/PDF; split into two sections here, see
// classes/section_selector.php). Values unchanged from that plugin's own
// lang file; its own 'pluginname',
// 'privacy:metadata', 'stackanalytics:view' (superseded by this plugin's
// unified strings/capability — the old capability id doesn't exist here at
// all), 'downloadpdfbutton', and 'pdfnosections' (identical text to the
// Quiz Analytics section's own copies above) were dropped here rather than
// duplicated.
$string['indicator:gradetrajectory'] = 'STACK grade trajectory';
$string['indicator:responselatencyanomaly'] = 'Anomalous STACK response latency';
$string['indicator:disengagemententropy'] = 'STACK disengagement entropy';
$string['indicator:helpseekinggap'] = 'STACK help-seeking gap';
$string['indicator:feedbackrevisiondistance'] = 'STACK feedback revision distance';
$string['target:studentatrisk'] = 'Student at risk in a STACK-based course';
$string['errornostackactivity'] = 'This course has no STACK (qtype_stack) question activity';
$string['indicator:questiondifficultyirt'] = 'STACK question difficulty';
$string['indicator:syntaxerrorrate'] = 'STACK syntax-error rate';
$string['indicator:unreachednoderatio'] = 'STACK PRT unreached-node ratio';
$string['indicator:feedbackineffectiveness'] = 'STACK feedback ineffectiveness';
$string['target:questionneedsreview'] = 'STACK question/PRT needs review';
$string['dashboardtitle'] = 'STACK q-type Analytics Dashboard';
$string['courseselectorlabel'] = 'Course:';
$string['quizselectorlabel'] = 'Quiz:';
$string['viewselectorlabel'] = 'View:';
$string['allquizzes'] = 'All Quizzes';
$string['largecoursenotice'] = 'This may take a little time to load for a large course. Please wait for the results below.';
$string['seedbiasheading'] = 'Seed bias (one-way ANOVA across random seeds)';
$string['bloatedtreeheading'] = 'PRT branch coverage';
$string['seedgroups'] = 'Distinct seeds observed';
$string['notenoughdata'] = 'Not enough attempt data yet to compute this.';
$string['noattemptsyet'] = 'No attempts recorded yet.';
$string['notenoughdatacount'] = 'Not enough attempt data yet to compute this ({$a} attempt(s) so far).';
$string['notavailable'] = 'n/a';
$string['etamagnitude_negligible'] = 'negligible effect';
$string['etamagnitude_small'] = 'small effect';
$string['etamagnitude_medium'] = 'medium effect';
$string['etamagnitude_large'] = 'large effect';
$string['node'] = 'Node';
$string['branch'] = 'Branch';
$string['traversals'] = 'Traversals observed';
$string['coverage'] = 'Coverage';
$string['coverage_unreached'] = 'Never reached: pruning candidate';
$string['coverage_low_traffic'] = 'Low traffic: review before pruning';
$string['coverage_adequate'] = 'Adequately traversed';
$string['unknownquestion'] = 'Unknown question';
$string['unknownquiz'] = 'Unknown quiz';
$string['model1heading'] = 'Model 1: Student Risk & Behavior';
$string['model1intro'] = 'Predicts which students are at risk of not passing the course, from five behavioral signals in their STACK question activity. It\'s recomputed at points through the course, so a warning can fire before the course ends rather than only at the final grade.';
$string['aboutthismodel'] = 'About this model';
$string['model1aboutbody'] = 'What\'s actually predicted (the "target") is simple: will this student\'s final grade fall below the course\'s own pass grade? The five indicators below are what a trained model would use as evidence for that prediction. Today, before any model is trained, this page just shows each indicator\'s current reading directly.';
$string['model1aboutfooter'] = 'This model ships disabled, so nothing here is a trained AI prediction yet. Only live readings of each signal are shown. An administrator can enable and train it under Site Administration > Analytics > Models, after which trained predictions appear in Moodle\'s own Insights report.';
$string['model1nostudents'] = 'No students are enrolled in this course yet.';
$string['columnstudent'] = 'Student';
$string['columncurrentstatus'] = 'Current status';
$string['gradestatusatrisk'] = 'At risk: {$a->grade}%, below the {$a->gradepass}% needed to pass';
$string['gradestatuspassing'] = 'On track: {$a->grade}%, at or above the {$a->gradepass}% needed to pass';
$string['gradestatusnogradeyet'] = 'No grade recorded yet';
$string['gradestatusnothreshold'] = 'This course has no pass grade set';
$string['band_good'] = 'Good';
$string['band_neutral'] = 'Typical';
$string['band_watch'] = 'Worth a look';
$string['truncatednotice'] = 'Showing the first {$a->shown} of {$a->total}. Use the selectors above to narrow this down.';
$string['model1desc_gradetrajectory'] = 'How this student\'s STACK scores compare to full marks.';
$string['model1sentence_gradetrajectory'] = 'Averaging {$a->meanpercent}% across {$a->attempts} finished attempt(s).';
$string['model1desc_responselatencyanomaly'] = 'Whether this student answers implausibly fast compared to the class. This is a correlational flag only, never evidence of misconduct on its own.';
$string['model1sentence_responselatencyanomaly'] = 'Averages {$a->userseconds}s between tries, vs. a class average of {$a->cohortseconds}s.';
$string['model1desc_disengagemententropy'] = 'Whether this student\'s attempts look mechanical (very regular timing, questions abandoned) rather than genuine problem-solving.';
$string['model1sentence_disengagemententropy'] = '{$a->abandonedcount} of {$a->attempts} attempt(s) abandoned before completion.';
$string['model1desc_helpseekinggap'] = 'Whether this student seeks help (forums, glossary, other resources) after a wrong answer as often as their classmates do.';
$string['model1sentence_helpseekinggap'] = 'Seeks help after {$a->studentpercent}% of mistakes, vs. a class average of {$a->baselinepercent}%.';
$string['model1desc_feedbackrevisiondistance'] = 'Whether this student meaningfully changes their answer after seeing feedback, or resubmits close to the same thing.';
$string['model1sentence_feedbackrevisiondistance'] = 'Changes their answer by {$a->changepercent}% on average, across {$a->revisions} revision(s).';
$string['model2heading'] = 'Model 2: Question & PRT Quality';
$string['model2intro'] = 'One row per STACK question (with the quiz it belongs to shown underneath), flagging ones that may be worth an instructor\'s review from four signals in how students actually answer them, including their PRT, the step-by-step marking logic that checks the answer and gives feedback.';
$string['model2aboutbody'] = 'What\'s actually predicted (the "target") is: does this question\'s pass rate fall below a threshold (50% by default, an admin setting)? The four indicators below are the evidence a trained model would use for that. Today, before any model is trained, this page just shows each indicator\'s current reading directly. Note: this pass-rate read and the difficulty indicator both ultimately come from the same pass rate, so treat "needs review" and "difficult" as related, not independent, signals.';
$string['model2noquestions'] = 'No STACK questions to show for this selection.';
$string['columnquestion'] = 'Question';
$string['quizlabel'] = 'Quiz: {$a}';
$string['quizoptionlabel'] = '{$a->name} ({$a->count} STACK question(s))';
$string['needsreviewyes'] = 'Needs review: {$a->passpercent}% pass rate, below the {$a->thresholdpercent}% threshold';
$string['needsreviewno'] = 'No flag: {$a->passpercent}% pass rate, at or above the {$a->thresholdpercent}% threshold';
$string['model2desc_questiondifficultyirt'] = 'How hard this question is in practice, from its empirical pass rate.';
$string['model2sentence_questiondifficultyirt'] = '{$a->passpercent}% pass rate across {$a->attempts} finished attempt(s).';
$string['model2desc_syntaxerrorrate'] = 'Whether most of this question\'s wrong answers are input/syntax mistakes (an input-format problem) rather than genuine maths errors.';
$string['model2sentence_syntaxerrorrate'] = '{$a->syntaxerrorcount} of {$a->totalfailed} failed attempt(s) were syntax/input errors.';
$string['model2desc_unreachednoderatio'] = 'How much of this question\'s PRT branching logic has never actually been exercised by a real attempt, a pruning candidate if it stays that way.';
$string['model2sentence_unreachednoderatio'] = '{$a->unreachedcount} of {$a->totalbranches} PRT branch(es) never reached.';
$string['model2desc_feedbackineffectiveness'] = 'Whether students who get this wrong tend to improve on their next try more than they would on a fresh question, a rough read on whether the feedback is actually helping.';
$string['model2sentence_feedbackineffectiveness'] = '{$a->improvepercent}% improve after a wrong try, vs. a {$a->baselinepercent}% first-try baseline.';
$string['diagnosticsheading'] = 'Diagnostics Dashboard';
$string['diagnosticsintrosummary'] = 'What Seed Bias and PRT Branch Coverage mean';
$string['diagnosticsintro'] = 'Two checks per STACK question, listed below with the quiz it belongs to. Every time a student attempts a STACK question, Moodle picks a random "seed" that changes its numbers (e.g. different coefficients) while keeping the same structure. <strong>Seed bias</strong> checks whether some of those seed variants are unfairly harder or easier than others, so a low grade isn\'t just "you got the harder version". Each STACK question also grades answers through a PRT (its step-by-step marking/feedback logic, made of "branches" for different right/wrong paths). <strong>PRT branch coverage</strong> checks whether some of those branches have ever actually been triggered by a real student answer. A branch that\'s never reached is either working feedback nobody\'s needed yet, or dead logic worth simplifying. A "Worth a look" badge is a prompt to open that question and check it makes sense for how you designed it, not proof something is broken. Click a question below to see the full numbers behind its badges.';
$string['conceptdependencynote'] = 'Concept-dependency mapping (finding which questions\' failures tend to predict failures on others) isn\'t implemented in this plugin yet. The architecture doc frames it as offline sequence-mining work outside a live dashboard page, not something to half-build here. Noted so it doesn\'t just silently not appear.';
$string['diagnosticsnoquestions'] = 'No STACK questions to show for this selection.';
$string['diagnosticsseedbiassentence'] = 'η²={$a->etasquared} ({$a->magnitude})';
$string['diagnosticsbloatedtreesentence'] = '{$a->unreached} of {$a->total} branch(es) never reached';
$string['modelpageintro'] = '<strong>Model 1</strong> looks at each student\'s behavior and flags who might be at risk of not passing. <strong>Model 2</strong> looks at each question\'s marking logic and flags ones that might be worth a teacher\'s review. Use the "View:" selector below to switch between them. Both models are disabled by default, so everything below is a live reading of each signal today, not a trained AI prediction. An administrator can enable and train a model under Site Administration, Analytics, Models. Once trained, real predictions appear alongside this page in Moodle\'s own Insights report.';
$string['modelpageintrosummary'] = 'About Model 1 and Model 2';
$string['diagnosticspageintro'] = 'The Diagnostics Dashboard is a set of statistical reports that don\'t fit either model: Seed Bias and PRT Branch Coverage, described below. These are not predictions, just direct calculations from the same attempt data.';
$string['diagnosticspageintrosummary'] = 'About the Diagnostics Dashboard';
$string['responsibleusecallout'] = 'A few things worth keeping in mind when reading the flags below. These are statistical patterns, not proof of anything. An anomalous response time is a prompt to check in with a student, not evidence of misconduct on its own. Small courses will show noisier and less reliable readings simply because they have fewer data points to work from. Every number here describes what a student did in this course, not who they are.';
$string['responsibleusesummary'] = 'Responsible use: a few things to keep in mind';
$string['pdfsectionslabel'] = 'Include in the PDF:';
$string['pdfnorows'] = 'Nothing to show for this section: no data yet, or nothing matched the current filters.';
$string['pdffooternote'] = 'STACK q-type Analytics Dashboard: live indicator readings, not a trained AI prediction';
$string['questionneedsreviewthreshold'] = 'Question-needs-review pass-rate threshold';
$string['questionneedsreviewthreshold_desc'] = 'A question is labeled "needs review" (Model 2\'s proxy label) when its empirical pass rate falls below this value (0.0-1.0). See the architecture doc\'s §3.3 circularity caveat before lowering this to chase a particular result.';
$string['task:warmanalyticscache'] = 'Warm STACK q-type Analytics result caches';
$string['task:warmsingleview'] = 'Warm one Quiz/Question Analytics view (on-demand background compute)';
$string['generatinginbackground'] = 'This may take a little time to load for a large course. It\'s being computed in the background. This page checks again automatically every 20 seconds. No need to reload it yourself.';
$string['showingstale'] = 'Showing the last completed report while a newer calculation runs in the background.';
$string['generatingstale'] = 'This has been queued for {$a} without finishing. That is longer than any real background compute has taken on this site, so this most likely means Moodle\'s cron isn\'t running, or the background task crashed or ran out of memory. Ask your Moodle administrator to check Site administration → Server → Scheduled tasks (is cron running at all?) and Site administration → Server → Tasks → Task logs (did this task fail?) rather than waiting longer.';
$string['cronstatusheading'] = 'Cron status';
$string['cronstatuswarning'] = 'This plugin depends on Moodle cron running regularly.';
$string['detectedresourcesheading'] = 'Detected server resources';
$string['detectedresources'] = 'Detected {$a->cores} CPU cores and {$a->memorygb} GB RAM on this server. Recommended cache-warming parallel workers: {$a->workers}. Applied automatically on install/upgrade. Press "Re-detect" below if this server\'s hardware has changed since (e.g. moved from a laptop to a dedicated server, or the reverse).';
$string['detectedresourcesfailed'] = 'Could not detect this server\'s CPU/RAM (unusual host or restricted container). Using this plugin\'s original conservative static defaults instead. The "Cache-warming parallel workers" setting below can still be set by hand.';
$string['redetectbutton'] = 'Re-detect and apply now';
$string['backgroundtimebudget'] = 'On-demand background-compute time budget (seconds)';
$string['backgroundtimebudget_desc'] = 'When a visitor hits a cold cache (Quiz Analytics, Question Analytics, or Solution Process Visualization), this plugin first times a real, small sample fetch (~100 attempts) for the actual quiz on this actual host, then extrapolates the full cost from that. This isn\'t a fixed attempt-count guess, since per-attempt cost is genuinely question-complexity- and host-speed-dependent (measured directly: a simple randomised question computed roughly 10x cheaper per attempt than a complex real one). If the estimate exceeds this many seconds, the compute is handed to a background task instead of run on that request. The synchronous on-demand path has no forking and no upper bound on quiz/course size, so a large enough one can outlive a reverse proxy\'s own timeout (Cloudflare\'s free/Pro default is ~100s) before this plugin\'s own ignore_user_abort(true) protection would even get a chance to help. The visitor sees a "generating in the background" notice and needs to revisit the page once it\'s done, rather than waiting on the same request. Set to 0 to disable and always compute inline (the old behaviour). The default leaves real margin under a 100s proxy timeout even after adding the analysis step on top and the sampling estimate\'s own conservative bias.';
$string['parallelworkers'] = 'Cache-warming parallel workers';
$string['parallelworkers_desc'] = 'The "Warm STACK q-type Analytics result caches" scheduled task forks up to this many worker processes to fetch several quizzes\' attempt data concurrently (CLI/cron only. This has no effect on the on-demand web pages, which always fetch serially). Each STACK question\'s response summary is graded live via CAS every time it\'s fetched, which is I/O-bound waiting on the Maxima backend, so running several quizzes\' fetches at once can meaningfully cut a large course\'s warm-up time. 1 disables forking (falls back to the original serial fetch). Keep this at or below your Maxima backend\'s own worker/queue capacity and your database\'s available connections. Every worker opens its own DB connection and makes CAS calls concurrently with the others. Also see "Cache-warming worker memory limit" below. The two settings need sizing together against your server\'s actual available RAM.';
$string['parallelworkermemory'] = 'Cache-warming worker memory limit (MB)';
$string['parallelworkermemory_desc'] = 'The PHP memory limit (in MB) each cache-warming worker process is allowed, including the non-forked case (a single fetch still uses this limit). Each worker\'s own fetch streams one quiz\'s records to disk at a time rather than accumulating its whole share in memory, but a single very large quiz\'s own attempt data still has to fit in memory on its own. Measured directly, one 2,136-attempt quiz alone peaked at just under 1GB, which is why the default here is higher than that. Size this together with "Cache-warming parallel workers": if that many workers can run at the same time, make sure workers × this value comfortably fits under your server\'s real available RAM, with room left for MariaDB/Postgres, the Maxima backend, and everything else already running. Going over doesn\'t just fail this one course cleanly, it risks the kernel\'s own out-of-memory killer picking an unrelated process (the database, Maxima) instead, which is a much bigger problem. When in doubt, lower parallelworkers rather than raising this past what you know is actually free.';
$string['lowtrafficfloor'] = 'Bloated-tree "low traffic" floor';
$string['lowtrafficfloor_desc'] = 'On the Diagnostics Dashboard, a PRT branch with at least one but fewer than this many observed traversals is reported as "low traffic" (needs a human look) rather than "never reached" (a pruning candidate).';
$string['helpseekinglookback'] = 'Help-seeking lookback window (seconds)';
$string['helpseekinglookback_desc'] = 'How long after a STACK question failure a forum/glossary/resource access still counts as "seeking help for it", for the help-seeking-gap indicator. Defaults to one hour.';
