import { BookOpen, Download, FileSpreadsheet, Layers, Settings, BarChart4 } from 'lucide-react';

export default function MoodleExportGuide() {
  return (
    <div className="space-y-6 max-w-4xl mx-auto">
      {/* Welcome Banner */}
      <div className="bg-gradient-to-r from-blue-600 to-indigo-700 rounded-2xl p-6 text-white shadow-xs relative overflow-hidden">
        <div className="absolute right-0 bottom-0 opacity-10 translate-x-12 translate-y-12">
          <BarChart4 className="w-64 h-64" />
        </div>
        <div className="relative z-10 max-w-2xl space-y-2">
          <span className="bg-blue-500/30 text-white font-bold text-[9px] uppercase tracking-widest px-2.5 py-1 rounded-full border border-white/10">Interactive STACK Data</span>
          <h2 className="text-xl md:text-2xl font-black tracking-tight">Interactive Quiz & Question Analytics</h2>
          <p className="text-xs md:text-sm text-blue-50/90 leading-relaxed">
            Analyze overall grade distributions, calculate correlation metrics, and drill down into specific student responses and potential response trees (PRTs) for STACK questions.
          </p>
        </div>
      </div>

      <div className="bg-white border border-slate-200 rounded-2xl p-6 shadow-xs">
        <h2 className="text-xl md:text-2xl font-bold text-slate-900 mb-2 flex items-center gap-2">
          <BookOpen className="w-6 h-6 text-blue-600 shrink-0" />
          How to Export Quiz Attempt Data from Moodle
        </h2>
        <p className="text-slate-600 text-sm mb-6 leading-relaxed">
          Follow these step-by-step instructions to download compliant quiz attempt datasets from your Moodle course. You can perform two types of analytics based on the reports you export.
        </p>

        {/* Step List */}
        <div className="bg-slate-50 border border-slate-150 rounded-xl p-5 mb-8">
          <h3 className="font-bold text-slate-900 text-sm mb-4 flex items-center gap-2">
            <Settings className="w-4 h-4 text-blue-600" />
            General Export Steps
          </h3>
          <div className="grid md:grid-cols-2 gap-6 text-xs text-slate-600">
            <div className="space-y-4">
              <div className="flex gap-3">
                <div className="bg-blue-50 text-blue-600 font-bold w-6 h-6 rounded-full flex items-center justify-center shrink-0 text-[11px]">1</div>
                <div>
                  <h4 className="font-semibold text-slate-950">Navigate to Your Quiz</h4>
                  <p className="text-slate-500 mt-0.5">Click into the specific STACK or standard Moodle quiz in your course workspace.</p>
                </div>
              </div>

              <div className="flex gap-3">
                <div className="bg-blue-50 text-blue-600 font-bold w-6 h-6 rounded-full flex items-center justify-center shrink-0 text-[11px]">2</div>
                <div>
                  <h4 className="font-semibold text-slate-950">Open Quiz Results</h4>
                  <p className="text-slate-500 mt-0.5">Select "Results" from the quiz secondary menu or settings menu.</p>
                </div>
              </div>
            </div>

            <div className="space-y-4">
              <div className="flex gap-3">
                <div className="bg-blue-50 text-blue-600 font-bold w-6 h-6 rounded-full flex items-center justify-center shrink-0 text-[11px]">3</div>
                <div>
                  <h4 className="font-semibold text-slate-950">Choose Report Type (Grades vs. Responses)</h4>
                  <p className="text-slate-500 mt-0.5">Select either "Grades" or "Responses" from the Moodle report dropdown, depending on your target workflow (see below).</p>
                </div>
              </div>

              <div className="flex gap-3">
                <div className="bg-blue-50 text-blue-600 font-bold w-6 h-6 rounded-full flex items-center justify-center shrink-0 text-[11px]">4</div>
                <div>
                  <h4 className="font-semibold text-slate-950">Download Table Data</h4>
                  <p className="text-slate-500 mt-0.5">Scroll to the bottom of the page, select Comma Separated Values (.csv) or Microsoft Excel (.xlsx), and click "Download".</p>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Dual Workflows */}
        <div className="grid md:grid-cols-2 gap-6">
          {/* Workflow A: Quiz Analysis Only */}
          <div className="bg-white border border-slate-200 rounded-xl p-5 space-y-4 shadow-2xs">
            <div className="flex items-center gap-2 pb-2 border-b border-slate-100">
              <div className="bg-blue-50 p-1.5 rounded-lg text-blue-600">
                <FileSpreadsheet className="w-4.5 h-4.5" />
              </div>
              <div>
                <h3 className="font-bold text-slate-900 text-sm">A. Quiz Analysis Only</h3>
                <p className="text-[10px] text-slate-400">For overall grade trends, box plots, and scatter charts</p>
              </div>
            </div>
            
            <p className="text-slate-600 text-xs leading-relaxed">
              To evaluate overall quiz attempts, export the standard **Grades** report. The downloaded spreadsheet must contain exactly these columns:
            </p>

            <div className="bg-slate-50 border border-slate-150 rounded-lg p-3.5 font-mono text-[11px] text-slate-700 space-y-1">
              <div>• <span className="font-semibold text-slate-900">Surname</span></div>
              <div>• <span className="font-semibold text-slate-900">First name</span></div>
              <div>• <span className="font-semibold text-slate-900">Email address</span></div>
              <div>• <span className="font-semibold text-slate-900">State</span></div>
              <div>• <span className="font-semibold text-slate-900">Started on</span></div>
              <div>• <span className="font-semibold text-slate-900">Completed</span></div>
              <div>• <span className="font-semibold text-slate-900">Time taken</span></div>
              <div>• <span className="font-semibold text-slate-900">Grade/10.00</span></div>
            </div>
          </div>

          {/* Workflow B: Question & PRT Analysis */}
          <div className="bg-white border border-slate-200 rounded-xl p-5 space-y-4 shadow-2xs">
            <div className="flex items-center gap-2 pb-2 border-b border-slate-100">
              <div className="bg-indigo-50 p-1.5 rounded-lg text-indigo-600">
                <Layers className="w-4.5 h-4.5" />
              </div>
              <div>
                <h3 className="font-bold text-slate-900 text-sm">B. Question & PRT Analysis</h3>
                <p className="text-[10px] text-slate-400">For detailed question-level metrics and PRT trees</p>
              </div>
            </div>
            
            <p className="text-slate-600 text-xs leading-relaxed">
              To evaluate specific question states and potential response trees (PRTs), export the **Responses** report. This process is identical to exporting grades, except you select "Responses" instead of "Grades".
            </p>

            <p className="text-slate-500 text-[10px] leading-relaxed">
              The responses report contains the identical columns above, plus a Response column for each question item:
            </p>

            <div className="bg-slate-50 border border-slate-150 rounded-lg p-3.5 font-mono text-[11px] text-slate-700 space-y-1">
              <div className="text-slate-400">// Identical metadata columns:</div>
              <div>• <span className="font-semibold text-slate-900">Surname</span>, <span className="font-semibold text-slate-900">First name</span>, <span className="font-semibold text-slate-900">Email address</span>...</div>
              <div>• <span className="font-semibold text-slate-900">Grade/10.00</span></div>
              <div className="text-slate-400 mt-2">// Plus response columns:</div>
              <div>• <span className="font-semibold text-slate-900">Response 1</span></div>
              <div>• <span className="font-semibold text-slate-900">Response 2</span></div>
              <div>• <span className="font-semibold text-slate-900">Response 3</span></div>
              <div>• <span className="font-semibold text-slate-400">...</span></div>
              <div>• <span className="font-semibold text-slate-900">Response N</span> <span className="text-slate-400 font-sans text-[10px]">(matches your quiz count)</span></div>
            </div>
          </div>
        </div>

        {/* Action / Help Section */}
        <div className="mt-8 pt-6 border-t border-slate-100 flex flex-col sm:flex-row justify-between items-center gap-4">
          <div className="space-y-1">
            <h4 className="font-semibold text-slate-900 text-xs">Want to try some sample data?</h4>
            <p className="text-slate-500 text-[10px]">Download pre-configured mock quizzes and response reports to see the app in action.</p>
          </div>
          <a
            href="https://drive.google.com/drive/folders/1r7c1asoMFwaLORaQVKisJk7xpWazzC5I?usp=sharing"
            target="_blank"
            rel="noreferrer noopener"
            className="inline-flex items-center gap-1.5 text-xs font-semibold text-blue-600 hover:text-blue-800 transition bg-blue-50/55 hover:bg-blue-50 px-3.5 py-1.5 rounded-lg border border-blue-100 shadow-3xs"
          >
            <Download className="w-3.5 h-3.5" />
            Download Sample Quiz Files
          </a>
        </div>

        <div className="mt-8 pt-6 border-t border-slate-100 flex flex-col md:flex-row justify-between items-center gap-4 text-xs text-slate-400">
          <p>Moodle STACK Analytics Hub is open-source and fully client-side.</p>
          <p>No quiz data is ever uploaded to external servers.</p>
        </div>
      </div>
    </div>
  );
}
