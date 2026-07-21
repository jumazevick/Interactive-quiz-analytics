import { useState, useMemo } from 'react';
import {
  TrendingDown,
  AlertTriangle,
  CheckCircle2,
  BarChart4,
  ArrowUpDown,
  BookOpen
} from 'lucide-react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as ChartTooltip,
  Legend,
  ResponsiveContainer,
  Cell
} from 'recharts';

import { ResponseLevelRow, QuestionMetricsSummary } from '../types';
import {
  buildPRTLevelTable,
  buildQuestionMetricsSummary,
  buildPRTPassRates,
  rankMostDifficultQuestions,
  buildRepeatedAttemptsSummary
} from '../utils/questionProcessor';

interface QuestionAnalysisSectionProps {
  responseRows: ResponseLevelRow[];
  selectedQuizzes: number[];
  uploadedFiles: { name: string; size: number; quizID: number }[];
}

export default function QuestionAnalysisSection({
  responseRows,
  selectedQuizzes,
  uploadedFiles
}: QuestionAnalysisSectionProps) {
  // Sort state for tables
  const [metricsSortField, setMetricsSortField] = useState<keyof QuestionMetricsSummary>('question');
  const [metricsSortOrder, setMetricsSortOrder] = useState<'asc' | 'desc'>('asc');

  // Detect unique quizIDs in response dataset
  const quizzesInResponseData = useMemo(() => {
    return Array.from(new Set(responseRows.map(r => r.quizID))).sort((a, b) => a - b);
  }, [responseRows]);

  const quizOptions = useMemo(() => {
    return quizzesInResponseData.map(qid => {
      const file = uploadedFiles.find(f => f.quizID === qid);
      return {
        id: qid,
        name: file ? file.name : `Quiz ${qid}`
      };
    });
  }, [quizzesInResponseData, uploadedFiles]);

  const [selectedQuizId, setSelectedQuizId] = useState<number | null>(null);

  const activeQuizId = useMemo(() => {
    if (selectedQuizId !== null && quizzesInResponseData.includes(selectedQuizId)) {
      return selectedQuizId;
    }
    return quizzesInResponseData[0] || null;
  }, [selectedQuizId, quizzesInResponseData]);

  // Filter rows based on selected quizzes (if 1 quiz) or active dropdown quiz (if multiple)
  const filteredResponseRows = useMemo(() => {
    const isMultiQuiz = quizzesInResponseData.length > 1;
    if (isMultiQuiz && activeQuizId !== null) {
      return responseRows.filter(r => r.quizID === activeQuizId);
    }
    return responseRows.filter(r => selectedQuizzes.includes(r.quizID));
  }, [responseRows, selectedQuizzes, quizzesInResponseData, activeQuizId]);

  // Generate data tables
  const prtLevelRows = useMemo(() => {
    return buildPRTLevelTable(filteredResponseRows);
  }, [filteredResponseRows]);

  const questionMetricsSummary = useMemo(() => {
    return buildQuestionMetricsSummary(filteredResponseRows);
  }, [filteredResponseRows]);

  const prtPassRates = useMemo(() => {
    return buildPRTPassRates(prtLevelRows);
  }, [prtLevelRows]);

  const difficultQuestions = useMemo(() => {
    return rankMostDifficultQuestions(questionMetricsSummary, 10);
  }, [questionMetricsSummary]);

  const repeatedWrongSummary = useMemo(() => {
    return buildRepeatedAttemptsSummary(filteredResponseRows);
  }, [filteredResponseRows]);

  // Sort Question Summary Table
  const sortedQuestionMetrics = useMemo(() => {
    return [...questionMetricsSummary].sort((a, b) => {
      let valA = a[metricsSortField];
      let valB = b[metricsSortField];

      if (typeof valA === 'string' && typeof valB === 'string') {
        const numA = parseInt(valA.replace(/\D/g, ''), 10) || 0;
        const numB = parseInt(valB.replace(/\D/g, ''), 10) || 0;
        return metricsSortOrder === 'asc' ? numA - numB : numB - numA;
      }

      valA = valA as number;
      valB = valB as number;
      return metricsSortOrder === 'asc' ? valA - valB : valB - valA;
    });
  }, [questionMetricsSummary, metricsSortField, metricsSortOrder]);

  const handleSort = (field: keyof QuestionMetricsSummary) => {
    if (metricsSortField === field) {
      setMetricsSortOrder(prev => (prev === 'asc' ? 'desc' : 'asc'));
    } else {
      setMetricsSortField(field);
      setMetricsSortOrder('asc');
    }
  };

  // Heatmap helper matrix
  const heatmapData = useMemo(() => {
    const questions = Array.from(new Set(prtPassRates.map(p => p.question))).sort((a, b) => {
      const numA = parseInt(a.replace(/\D/g, ''), 10) || 0;
      const numB = parseInt(b.replace(/\D/g, ''), 10) || 0;
      return numA - numB;
    });

    const prts = Array.from(new Set(prtPassRates.map(p => p.prtName))).sort();

    return { questions, prts };
  }, [prtPassRates]);

  // Defensive Empty State Checks
  if (responseRows.length === 0) {
    return (
      <div className="bg-white border border-slate-200 rounded-2xl p-10 text-center shadow-xs max-w-2xl mx-auto my-8">
        <BarChart4 className="w-12 h-12 text-slate-300 mx-auto mb-4" />
        <h3 className="font-bold text-slate-800 text-lg">No Response Data Available</h3>
        <p className="text-slate-500 text-sm mt-1.5 leading-relaxed">
          Please upload your Moodle Quiz Responses export file first. The Question and PRT analysis relies on detailed response columns to extract potential response trees and error types.
        </p>
        <div className="mt-6 inline-flex items-center gap-2 text-xs font-semibold bg-indigo-50 border border-indigo-100 text-indigo-700 px-4 py-2 rounded-xl">
          <BookOpen className="w-4 h-4 shrink-0" />
          Refer to the Home page for instructions on obtaining the Responses report.
        </div>
      </div>
    );
  }

  // Heatmap color mapper
  const getHeatmapColor = (rate: number | undefined) => {
    if (rate === undefined) return 'bg-slate-50 text-slate-300';
    if (rate >= 85) return 'bg-emerald-500 text-white font-bold';
    if (rate >= 70) return 'bg-emerald-400/90 text-emerald-950 font-semibold';
    if (rate >= 50) return 'bg-amber-300 text-amber-950 font-medium';
    if (rate >= 30) return 'bg-orange-300 text-orange-950 font-medium';
    return 'bg-red-400 text-white font-bold';
  };

  return (
    <div className="space-y-8">
      {/* Quiz Selector Dropdown for Multi-Quiz Responses */}
      {quizzesInResponseData.length > 1 && (
        <div className="bg-white border border-slate-200 rounded-2xl p-5 shadow-xs flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <h4 className="text-sm font-bold text-slate-900 flex items-center gap-2">
              <BookOpen className="w-4 h-4 text-blue-600" />
              Select Quiz
            </h4>
            <p className="text-xs text-slate-500 mt-1">This dataset contains response data for multiple quizzes. Choose one to analyze below.</p>
          </div>
          <div className="w-full sm:w-72">
            <select
              value={activeQuizId || ''}
              onChange={(e) => setSelectedQuizId(Number(e.target.value))}
              className="w-full bg-white border border-slate-200 rounded-xl px-3 py-2 text-sm font-medium focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-slate-700 shadow-3xs"
            >
              {quizOptions.map(opt => (
                <option key={opt.id} value={opt.id}>
                  {opt.name}
                </option>
              ))}
            </select>
          </div>
        </div>
      )}

      {/* Overview Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-white border border-slate-200 p-5 rounded-2xl shadow-xs flex items-center gap-4">
          <div className="p-3 bg-blue-50 text-blue-600 rounded-xl">
            <BarChart4 className="w-5 h-5" />
          </div>
          <div>
            <span className="text-slate-400 text-[10px] uppercase tracking-wider font-bold">Total Questions</span>
            <p className="text-xl font-bold text-slate-900 mt-0.5">{questionMetricsSummary.length}</p>
          </div>
        </div>

        <div className="bg-white border border-slate-200 p-5 rounded-2xl shadow-xs flex items-center gap-4">
          <div className="p-3 bg-emerald-50 text-emerald-600 rounded-xl">
            <CheckCircle2 className="w-5 h-5" />
          </div>
          <div>
            <span className="text-slate-400 text-[10px] uppercase tracking-wider font-bold">Overall PRT Elements</span>
            <p className="text-xl font-bold text-slate-900 mt-0.5">
              {new Set(prtPassRates.map(p => `${p.question}_${p.prtName}`)).size}
            </p>
          </div>
        </div>

        <div className="bg-white border border-slate-200 p-5 rounded-2xl shadow-xs flex items-center gap-4">
          <div className="p-3 bg-red-50 text-red-600 rounded-xl">
            <TrendingDown className="w-5 h-5" />
          </div>
          <div>
            <span className="text-slate-400 text-[10px] uppercase tracking-wider font-bold">Most Difficult Q</span>
            <p className="text-xl font-bold text-slate-900 mt-0.5">
              {difficultQuestions[0]?.question || '-'}
            </p>
          </div>
        </div>

        <div className="bg-white border border-slate-200 p-5 rounded-2xl shadow-xs flex items-center gap-4">
          <div className="p-3 bg-amber-50 text-amber-600 rounded-xl">
            <AlertTriangle className="w-5 h-5" />
          </div>
          <div>
            <span className="text-slate-400 text-[10px] uppercase tracking-wider font-bold">Syntax Error Count</span>
            <p className="text-xl font-bold text-slate-900 mt-0.5 font-mono">
              {questionMetricsSummary.reduce((sum, q) => sum + q.syntaxErrorCount, 0)}
            </p>
          </div>
        </div>
      </div>

      {/* Visual Charts Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Chart 1: Correct vs Incorrect Bar Chart */}
        <div className="bg-white border border-slate-200 rounded-2xl p-5 shadow-xs">
          <div className="mb-4">
            <h3 className="font-bold text-slate-900 text-base">Response Outcome Percentages</h3>
            <p className="text-slate-500 text-xs mt-0.5">Ratio of Correct vs Incorrect submissions for every question item</p>
          </div>
          <div className="h-72 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={questionMetricsSummary} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
                <XAxis dataKey="question" stroke="#94a3b8" style={{ fontSize: 10 }} tickLine={false} />
                <YAxis stroke="#94a3b8" style={{ fontSize: 10 }} tickLine={false} domain={[0, 100]} />
                <ChartTooltip
                  content={({ active, payload }) => {
                    if (active && payload && payload.length) {
                      return (
                        <div className="bg-white/95 backdrop-blur-md border border-slate-200 p-3 rounded-xl shadow-lg text-xs space-y-1">
                          <p className="font-bold text-slate-950">{payload[0].payload.question}</p>
                          <p className="text-emerald-600 flex justify-between gap-4 font-semibold">
                            Correct: <span className="font-mono">{payload[0].value}%</span>
                          </p>
                          <p className="text-red-500 flex justify-between gap-4 font-semibold">
                            Incorrect: <span className="font-mono">{payload[1].value}%</span>
                          </p>
                          <p className="text-slate-400 text-[10px] mt-1">Attempts: {payload[0].payload.attempts}</p>
                        </div>
                      );
                    }
                    return null;
                  }}
                />
                <Legend style={{ fontSize: 11 }} />
                <Bar dataKey="percentCorrect" name="Correct %" fill="#10b981" radius={[4, 4, 0, 0]} />
                <Bar dataKey="percentIncorrect" name="Incorrect %" fill="#f43f5e" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Chart 2: Valid vs Invalid Bar Chart */}
        <div className="bg-white border border-slate-200 rounded-2xl p-5 shadow-xs">
          <div className="mb-4">
            <h3 className="font-bold text-slate-900 text-base">Valid vs Invalid Attempts</h3>
            <p className="text-slate-500 text-xs mt-0.5">Ratio of mathematically parsed answers vs invalid/syntax-error state attempts</p>
          </div>
          <div className="h-72 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={questionMetricsSummary} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
                <XAxis dataKey="question" stroke="#94a3b8" style={{ fontSize: 10 }} tickLine={false} />
                <YAxis stroke="#94a3b8" style={{ fontSize: 10 }} tickLine={false} domain={[0, 100]} />
                <ChartTooltip
                  content={({ active, payload }) => {
                    if (active && payload && payload.length) {
                      return (
                        <div className="bg-white/95 backdrop-blur-md border border-slate-200 p-3 rounded-xl shadow-lg text-xs space-y-1">
                          <p className="font-bold text-slate-955">{payload[0].payload.question}</p>
                          <p className="text-blue-600 flex justify-between gap-4 font-semibold">
                            Valid: <span className="font-mono">{payload[0].value}%</span>
                          </p>
                          <p className="text-amber-500 flex justify-between gap-4 font-semibold">
                            Invalid/Syntax error: <span className="font-mono">{payload[1].value}%</span>
                          </p>
                          <p className="text-slate-400 text-[10px] mt-1">Attempts: {payload[0].payload.attempts}</p>
                        </div>
                      );
                    }
                    return null;
                  }}
                />
                <Legend style={{ fontSize: 11 }} />
                <Bar dataKey="percentValid" name="Valid %" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                <Bar dataKey="percentInvalid" name="Invalid/Syntax Error %" fill="#f59e0b" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Chart 3: Ranked Difficulty Chart */}
        <div className="bg-white border border-slate-200 rounded-2xl p-5 shadow-xs">
          <div className="mb-4">
            <h3 className="font-bold text-slate-900 text-base">Top 10 Most Difficult Questions</h3>
            <p className="text-slate-500 text-xs mt-0.5">Questions ranked by lowest average score (scaled to 10.0)</p>
          </div>
          <div className="h-72 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={difficultQuestions} layout="vertical" margin={{ top: 10, right: 10, left: -10, bottom: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" horizontal={false} />
                <XAxis type="number" stroke="#94a3b8" style={{ fontSize: 10 }} domain={[0, 10]} tickLine={false} />
                <YAxis type="category" dataKey="question" stroke="#94a3b8" style={{ fontSize: 10 }} tickLine={false} />
                <ChartTooltip
                  content={({ active, payload }) => {
                    if (active && payload && payload.length) {
                      const data = payload[0].payload;
                      return (
                        <div className="bg-white border border-slate-200 p-3 rounded-xl shadow-lg text-xs space-y-1">
                          <p className="font-bold text-slate-900">{data.question}</p>
                          <p className="text-red-600 flex justify-between gap-4 font-semibold">
                            Avg Score: <span className="font-mono">{data.avgScore} / 10</span>
                          </p>
                          <p className="text-indigo-600 flex justify-between gap-4 font-semibold">
                            Correct rate: <span className="font-mono">{data.percentCorrect}%</span>
                          </p>
                          <p className="text-slate-400 text-[10px] mt-1">Total Attempts: {data.attempts}</p>
                        </div>
                      );
                    }
                    return null;
                  }}
                />
                <Bar dataKey="avgScore" name="Avg Score" fill="#f43f5e" radius={[0, 4, 4, 0]} maxBarSize={24}>
                  {difficultQuestions.map((_, index) => (
                    <Cell key={`cell-${index}`} fill={index === 0 ? '#b91c1c' : index < 3 ? '#e11d48' : '#f43f5e'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Chart 4: Question Score Boxplot Visuals */}
        <div className="bg-white border border-slate-200 rounded-2xl p-5 shadow-xs flex flex-col justify-between">
          <div className="mb-4">
            <h3 className="font-bold text-slate-900 text-base">Question Score Distributions</h3>
            <p className="text-slate-500 text-xs mt-0.5">Average scores of student submissions grouped per question item (scaled to 10.0)</p>
          </div>
          <div className="space-y-4 max-h-72 overflow-y-auto pr-1">
            {questionMetricsSummary.map(q => {
              const scorePct = `${q.avgScore * 10}%`;
              return (
                <div key={q.question} className="grid grid-cols-12 items-center gap-3">
                  <span className="col-span-2 text-xs font-bold text-slate-700">{q.question}</span>
                  <div className="col-span-8 relative h-6 bg-slate-50 border border-slate-150 rounded-md overflow-hidden shadow-3xs">
                    {/* Tick line */}
                    <div className="absolute inset-x-0 inset-y-0 flex justify-between pointer-events-none opacity-40">
                      {[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10].map(t => (
                        <div key={t} className="w-px h-full bg-slate-200" />
                      ))}
                    </div>
                    {/* Fill bar */}
                    <div
                      className={`absolute left-0 top-0 bottom-0 transition-all rounded-r-sm ${
                        q.avgScore >= 8.5
                          ? 'bg-emerald-500'
                          : q.avgScore >= 6.5
                          ? 'bg-blue-500'
                          : q.avgScore >= 4.5
                          ? 'bg-amber-500'
                          : 'bg-red-500'
                      }`}
                      style={{ width: scorePct }}
                    />
                    <span className="absolute right-2 top-1/2 -translate-y-1/2 text-[9px] font-mono font-black text-slate-600 mix-blend-difference bg-white/70 px-1 rounded-sm">
                      {q.avgScore}
                    </span>
                  </div>
                  <div className="col-span-2 text-right text-[10px] text-slate-400 font-mono">
                    {q.attempts} att
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* Heatmap Section */}
      <div className="bg-white border border-slate-200 rounded-2xl p-5 shadow-xs">
        <div className="mb-4">
          <h3 className="font-bold text-slate-900 text-base">Potential Response Tree (PRT) Pass Rate Heatmap</h3>
          <p className="text-slate-500 text-xs mt-0.5">Cell percentages represent the passing rate of individual STACK question PRT blocks</p>
        </div>

        {heatmapData.questions.length > 0 && heatmapData.prts.length > 0 ? (
          <div className="overflow-x-auto border border-slate-150 rounded-xl">
            <table className="w-full text-center border-collapse text-xs">
              <thead>
                <tr className="bg-slate-50 border-b border-slate-200 text-slate-500 font-bold">
                  <th className="p-3 text-left">Question</th>
                  {heatmapData.prts.map(p => (
                    <th key={p} className="p-3 uppercase">{p}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {heatmapData.questions.map(q => (
                  <tr key={q} className="border-b border-slate-100 hover:bg-slate-50/50 transition">
                    <td className="p-3 font-bold text-slate-800 text-left bg-slate-50/30 border-r border-slate-100 w-32 shrink-0">{q}</td>
                    {heatmapData.prts.map(p => {
                      const match = prtPassRates.find(item => item.question === q && item.prtName === p);
                      const rate = match?.passRate;

                      return (
                        <td key={p} className="p-2 border-r border-slate-100 font-mono">
                          <div
                            className={`py-2 px-1.5 rounded-lg text-[11px] transition shadow-4xs ${getHeatmapColor(rate)}`}
                            title={match ? `${q} -> ${p}: ${rate}% (${match.attempts} attempts)` : 'Not present in question'}
                          >
                            {rate !== undefined ? `${rate}%` : '-'}
                          </div>
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="p-8 text-center bg-slate-50 border border-dashed border-slate-150 rounded-xl text-slate-400 text-xs">
            No PRT level branches identified in the loaded response columns
          </div>
        )}
      </div>

      {/* Main Stats Summary Table */}
      <div className="bg-white border border-slate-200 rounded-2xl p-5 shadow-xs">
        <div className="mb-4">
          <h3 className="font-bold text-slate-900 text-base">Question Analysis Metrics Summary</h3>
          <p className="text-slate-500 text-xs mt-0.5">Detailed indicators grouped at the individual question level</p>
        </div>

        <div className="overflow-x-auto border border-slate-150 rounded-xl">
          <table className="w-full text-left border-collapse text-xs">
            <thead>
              <tr className="bg-slate-50 border-b border-slate-200 text-slate-500 font-bold">
                {[
                  { field: 'question', label: 'Question' },
                  { field: 'attempts', label: 'Attempts' },
                  { field: 'avgScore', label: 'Avg Score (scaled)' },
                  { field: 'percentCorrect', label: 'Correct %' },
                  { field: 'percentIncorrect', label: 'Incorrect %' },
                  { field: 'percentValid', label: 'Valid %' },
                  { field: 'percentInvalid', label: 'Invalid %' },
                  { field: 'syntaxErrorCount', label: 'Syntax Errors' }
                ].map(col => (
                  <th
                    key={col.field}
                    onClick={() => handleSort(col.field as any)}
                    className="p-3 cursor-pointer hover:bg-slate-100 transition"
                  >
                    <span className="flex items-center gap-1">
                      {col.label}
                      <ArrowUpDown className="w-3.5 h-3.5 text-slate-400 shrink-0" />
                    </span>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {sortedQuestionMetrics.map((row, index) => (
                <tr key={index} className="border-b border-slate-100 hover:bg-slate-50 transition">
                  <td className="p-3 font-bold text-slate-900">{row.question}</td>
                  <td className="p-3 font-mono text-slate-600">{row.attempts}</td>
                  <td className="p-3 font-mono font-bold text-indigo-600 bg-indigo-50/20">{row.avgScore} / 10</td>
                  <td className="p-3 font-mono text-emerald-600 font-semibold">{row.percentCorrect}%</td>
                  <td className="p-3 font-mono text-red-500">{row.percentIncorrect}%</td>
                  <td className="p-3 font-mono text-blue-600">{row.percentValid}%</td>
                  <td className="p-3 font-mono text-amber-500">{row.percentInvalid}%</td>
                  <td className="p-3 font-mono text-slate-500">
                    <span className={`px-1.5 py-0.5 rounded-md font-semibold text-[10px] ${
                      row.syntaxErrorCount > 0 ? 'bg-amber-100 text-amber-700 font-black' : 'bg-slate-100 text-slate-400'
                    }`}>
                      {row.syntaxErrorCount} ({row.syntaxErrorPercent}%)
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Side-by-Side Tables: Difficulty & Repeated Wrongs */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Difficult Rank table */}
        <div className="bg-white border border-slate-200 rounded-2xl p-5 shadow-xs">
          <div className="mb-4">
            <h3 className="font-bold text-slate-900 text-base">Ranked Difficulty Analysis</h3>
            <p className="text-slate-500 text-xs mt-0.5">Questions needing pedagogical remediation (ordered by lowest average score)</p>
          </div>

          <div className="overflow-x-auto border border-slate-150 rounded-xl">
            <table className="w-full text-left border-collapse text-xs">
              <thead>
                <tr className="bg-slate-50 border-b border-slate-200 text-slate-500 font-bold">
                  <th className="p-3">Rank</th>
                  <th className="p-3">Question</th>
                  <th className="p-3">Avg Score</th>
                  <th className="p-3">Correct Rate</th>
                  <th className="p-3 text-right">Attempts</th>
                </tr>
              </thead>
              <tbody>
                {difficultQuestions.map((q, idx) => (
                  <tr key={idx} className="border-b border-slate-100 hover:bg-slate-50 transition">
                    <td className="p-3 font-bold font-mono">
                      <span className={`w-5 h-5 rounded-full inline-flex items-center justify-center text-[10px] text-white shadow-3xs ${
                        idx === 0 ? 'bg-red-600' : idx < 3 ? 'bg-orange-500' : 'bg-slate-400'
                      }`}>
                        {idx + 1}
                      </span>
                    </td>
                    <td className="p-3 font-bold text-slate-800">{q.question}</td>
                    <td className="p-3 font-mono text-red-600 font-bold">{q.avgScore} / 10</td>
                    <td className="p-3 font-mono text-slate-600">{q.percentCorrect}%</td>
                    <td className="p-3 font-mono text-slate-400 text-right">{q.attempts}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Repeated Wrongs table */}
        <div className="bg-white border border-slate-200 rounded-2xl p-5 shadow-xs">
          <div className="mb-4">
            <h3 className="font-bold text-slate-900 text-base">Repeated Wrong Answer Trends</h3>
            <p className="text-slate-500 text-xs mt-0.5">Identifies common cognitive misconceptions and frequent syntax errors</p>
          </div>

          <div className="overflow-x-auto border border-slate-150 rounded-xl">
            <table className="w-full text-left border-collapse text-xs">
              <thead>
                <tr className="bg-slate-50 border-b border-slate-200 text-slate-500 font-bold">
                  <th className="p-3">Question</th>
                  <th className="p-3">Repeated Fail Count</th>
                  <th className="p-3">Most Frequent Wrong Answer</th>
                </tr>
              </thead>
              <tbody>
                {repeatedWrongSummary.map((row, index) => (
                  <tr key={index} className="border-b border-slate-100 hover:bg-slate-50 transition">
                    <td className="p-3 font-bold text-slate-800">{row.question}</td>
                    <td className="p-3 font-mono">
                      <span className={`px-2 py-0.5 rounded-full text-[10px] font-bold font-mono ${
                        row.totalRepeatedWrongCount > 5
                          ? 'bg-rose-100 text-rose-700 shadow-4xs'
                          : row.totalRepeatedWrongCount > 0
                          ? 'bg-amber-100 text-amber-700 shadow-4xs'
                          : 'bg-slate-100 text-slate-400'
                      }`}>
                        {row.totalRepeatedWrongCount}
                      </span>
                    </td>
                    <td className="p-3 text-slate-600 font-mono text-[10px] italic whitespace-pre-wrap break-words max-w-sm" title={row.mostFrequentWrongAnswer}>
                      {row.mostFrequentWrongAnswer}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}
