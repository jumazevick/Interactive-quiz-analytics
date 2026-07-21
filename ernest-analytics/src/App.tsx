import React, { useState, useRef } from 'react';
import {
  FileText,
  Upload,
  BarChart4,
  TrendingUp,
  Users,
  Search,
  Calendar,
  AlertTriangle,
  Clock,
  Info,
  HelpCircle,
  Sparkles,
  Filter,
  CheckCircle2,
  GitCommit,
  Grid,
  Home
} from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import * as XLSX from 'xlsx';
import Papa from 'papaparse';

import { Attempt, ResponseLevelRow } from './types';
import {
  processRawRow,
  computeQuizStats,
  computeBoxPlotStats,
  computeKDE,
  calculateCorrelation,
  formatDuration
} from './utils/dataProcessor';
import { buildResponseLevelTable } from './utils/questionProcessor';
import QuestionAnalysisSection from './components/QuestionAnalysisSection';
import MoodleExportGuide from './components/MoodleExportGuide';

import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as ChartTooltip,
  Legend,
  ResponsiveContainer,
  LineChart,
  Line,
  AreaChart,
  Area
} from 'recharts';

export default function App() {
  const [activeTab, setActiveTab] = useState<'home' | 'dashboard' | 'questions'>('home');
  const [attempts, setAttempts] = useState<Attempt[]>([]);
  const [responseRows, setResponseRows] = useState<ResponseLevelRow[]>([]);
  const [uploadedFiles, setUploadedFiles] = useState<{ name: string; size: number; quizID: number }[]>([]);
  const [selectedQuizzes, setSelectedQuizzes] = useState<number[]>([]);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  
  // Search query for merged table
  const [searchQuery, setSearchQuery] = useState('');
  
  // Dashboard Section Visibilities
  const [showMergedList, setShowMergedList] = useState(true);
  const [showSummaryStats, setShowSummaryStats] = useState(true);
  const [showBoxPlot, setShowBoxPlot] = useState(true);
  const [showEngagement, setShowEngagement] = useState(true);
  const [showScatter, setShowScatter] = useState(true);
  const [showMetricsLine, setShowMetricsLine] = useState(true);

  // Scatter plot parameters
  const [scatterGradeType, setScatterGradeType] = useState<'Highest Grade' | 'Average Grade' | 'Minimum Grade'>('Highest Grade');

  // Selected statistics to display in table/line chart
  const [selectedStatsKeys, setSelectedStatsKeys] = useState<string[]>([
    'student_count',
    'attempt_rate',
    'mean_grade',
    'grade_variance',
    'mean_highest_grade',
    'attempt_count'
  ]);

  // Pagination for merged table
  const [currentPage, setCurrentPage] = useState(1);
  const itemsPerPage = 10;

  // Drag and drop state
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Parse files
  const handleFiles = (files: FileList) => {
    setErrorMsg(null);
    const loadedAttempts: Attempt[] = [];
    const loadedResponseRows: ResponseLevelRow[] = [];
    let processedCount = 0;
    const fileArray = Array.from(files);

    if (fileArray.length === 0) return;

    fileArray.forEach((file, index) => {
      const nextQuizID = uploadedFiles.length + index + 1;
      const fileMeta = { name: file.name, size: file.size, quizID: nextQuizID };

      const parseCallback = (rawData: any[]) => {
        try {
          if (rawData.length === 0) {
            throw new Error(`File "${file.name}" is empty`);
          }

          const headers = Object.keys(rawData[0]);
          
          // Verify critical columns
          const hasState = headers.some(h => /state/i.test(h));
          const hasStartedOn = headers.some(h => /started\s*on/i.test(h));
          const hasCompleted = headers.some(h => /completed/i.test(h));
          const hasGrade = headers.some(h => /grade/i.test(h));

          if (!hasState || !hasStartedOn || !hasCompleted || !hasGrade) {
            throw new Error(
              `Required columns missing. Ensure columns match: "State", "Started on", "Completed", "Grade/10.00". Found headers: ${headers.slice(0, 5).join(', ')}...`
            );
          }

          const formatted = rawData
            .map((row, rowIdx) => processRawRow(row, headers, nextQuizID, rowIdx, file.name))
            // Only keep "Finished" state as in Python load_data
            .filter(a => a.state.trim().toLowerCase() === 'finished');

          const responseFormatted = buildResponseLevelTable(rawData, headers, nextQuizID);

          loadedAttempts.push(...formatted);
          loadedResponseRows.push(...responseFormatted);
          setUploadedFiles(prev => [...prev, fileMeta]);
          processedCount++;

          if (processedCount === fileArray.length) {
            setAttempts(prev => {
              const updated = [...prev, ...loadedAttempts];
              // Reset quiz selections to include all loaded
              const uniqueQuizIDs = Array.from(new Set(updated.map(a => a.quizID)));
              setSelectedQuizzes(uniqueQuizIDs);
              return updated;
            });
            setResponseRows(prev => [...prev, ...loadedResponseRows]);
          }
        } catch (err: any) {
          setErrorMsg(err.message || 'Error parsing file columns');
        }
      };

      if (file.name.endsWith('.csv')) {
        Papa.parse(file, {
          header: true,
          skipEmptyLines: true,
          complete: (results) => {
            parseCallback(results.data);
          },
          error: (err) => {
            setErrorMsg(`CSV read error on ${file.name}: ${err.message}`);
          }
        });
      } else if (file.name.endsWith('.xls') || file.name.endsWith('.xlsx')) {
        const reader = new FileReader();
        reader.onload = (e) => {
          try {
            const data = new Uint8Array(e.target?.result as ArrayBuffer);
            const workbook = XLSX.read(data, { type: 'array' });
            const sheetName = workbook.SheetNames[0];
            const worksheet = workbook.Sheets[sheetName];
            const jsonData = XLSX.utils.sheet_to_json(worksheet);
            parseCallback(jsonData);
          } catch (err: any) {
            setErrorMsg(`Excel read error on ${file.name}: ${err.message}`);
          }
        };
        reader.readAsArrayBuffer(file);
      } else {
        setErrorMsg(`Unsupported file format: ${file.name}. Only .csv, .xls, .xlsx are supported.`);
      }
    });
  };

  const onDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const onDragLeave = () => {
    setIsDragging(false);
  };

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFiles(e.dataTransfer.files);
    }
  };

  const triggerFileInput = () => {
    fileInputRef.current?.click();
  };

  const resetAllData = () => {
    setAttempts([]);
    setResponseRows([]);
    setUploadedFiles([]);
    setSelectedQuizzes([]);
    setErrorMsg(null);
    setSearchQuery('');
  };

  // Filter logic
  const filteredAttempts = attempts.filter(a => {
    const matchesQuiz = selectedQuizzes.includes(a.quizID);
    const fullName = `${a.firstname} ${a.surname}`.toLowerCase();
    const email = a.email.toLowerCase();
    const matchesSearch = searchQuery === '' || 
      fullName.includes(searchQuery.toLowerCase()) || 
      email.includes(searchQuery.toLowerCase()) ||
      a.student_id.toLowerCase().includes(searchQuery.toLowerCase());
    return matchesQuiz && matchesSearch;
  });

  // Unique Quiz IDs loaded
  const allLoadedQuizzes = Array.from(new Set(attempts.map(a => a.quizID))).sort((a, b) => a - b);

  // Compute Statistics
  const summaryStats = computeQuizStats(filteredAttempts, selectedStatsKeys);

  // Compute Box-Plot metrics
  const boxPlotStats = computeBoxPlotStats(filteredAttempts);

  // Compute KDE timeline data
  const kdeData = computeKDE(filteredAttempts, selectedQuizzes);

  // Compute Scatter data (attempts count vs grade metrics per student/quiz)
  interface StudentGroup {
    quizID: number;
    student_id: string;
    studentName: string;
    email: string;
    attemptsCount: number;
    gradesList: number[];
  }

  const studentQuizGroups: Record<string, StudentGroup> = {};
  filteredAttempts.forEach(a => {
    const key = `${a.quizID}_${a.student_id}`;
    if (!studentQuizGroups[key]) {
      studentQuizGroups[key] = {
        quizID: a.quizID,
        student_id: a.student_id,
        studentName: `${a.firstname} ${a.surname}`.trim() || 'Anonymized Student',
        email: a.email,
        attemptsCount: 0,
        gradesList: []
      };
    }
    studentQuizGroups[key].attemptsCount += 1;
    studentQuizGroups[key].gradesList.push(a.grade);
  });

  const scatterPoints = Object.values(studentQuizGroups).map(group => {
    let gradeVal = 0;
    if (scatterGradeType === 'Highest Grade') {
      gradeVal = Math.max(...group.gradesList);
    } else if (scatterGradeType === 'Minimum Grade') {
      gradeVal = Math.min(...group.gradesList);
    } else {
      // Average Grade
      gradeVal = group.gradesList.reduce((x, y) => x + y, 0) / group.gradesList.length;
    }

    return {
      quizID: group.quizID,
      quizLabel: `Quiz ${group.quizID}`,
      student_id: group.student_id,
      studentName: group.studentName,
      attempts: group.attemptsCount,
      grade: parseFloat(gradeVal.toFixed(2))
    };
  });

  const scatterXValues = scatterPoints.map(p => p.attempts);
  const scatterYValues = scatterPoints.map(p => p.grade);
  const pearsonCorrelation = calculateCorrelation(scatterXValues, scatterYValues);

  // Highlight metrics
  const totalAttempts = filteredAttempts.length;
  const uniqueStudentsCount = Array.from(new Set(filteredAttempts.map(a => a.student_id))).length;
  const overallMeanGrade = totalAttempts > 0 
    ? parseFloat((filteredAttempts.reduce((s, a) => s + a.grade, 0) / totalAttempts).toFixed(2)) 
    : 0;

  // Multi-select quiz checkbox toggler
  const toggleQuizSelection = (qid: number) => {
    if (selectedQuizzes.includes(qid)) {
      setSelectedQuizzes(prev => prev.filter(x => x !== qid));
    } else {
      setSelectedQuizzes(prev => [...prev, qid].sort((a, b) => a - b));
    }
  };

  const selectAllQuizzes = () => {
    setSelectedQuizzes(allLoadedQuizzes);
  };

  const selectNoQuizzes = () => {
    setSelectedQuizzes([]);
  };

  const toggleStatKey = (key: string) => {
    if (selectedStatsKeys.includes(key)) {
      setSelectedStatsKeys(prev => prev.filter(x => x !== key));
    } else {
      setSelectedStatsKeys(prev => [...prev, key]);
    }
  };

  // Render visual stats table headers mapping
  const statLabels: Record<string, string> = {
    student_count: 'Student Count',
    attempt_rate: 'Avg Attempts / Student',
    mean_grade: 'Mean Grade',
    grade_variance: 'Grade Variance',
    mean_highest_grade: 'Mean Highest Grade',
    attempt_count: 'Total Attempt Count'
  };

  // Paginated merged table rows
  const startIndex = (currentPage - 1) * itemsPerPage;
  const paginatedAttempts = filteredAttempts.slice(startIndex, startIndex + itemsPerPage);
  const totalPages = Math.ceil(filteredAttempts.length / itemsPerPage);

  // Color mapper for Quiz IDs
  const getQuizColor = (quizID: number) => {
    const colors = [
      '#3b82f6', // blue
      '#10b981', // emerald
      '#f59e0b', // amber
      '#ef4444', // red
      '#8b5cf6', // violet
      '#ec4899', // pink
      '#06b6d4', // cyan
      '#14b8a6'  // teal
    ];
    return colors[(quizID - 1) % colors.length];
  };

  return (
    <div className="min-h-screen bg-[#f8fafc] text-slate-800 flex flex-col">
      {/* Top Banner / Header */}
      <header className="sticky top-0 z-40 bg-white border-b border-slate-200 shadow-xs px-6 py-4">
        <div className="max-w-7xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="bg-gradient-to-tr from-blue-600 to-indigo-600 p-2.5 rounded-xl text-white shadow-md shadow-blue-100">
              <BarChart4 className="w-6 h-6" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-slate-900 tracking-tight flex items-center gap-2">
                Moodle STACK Analytics Hub
                <span className="text-xs bg-indigo-50 text-indigo-600 border border-indigo-100 font-semibold px-2 py-0.5 rounded-full">v1.1</span>
              </h1>
              <p className="text-xs text-slate-500">Streamlining Data Analysis for Moodle STACK Quiz Attempts</p>
            </div>
          </div>

          <div className="flex items-center bg-slate-100 p-1 rounded-xl">
            <button
              onClick={() => setActiveTab('home')}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                activeTab === 'home'
                  ? 'bg-white text-slate-950 shadow-xs'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              <Home className="w-4 h-4" />
              Home
            </button>
            <button
              onClick={() => setActiveTab('dashboard')}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                activeTab === 'dashboard'
                  ? 'bg-white text-slate-950 shadow-xs'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              <TrendingUp className="w-4 h-4" />
              Quiz Analysis
            </button>
            <button
              onClick={() => setActiveTab('questions')}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                activeTab === 'questions'
                  ? 'bg-white text-slate-950 shadow-xs'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              <Grid className="w-4 h-4" />
              Question & PRT Analysis
            </button>
          </div>
        </div>
      </header>

      {/* Main Content Body */}
      <main className="flex-1 w-full max-w-7xl mx-auto px-4 py-6 md:px-6">
        <AnimatePresence mode="wait">
          {activeTab === 'home' && (
            /* Tab: Home */
            <motion.div
              key="home"
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -15 }}
              transition={{ duration: 0.2 }}
              className="space-y-6"
            >
              <MoodleExportGuide />
            </motion.div>
          )}

          {activeTab === 'questions' && (
            /* Tab: Question & PRT Analysis */
            <motion.div
              key="questions"
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -15 }}
              transition={{ duration: 0.2 }}
            >
              <QuestionAnalysisSection
                responseRows={responseRows}
                selectedQuizzes={selectedQuizzes}
                uploadedFiles={uploadedFiles}
              />
            </motion.div>
          )}

          {activeTab === 'dashboard' && (
            /* Tab: Dashboard & Analysis */
            attempts.length === 0 ? (
              <motion.div
                key="dashboard-empty"
                initial={{ opacity: 0, y: 15 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -15 }}
                transition={{ duration: 0.2 }}
                className="max-w-2xl mx-auto space-y-6"
              >
                {/* Drag and Drop Card on Top */}
                <div className="bg-white border border-slate-200 rounded-2xl p-6 shadow-xs">
                  <h3 className="font-bold text-slate-900 text-sm mb-3 flex items-center justify-center gap-2">
                    <Upload className="w-4 h-4 text-blue-600" />
                    Upload Quiz Files
                  </h3>

                  <div
                    onDragOver={onDragOver}
                    onDragLeave={onDragLeave}
                    onDrop={onDrop}
                    onClick={triggerFileInput}
                    className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all flex flex-col items-center justify-center ${
                      isDragging
                        ? 'border-blue-500 bg-blue-50/50 scale-[0.98]'
                        : 'border-slate-200 hover:border-slate-300 hover:bg-slate-50'
                    }`}
                  >
                    <input
                      type="file"
                      ref={fileInputRef}
                      onChange={(e) => e.target.files && handleFiles(e.target.files)}
                      multiple
                      accept=".csv, .xls, .xlsx"
                      className="hidden"
                    />
                    <div className="bg-blue-50 p-2.5 rounded-full text-blue-600 mb-2">
                      <FileText className="w-5 h-5" />
                    </div>
                    <p className="text-xs font-semibold text-slate-700">Drag & drop files here</p>
                    <p className="text-[10px] text-slate-400 mt-1">Supports CSV, XLS, or XLSX</p>
                    <button className="mt-3 text-xs bg-white border border-slate-200 hover:border-slate-300 text-slate-700 font-semibold px-3.5 py-1.5 rounded-lg shadow-2xs transition">
                      Select Files
                    </button>
                  </div>

                  {errorMsg && (
                    <div className="mt-3 bg-red-50 border border-red-200 rounded-xl p-3 text-[11px] text-red-800 flex gap-1.5 items-start">
                      <AlertTriangle className="w-4 h-4 text-red-600 shrink-0 mt-0.5" />
                      <div>
                        <span className="font-semibold">Error: </span>
                        {errorMsg}
                      </div>
                    </div>
                  )}
                </div>

                {/* Info Card below it */}
                <div className="bg-white border border-slate-200 rounded-2xl p-8 text-center shadow-xs flex flex-col items-center justify-center space-y-4">
                  <div className="bg-blue-50 p-4 rounded-full text-blue-600 shadow-inner">
                    <BarChart4 className="w-8 h-8" />
                  </div>
                  <div className="max-w-md space-y-2">
                    <h3 className="text-base font-bold text-slate-900">No Quiz Data Loaded</h3>
                    <p className="text-slate-500 text-xs leading-relaxed">
                      To view interactive graphs, time series, and correlational statistics, please upload your Moodle Quiz export files above using drag & drop or clicking "Select Files".
                    </p>
                  </div>
                  <div className="flex justify-center pt-2">
                    <button
                      onClick={() => setActiveTab('home')}
                      className="bg-white border border-slate-200 hover:border-slate-300 text-slate-700 font-semibold text-xs px-5 py-2.5 rounded-xl shadow-3xs transition"
                    >
                      Go to Home & Guide
                    </button>
                  </div>
                </div>
              </motion.div>
            ) : (
              <motion.div
                key="dashboard-active"
                initial={{ opacity: 0, y: 15 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -15 }}
                transition={{ duration: 0.2 }}
                className="grid lg:grid-cols-4 gap-6"
              >
                {/* Sidebar Panel - Controls & File Upload */}
                <div className="lg:col-span-1 space-y-6">
                  {/* File Uploader widget */}
                  <div className="bg-white border border-slate-200 rounded-2xl p-5 shadow-xs">
                    <h3 className="font-bold text-slate-900 text-sm mb-3 flex items-center gap-2">
                      <Upload className="w-4 h-4 text-blue-600" />
                      Upload Quiz Files
                    </h3>

                    <div
                      onDragOver={onDragOver}
                      onDragLeave={onDragLeave}
                      onDrop={onDrop}
                      onClick={triggerFileInput}
                      className={`border-2 border-dashed rounded-xl p-5 text-center cursor-pointer transition-all flex flex-col items-center justify-center ${
                        isDragging
                          ? 'border-blue-500 bg-blue-50/50 scale-[0.98]'
                          : 'border-slate-200 hover:border-slate-300 hover:bg-slate-50'
                      }`}
                    >
                      <input
                        type="file"
                        ref={fileInputRef}
                        onChange={(e) => e.target.files && handleFiles(e.target.files)}
                        multiple
                        accept=".csv, .xls, .xlsx"
                        className="hidden"
                      />
                      <div className="bg-blue-50 p-2.5 rounded-full text-blue-600 mb-2">
                        <FileText className="w-5 h-5" />
                      </div>
                      <p className="text-xs font-semibold text-slate-700">Drag & drop files here</p>
                      <p className="text-[10px] text-slate-400 mt-1">Supports CSV, XLS, or XLSX</p>
                      <button className="mt-3 text-xs bg-white border border-slate-200 hover:border-slate-300 text-slate-700 font-medium px-3 py-1.5 rounded-lg shadow-2xs transition">
                        Select Files
                      </button>
                    </div>

                    {errorMsg && (
                      <div className="mt-3 bg-red-50 border border-red-200 rounded-xl p-3 text-[11px] text-red-800 flex gap-1.5 items-start">
                        <AlertTriangle className="w-4 h-4 text-red-600 shrink-0 mt-0.5" />
                        <div>
                          <span className="font-semibold">Error: </span>
                          {errorMsg}
                        </div>
                      </div>
                    )}

                    {/* List of uploaded files */}
                    {uploadedFiles.length > 0 && (
                      <div className="mt-4 pt-4 border-t border-slate-100 space-y-2">
                        <div className="flex items-center justify-between">
                          <span className="text-[11px] font-bold text-slate-400 uppercase tracking-wider">Loaded Files ({uploadedFiles.length})</span>
                          <button
                            onClick={resetAllData}
                            className="text-[10px] text-red-600 hover:text-red-800 hover:underline font-semibold"
                          >
                            Clear All
                          </button>
                        </div>
                        <div className="max-h-36 overflow-y-auto space-y-1.5 pr-1">
                          {uploadedFiles.map((file, i) => (
                            <div key={i} className="bg-slate-50 border border-slate-150 rounded-lg px-2.5 py-1.5 flex items-center justify-between text-xs">
                              <div className="min-w-0 flex items-center gap-1.5">
                                <span
                                  className="w-1.5 h-3 rounded-full shrink-0"
                                  style={{ backgroundColor: getQuizColor(file.quizID) }}
                                />
                                <p className="font-medium text-slate-700 truncate max-w-[120px]" title={file.name}>
                                  {file.name}
                                </p>
                              </div>
                              <span className="text-[9px] bg-white border border-slate-200 text-slate-500 font-bold px-1.5 py-0.5 rounded-sm shrink-0">
                                ID {file.quizID}
                              </span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Filter and View togglers */}
                  {attempts.length > 0 && (
                    <motion.div
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      className="space-y-6"
                    >
                      {/* Quiz Selector */}
                      <div className="bg-white border border-slate-200 rounded-2xl p-5 shadow-xs">
                        <div className="flex items-center justify-between mb-3">
                          <h3 className="font-bold text-slate-900 text-sm flex items-center gap-1.5">
                            <Filter className="w-4 h-4 text-blue-600" />
                            Filter Quiz IDs
                          </h3>
                          <div className="flex gap-2">
                            <button
                              onClick={selectAllQuizzes}
                              className="text-[10px] text-blue-600 hover:underline font-medium"
                            >
                              All
                            </button>
                            <span className="text-slate-300 text-[10px]">|</span>
                            <button
                              onClick={selectNoQuizzes}
                              className="text-[10px] text-slate-600 hover:underline font-medium"
                            >
                              None
                            </button>
                          </div>
                        </div>

                        <div className="space-y-1 max-h-44 overflow-y-auto pr-1">
                          {allLoadedQuizzes.map((qid) => {
                            const isChecked = selectedQuizzes.includes(qid);
                            const color = getQuizColor(qid);
                            return (
                              <label
                                key={qid}
                                className="flex items-center justify-between p-1.5 hover:bg-slate-50 rounded-lg cursor-pointer transition"
                              >
                                <div className="flex items-center gap-2">
                                  <input
                                    type="checkbox"
                                    checked={isChecked}
                                    onChange={() => toggleQuizSelection(qid)}
                                    className="rounded text-blue-600 border-slate-300 focus:ring-blue-500 w-3.5 h-3.5"
                                  />
                                  <span className="text-xs font-medium text-slate-700">Quiz {qid}</span>
                                </div>
                                <span
                                  className="w-2.5 h-2.5 rounded-full border border-white shadow-xs"
                                  style={{ backgroundColor: color }}
                                />
                              </label>
                            );
                          })}
                        </div>
                      </div>

                      {/* View Options Toggle */}
                      <div className="bg-white border border-slate-200 rounded-2xl p-5 shadow-xs">
                        <h3 className="font-bold text-slate-900 text-sm mb-3 flex items-center gap-1.5">
                          <Grid className="w-4 h-4 text-blue-600" />
                          Display Sections
                        </h3>

                        <div className="space-y-2">
                          <label className="flex items-center gap-2.5 p-1 hover:bg-slate-50 rounded-lg cursor-pointer transition">
                            <input
                              type="checkbox"
                              checked={showSummaryStats}
                              onChange={(e) => setShowSummaryStats(e.target.checked)}
                              className="rounded text-blue-600 border-slate-300 focus:ring-blue-500 w-3.5 h-3.5"
                            />
                            <span className="text-xs font-medium text-slate-700">Summary Statistics</span>
                          </label>

                          <label className="flex items-center gap-2.5 p-1 hover:bg-slate-50 rounded-lg cursor-pointer transition">
                            <input
                              type="checkbox"
                              checked={showBoxPlot}
                              onChange={(e) => setShowBoxPlot(e.target.checked)}
                              className="rounded text-blue-600 border-slate-300 focus:ring-blue-500 w-3.5 h-3.5"
                            />
                            <span className="text-xs font-medium text-slate-700">Grade Box Plot</span>
                          </label>

                          <label className="flex items-center gap-2.5 p-1 hover:bg-slate-50 rounded-lg cursor-pointer transition">
                            <input
                              type="checkbox"
                              checked={showEngagement}
                              onChange={(e) => setShowEngagement(e.target.checked)}
                              className="rounded text-blue-600 border-slate-300 focus:ring-blue-500 w-3.5 h-3.5"
                            />
                            <span className="text-xs font-medium text-slate-700">Engagement Over Time</span>
                          </label>

                          <label className="flex items-center gap-2.5 p-1 hover:bg-slate-50 rounded-lg cursor-pointer transition">
                            <input
                              type="checkbox"
                              checked={showScatter}
                              onChange={(e) => setShowScatter(e.target.checked)}
                              className="rounded text-blue-600 border-slate-300 focus:ring-blue-500 w-3.5 h-3.5"
                            />
                            <span className="text-xs font-medium text-slate-700">Attempts vs Grades</span>
                          </label>

                          <label className="flex items-center gap-2.5 p-1 hover:bg-slate-50 rounded-lg cursor-pointer transition">
                            <input
                              type="checkbox"
                              checked={showMetricsLine}
                              onChange={(e) => setShowMetricsLine(e.target.checked)}
                              className="rounded text-blue-600 border-slate-300 focus:ring-blue-500 w-3.5 h-3.5"
                            />
                            <span className="text-xs font-medium text-slate-700">Metrics Line Graph</span>
                          </label>

                          <label className="flex items-center gap-2.5 p-1 hover:bg-slate-50 rounded-lg cursor-pointer transition">
                            <input
                              type="checkbox"
                              checked={showMergedList}
                              onChange={(e) => setShowMergedList(e.target.checked)}
                              className="rounded text-blue-600 border-slate-300 focus:ring-blue-500 w-3.5 h-3.5"
                            />
                            <span className="text-xs font-medium text-slate-700">Attempts Raw Table</span>
                          </label>
                        </div>
                      </div>
                    </motion.div>
                  )}
                </div>

                {/* Main Workspace (Graphs and Tables) */}
                <div className="lg:col-span-3 space-y-6">
                  {/* Active Dashboard state */}
                  <div className="space-y-6">
                    {/* Aggregated Insight Cards */}
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                      <div className="bg-white border border-slate-200 rounded-2xl p-4 shadow-3xs">
                        <div className="flex items-center justify-between text-slate-400 mb-1">
                          <span className="text-[11px] font-bold uppercase tracking-wider">Total Quizzes</span>
                          <GitCommit className="w-4 h-4 text-blue-500" />
                        </div>
                        <p className="text-2xl font-black text-slate-900">{allLoadedQuizzes.length}</p>
                        <p className="text-[10px] text-slate-400 mt-1">Unique files parsed</p>
                      </div>

                      <div className="bg-white border border-slate-200 rounded-2xl p-4 shadow-3xs">
                        <div className="flex items-center justify-between text-slate-400 mb-1">
                          <span className="text-[11px] font-bold uppercase tracking-wider">Finished Attempts</span>
                          <CheckCircle2 className="w-4 h-4 text-emerald-500" />
                        </div>
                        <p className="text-2xl font-black text-slate-900">{filteredAttempts.length}</p>
                        <p className="text-[10px] text-slate-400 mt-1">
                          {totalAttempts !== filteredAttempts.length ? `${totalAttempts} total attempts` : 'All filtered'}
                        </p>
                      </div>

                      <div className="bg-white border border-slate-200 rounded-2xl p-4 shadow-3xs">
                        <div className="flex items-center justify-between text-slate-400 mb-1">
                          <span className="text-[11px] font-bold uppercase tracking-wider">Students Count</span>
                          <Users className="w-4 h-4 text-violet-500" />
                        </div>
                        <p className="text-2xl font-black text-slate-900">{uniqueStudentsCount}</p>
                        <p className="text-[10px] text-slate-400 mt-1">Unique student IDs</p>
                      </div>

                      <div className="bg-white border border-slate-200 rounded-2xl p-4 shadow-3xs">
                        <div className="flex items-center justify-between text-slate-400 mb-1">
                          <span className="text-[11px] font-bold uppercase tracking-wider">Mean Score / 10</span>
                          <Sparkles className="w-4 h-4 text-amber-500" />
                        </div>
                        <p className="text-2xl font-black text-slate-900">{overallMeanGrade}</p>
                        <p className="text-[10px] text-slate-400 mt-1">Across all filtered records</p>
                      </div>
                    </div>

                    {/* Filtered Alert / Notice */}
                    {selectedQuizzes.length === 0 && (
                      <div className="bg-amber-50 border border-amber-200 rounded-xl p-4 text-amber-950 text-xs flex gap-2.5 items-center">
                        <AlertTriangle className="w-5 h-5 text-amber-600 shrink-0" />
                        <p>
                          <strong>No Quizzes Selected:</strong> Select at least one quiz checkbox in the left-hand sidebar to view active analytics calculations and graphs!
                        </p>
                      </div>
                    )}

                    {/* Section 1: Summary of Quiz Statistics */}
                    {showSummaryStats && selectedQuizzes.length > 0 && (
                      <div className="bg-white border border-slate-200 rounded-2xl p-5 shadow-xs">
                        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 mb-4">
                          <div>
                            <h3 className="font-bold text-slate-900 text-base">Summary of Quiz Statistics</h3>
                            <p className="text-slate-500 text-xs mt-0.5">Statistical distributions aggregated per loaded quiz ID</p>
                          </div>

                          {/* Column picker in the card */}
                          <div className="flex flex-wrap gap-1.5 bg-slate-50 border border-slate-200 p-1 rounded-xl">
                            {['student_count', 'attempt_count', 'attempt_rate', 'mean_grade', 'grade_variance', 'mean_highest_grade'].map((key) => {
                              const isChecked = selectedStatsKeys.includes(key);
                              return (
                                <button
                                  key={key}
                                  onClick={() => toggleStatKey(key)}
                                  className={`text-[10px] font-medium px-2 py-1 rounded-md border transition-all ${
                                    isChecked
                                      ? 'bg-white text-blue-600 border-slate-200 shadow-2xs font-semibold'
                                      : 'text-slate-400 border-transparent hover:text-slate-700'
                                  }`}
                                >
                                  {key.replace('_', ' ')}
                                </button>
                              );
                            })}
                          </div>
                        </div>

                        {/* List of explanatory notes */}
                        <div className="bg-slate-50 border border-slate-200 rounded-xl p-3 mb-4 grid grid-cols-2 md:grid-cols-3 gap-2.5 text-[11px] text-slate-500">
                          {selectedStatsKeys.includes('student_count') && (
                            <div>
                              <strong className="text-slate-700">Student Count:</strong> Unique participants per quiz.
                            </div>
                          )}
                          {selectedStatsKeys.includes('attempt_count') && (
                            <div>
                              <strong className="text-slate-700">Total Attempts:</strong> Aggregate number of completed logs.
                            </div>
                          )}
                          {selectedStatsKeys.includes('attempt_rate') && (
                            <div>
                              <strong className="text-slate-700">Attempt Rate:</strong> Average attempts per active student.
                            </div>
                          )}
                          {selectedStatsKeys.includes('mean_grade') && (
                            <div>
                              <strong className="text-slate-700">Mean Grade:</strong> Average score normalized to 10.
                            </div>
                          )}
                          {selectedStatsKeys.includes('grade_variance') && (
                            <div>
                              <strong className="text-slate-700">Grade Variance:</strong> Dispersion scale of student scores.
                            </div>
                          )}
                          {selectedStatsKeys.includes('mean_highest_grade') && (
                            <div>
                              <strong className="text-slate-700">Mean Highest Grade:</strong> Average maximum student score.
                            </div>
                          )}
                        </div>

                        {/* Summary table */}
                        <div className="overflow-x-auto border border-slate-150 rounded-xl">
                          <table className="w-full text-left border-collapse text-xs">
                            <thead>
                              <tr className="bg-slate-50 border-b border-slate-200 text-slate-500 font-bold">
                                <th className="p-3">Quiz ID</th>
                                {selectedStatsKeys.includes('student_count') && <th className="p-3">{statLabels.student_count}</th>}
                                {selectedStatsKeys.includes('attempt_count') && <th className="p-3">{statLabels.attempt_count}</th>}
                                {selectedStatsKeys.includes('attempt_rate') && <th className="p-3">{statLabels.attempt_rate}</th>}
                                {selectedStatsKeys.includes('mean_grade') && <th className="p-3">{statLabels.mean_grade}</th>}
                                {selectedStatsKeys.includes('grade_variance') && <th className="p-3">{statLabels.grade_variance}</th>}
                                {selectedStatsKeys.includes('mean_highest_grade') && <th className="p-3">{statLabels.mean_highest_grade}</th>}
                              </tr>
                            </thead>
                            <tbody>
                              {summaryStats.map((row) => (
                                <tr key={row.quizID} className="border-b border-slate-100 hover:bg-slate-50 transition">
                                  <td className="p-3 font-semibold text-slate-900 flex items-center gap-2">
                                    <span
                                      className="w-2.5 h-2.5 rounded-full"
                                      style={{ backgroundColor: getQuizColor(row.quizID) }}
                                    />
                                    Quiz {row.quizID}
                                  </td>
                                  {selectedStatsKeys.includes('student_count') && <td className="p-3 text-slate-600 font-mono">{row.student_count ?? '-'}</td>}
                                  {selectedStatsKeys.includes('attempt_count') && <td className="p-3 text-slate-600 font-mono">{row.attempt_count ?? '-'}</td>}
                                  {selectedStatsKeys.includes('attempt_rate') && <td className="p-3 text-slate-600 font-mono">{row.attempt_rate ?? '-'}</td>}
                                  {selectedStatsKeys.includes('mean_grade') && (
                                    <td className="p-3 text-slate-800 font-semibold font-mono">
                                      {row.mean_grade !== undefined ? `${row.mean_grade} / 10` : '-'}
                                    </td>
                                  )}
                                  {selectedStatsKeys.includes('grade_variance') && <td className="p-3 text-slate-600 font-mono">{row.grade_variance ?? '-'}</td>}
                                  {selectedStatsKeys.includes('mean_highest_grade') && (
                                    <td className="p-3 text-slate-800 font-semibold font-mono">
                                      {row.mean_highest_grade !== undefined ? `${row.mean_highest_grade} / 10` : '-'}
                                    </td>
                                  )}
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      </div>
                    )}

                    {/* Section 2: Quiz Grade Distribution (Box Plot + Jitter Overlay) */}
                    {showBoxPlot && selectedQuizzes.length > 0 && (
                      <div className="bg-white border border-slate-200 rounded-2xl p-5 shadow-xs">
                        <div className="mb-4">
                          <h3 className="font-bold text-slate-900 text-base">Quiz Grade Distribution (Box plot)</h3>
                          <p className="text-slate-500 text-xs mt-0.5">Spread of normalized student scores (0–10) with individual score overlays (strip plot)</p>
                        </div>

                        {/* Interactive Explanation Legend */}
                        <div className="flex flex-wrap items-center gap-5 bg-slate-50 border border-slate-200 p-3 rounded-xl text-[11px] mb-6 text-slate-500">
                          <div className="flex items-center gap-1.5">
                            <div className="w-4 h-2 bg-indigo-100 border border-indigo-400 rounded-sm" />
                            <span>Box: Interquartile Range (Q1 to Q3)</span>
                          </div>
                          <div className="flex items-center gap-1.5">
                            <div className="w-0.5 h-3 bg-indigo-600" />
                            <span>Median Line (Q2 / 50th percentile)</span>
                          </div>
                          <div className="flex items-center gap-1.5">
                            <div className="w-2.5 h-2.5 bg-red-500 rounded-full" />
                            <span>Red Circle: Mean Grade</span>
                          </div>
                          <div className="flex items-center gap-1.5">
                            <div className="w-1.5 h-1.5 bg-slate-400 rounded-full opacity-60" />
                            <span>Scatter Dots: Individual student scores</span>
                          </div>
                        </div>

                        {/* HORIZONTAL BOX PLOTS GRID */}
                        <div className="space-y-6">
                          {boxPlotStats.map((stats) => {
                            if (!selectedQuizzes.includes(stats.quizID)) return null;

                            const quizColor = getQuizColor(stats.quizID);
                            
                            // Map values to percentages (0 - 10 translates to 0% - 100%)
                            const toPct = (val: number) => `${Math.min(100, Math.max(0, val * 10))}%`;

                            return (
                              <div key={stats.quizID} className="grid grid-cols-12 items-center gap-4 border-b border-slate-100 pb-5 last:border-0 last:pb-0">
                                {/* Quiz details label */}
                                <div className="col-span-12 md:col-span-2">
                                  <div className="flex items-center gap-2">
                                    <span className="w-2.5 h-2.5 rounded-full shrink-0" style={{ backgroundColor: quizColor }} />
                                    <h4 className="font-bold text-slate-800 text-sm">Quiz {stats.quizID}</h4>
                                  </div>
                                  <div className="text-[10px] text-slate-400 mt-1 space-y-0.5">
                                    <p>Mean: <span className="font-semibold text-slate-700 font-mono">{stats.mean}</span></p>
                                    <p>Median: <span className="font-semibold text-slate-700 font-mono">{stats.median}</span></p>
                                  </div>
                                </div>

                                {/* Box Plot track container */}
                                <div className="col-span-12 md:col-span-10">
                                  <div className="relative h-14 bg-slate-50 border border-slate-150 rounded-lg flex items-center px-4">
                                    {/* 0 to 10 scale ticks behind the plot */}
                                    <div className="absolute inset-x-4 inset-y-0 flex justify-between pointer-events-none">
                                      {[0,1,2,3,4,5,6,7,8,9,10].map((t) => (
                                        <div key={t} className="flex flex-col justify-end items-center h-full pb-0.5">
                                          <div className="w-0.5 h-1.5 bg-slate-200" />
                                          <span className="text-[8px] text-slate-300 font-mono">{t}</span>
                                        </div>
                                      ))}
                                    </div>

                                    {/* Box plot axis/track line */}
                                    <div className="absolute inset-x-4 h-0.5 bg-slate-200 pointer-events-none" />

                                    {/* Whisker thin line from Min to Max */}
                                    <div
                                      className="absolute h-0.5 bg-indigo-300 pointer-events-none"
                                      style={{
                                        left: `calc(1rem + ${toPct(stats.min)} - ${stats.min / 10 * 2}rem)`,
                                        right: `calc(1rem + ${toPct(10 - stats.max)} - ${(10 - stats.max) / 10 * 2}rem)`
                                      }}
                                    />

                                    {/* Whisker left-end cap */}
                                    <div
                                      className="absolute w-0.5 h-3 bg-indigo-400 pointer-events-none"
                                      style={{ left: `calc(1rem + ${toPct(stats.min)} - ${stats.min / 10 * 2}rem)` }}
                                    />

                                    {/* Whisker right-end cap */}
                                    <div
                                      className="absolute w-0.5 h-3 bg-indigo-400 pointer-events-none"
                                      style={{ left: `calc(1rem + ${toPct(stats.max)} - ${stats.max / 10 * 2}rem)` }}
                                    />

                                    {/* Box block representing Q1 to Q3 */}
                                    <div
                                      className="absolute h-6 bg-indigo-100/75 border border-indigo-300 rounded-sm pointer-events-none shadow-2xs"
                                      style={{
                                        left: `calc(1rem + ${toPct(stats.q1)} - ${stats.q1 / 10 * 2}rem)`,
                                        width: `calc(${toPct(stats.q3 - stats.q1)})`
                                      }}
                                    />

                                    {/* Median Line (Q2) */}
                                    <div
                                      className="absolute w-1.5 h-8 bg-indigo-600 rounded-xs pointer-events-none shadow-xs z-10"
                                      style={{ left: `calc(1rem + ${toPct(stats.median)} - ${stats.median / 10 * 2}rem)` }}
                                      title={`Median: ${stats.median}`}
                                    />

                                    {/* Mean Dot Indicator */}
                                    <div
                                      className="absolute w-3 h-3 bg-red-500 rounded-full border border-white shadow-xs pointer-events-none z-10"
                                      style={{ left: `calc(1rem + ${toPct(stats.mean)} - ${stats.mean / 10 * 2}rem)` }}
                                      title={`Mean: ${stats.mean}`}
                                    />

                                    {/* Strip Plot Points (Jitter dots overlay) */}
                                    <div className="absolute inset-x-4 inset-y-0 overflow-hidden pointer-events-none">
                                      {stats.points.map((pt, ptIdx) => {
                                        // Vertical jitter centering offset: 50% + random jitter %
                                        const topOffset = `${50 + pt.jitter * 100}%`;
                                        return (
                                          <div
                                            key={ptIdx}
                                            className="absolute w-1.5 h-1.5 rounded-full border border-white pointer-events-auto transition hover:scale-150 hover:bg-slate-900 shadow-3xs"
                                            style={{
                                              left: `calc(${toPct(pt.grade)} - ${pt.grade / 10 * 0.375}rem)`,
                                              top: topOffset,
                                              backgroundColor: quizColor,
                                              opacity: 0.55
                                            }}
                                            title={`Grade: ${pt.grade}`}
                                          />
                                        );
                                      })}
                                    </div>
                                  </div>
                                </div>
                              </div>
                            );
                          })}
                        </div>
                      </div>
                    )}

                    {/* Section 3: Frequency Density (Engagement Over Time) */}
                    {showEngagement && selectedQuizzes.length > 0 && (
                      <div className="bg-white border border-slate-200 rounded-2xl p-5 shadow-xs">
                        <div className="mb-4">
                          <h3 className="font-bold text-slate-900 text-base">Engagement Over Time</h3>
                          <p className="text-slate-500 text-xs mt-0.5">Smooth probability curve (Gaussian Kernel Density Estimation) indicating quiz start activity peaks</p>
                        </div>

                        {kdeData.length > 0 ? (
                          <div className="h-72 w-full">
                            <ResponsiveContainer width="100%" height="100%">
                              <AreaChart data={kdeData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                                <defs>
                                  {selectedQuizzes.map((qid) => (
                                    <linearGradient key={qid} id={`grad_${qid}`} x1="0" y1="0" x2="0" y2="1">
                                      <stop offset="5%" stopColor={getQuizColor(qid)} stopOpacity={0.2} />
                                      <stop offset="95%" stopColor={getQuizColor(qid)} stopOpacity={0} />
                                    </linearGradient>
                                  ))}
                                </defs>
                                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
                                <XAxis
                                  dataKey="dateLabel"
                                  tickLine={false}
                                  axisLine={false}
                                  stroke="#94a3b8"
                                  style={{ fontSize: 10, fontFamily: 'var(--font-sans)' }}
                                />
                                <YAxis
                                  tickLine={false}
                                  axisLine={false}
                                  stroke="#94a3b8"
                                  style={{ fontSize: 10, fontFamily: 'var(--font-sans)' }}
                                  label={{ value: 'Frequency Density', angle: -90, position: 'insideLeft', offset: 0, style: { fontSize: 10, fill: '#64748b' } }}
                                />
                                <ChartTooltip
                                  content={({ active, payload }) => {
                                    if (active && payload && payload.length) {
                                      return (
                                        <div className="bg-white/95 backdrop-blur-md border border-slate-200 p-3 rounded-xl shadow-lg text-xs space-y-1.5">
                                          <p className="font-bold text-slate-900">{payload[0].payload.dateLabel}</p>
                                          {payload.map((entry, idx) => (
                                            <div key={idx} className="flex items-center gap-2">
                                              <span className="w-2 h-2 rounded-full" style={{ backgroundColor: entry.color }} />
                                              <span className="text-slate-500">{entry.name}:</span>
                                              <span className="font-bold text-slate-800 font-mono">{entry.value}</span>
                                            </div>
                                          ))}
                                        </div>
                                      );
                                    }
                                    return null;
                                  }}
                                />
                                {selectedQuizzes.map((qid) => (
                                  <Area
                                    key={qid}
                                    type="monotone"
                                    dataKey={`Quiz ${qid}`}
                                    stroke={getQuizColor(qid)}
                                    fillOpacity={1}
                                    fill={`url(#grad_${qid})`}
                                    strokeWidth={2}
                                  />
                                ))}
                              </AreaChart>
                            </ResponsiveContainer>
                          </div>
                        ) : (
                          <div className="h-44 flex items-center justify-center bg-slate-50 border border-dashed border-slate-200 rounded-xl text-slate-400 text-xs gap-2">
                            <Calendar className="w-5 h-5 text-slate-300" />
                            No valid datetime values present in the loaded dataset
                          </div>
                        )}
                        <div className="mt-4 text-xs text-slate-400 leading-relaxed max-w-2xl bg-slate-50 border border-slate-150 p-3 rounded-xl">
                          <p className="font-medium text-slate-500 mb-1 flex items-center gap-1">
                            <Info className="w-3.5 h-3.5 text-blue-500 shrink-0" />
                            How to read this chart:
                          </p>
                          Peaks in each line represent date ranges where a higher frequency of attempts were initiated. Use this to track student study pacing, deadline cramming behaviors, and to compare engagement rhythms between distinct quiz models.
                        </div>
                      </div>
                    )}

                    {/* Section 4: Scatter Plot (Attempts vs Grades) */}
                    {showScatter && selectedQuizzes.length > 0 && (
                      <div className="bg-white border border-slate-200 rounded-2xl p-5 shadow-xs">
                        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-4">
                          <div>
                            <h3 className="font-bold text-slate-900 text-base">Scatter plot: Attempts vs Grades</h3>
                            <p className="text-slate-500 text-xs mt-0.5">Correlation mapping between a student's attempt rate and their corresponding score outcome</p>
                          </div>

                          {/* Grade metrics switch radio */}
                          <div className="flex items-center gap-1.5 bg-slate-50 border border-slate-250 p-1 rounded-xl">
                            {['Highest Grade', 'Average Grade', 'Minimum Grade'].map((type) => (
                              <button
                                key={type}
                                onClick={() => setScatterGradeType(type as any)}
                                className={`text-[10px] font-semibold px-3 py-1.5 rounded-lg transition-all ${
                                  scatterGradeType === type
                                    ? 'bg-white text-slate-950 shadow-2xs'
                                    : 'text-slate-500 hover:text-slate-900'
                                }`}
                              >
                                {type}
                              </button>
                            ))}
                          </div>
                        </div>

                        {/* Correlation output badge */}
                        <div className="bg-blue-50/70 border border-blue-150 text-blue-950 rounded-xl p-3.5 mb-4 text-xs flex flex-col md:flex-row items-start md:items-center justify-between gap-3">
                          <div className="flex gap-2.5 items-start">
                            <HelpCircle className="w-5 h-5 text-blue-600 shrink-0 mt-0.5" />
                            <div>
                              <p className="font-semibold text-blue-900">
                                Pearson Correlation: <span className="font-mono bg-blue-100 text-blue-800 px-1.5 py-0.5 rounded text-[13px] font-black">r = {pearsonCorrelation.toFixed(2)}</span>
                              </p>
                              <p className="text-slate-500 text-[10px] mt-0.5">
                                This coefficient measures the linear association between the count of attempts student took vs their performance grade.
                              </p>
                            </div>
                          </div>
                          <span className="shrink-0 text-[11px] font-bold bg-white text-slate-700 border border-slate-200 px-3 py-1 rounded-full">
                            {Math.abs(pearsonCorrelation) < 0.1 ? 'No clear correlation' : pearsonCorrelation > 0 ? 'Positive correlation (+)' : 'Negative correlation (-)'}
                          </span>
                        </div>

                        {/* Recharts Scatter chart */}
                        <div className="h-72 w-full">
                          <ResponsiveContainer width="100%" height="100%">
                            <ScatterChart margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
                              <XAxis
                                type="number"
                                dataKey="attempts"
                                name="No. of Attempts"
                                stroke="#94a3b8"
                                style={{ fontSize: 10, fontFamily: 'var(--font-sans)' }}
                                label={{ value: 'Number of Attempts', position: 'insideBottom', offset: -5, style: { fontSize: 10, fill: '#64748b' } }}
                              />
                              <YAxis
                                type="number"
                                dataKey="grade"
                                name="Grade"
                                domain={[0, 10]}
                                stroke="#94a3b8"
                                style={{ fontSize: 10, fontFamily: 'var(--font-sans)' }}
                                label={{ value: scatterGradeType, angle: -90, position: 'insideLeft', offset: 0, style: { fontSize: 10, fill: '#64748b' } }}
                              />
                              <ChartTooltip
                                content={({ active, payload }) => {
                                  if (active && payload && payload.length) {
                                    const data = payload[0].payload;
                                    return (
                                      <div className="bg-white border border-slate-200 p-3 rounded-xl shadow-lg text-xs space-y-1.5">
                                        <p className="font-bold text-slate-900">{data.studentName}</p>
                                        <div className="grid grid-cols-2 gap-x-3 gap-y-1 text-slate-500 text-[10px]">
                                          <div>Quiz ID:</div>
                                          <div className="font-semibold text-slate-800">{data.quizID}</div>
                                          <div>Attempts:</div>
                                          <div className="font-semibold text-slate-800 font-mono">{data.attempts}</div>
                                          <div>{scatterGradeType}:</div>
                                          <div className="font-semibold text-indigo-600 font-mono">{data.grade} / 10</div>
                                        </div>
                                      </div>
                                    );
                                  }
                                  return null;
                                }}
                              />
                              {selectedQuizzes.map((qid) => (
                                <Scatter
                                  key={qid}
                                  name={`Quiz ${qid}`}
                                  data={scatterPoints.filter(p => p.quizID === qid)}
                                  fill={getQuizColor(qid)}
                                  shape="circle"
                                />
                              ))}
                              <Legend verticalAlign="top" height={36} style={{ fontSize: 11 }} />
                            </ScatterChart>
                          </ResponsiveContainer>
                        </div>
                      </div>
                    )}

                    {/* Section 5: Line Graph of Various Metrics */}
                    {showMetricsLine && selectedQuizzes.length > 0 && (
                      <div className="bg-white border border-slate-200 rounded-2xl p-5 shadow-xs">
                        <div className="mb-4">
                          <h3 className="font-bold text-slate-900 text-base">Line Graph of Various Metrics</h3>
                          <p className="text-slate-500 text-xs mt-0.5">Plot trends in aggregated dimensions across the evaluated Quiz IDs</p>
                        </div>

                        {/* Metric lines visual comparison */}
                        <div className="h-72 w-full">
                          <ResponsiveContainer width="100%" height="100%">
                            <LineChart data={summaryStats} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
                              <XAxis
                                dataKey="quizID"
                                tickFormatter={(val) => `Quiz ${val}`}
                                stroke="#94a3b8"
                                style={{ fontSize: 10, fontFamily: 'var(--font-sans)' }}
                              />
                              <YAxis stroke="#94a3b8" style={{ fontSize: 10, fontFamily: 'var(--font-sans)' }} />
                              <ChartTooltip
                                content={({ active, payload }) => {
                                  if (active && payload && payload.length) {
                                    return (
                                      <div className="bg-white border border-slate-200 p-3 rounded-xl shadow-lg text-xs space-y-1.5">
                                        <p className="font-bold text-slate-900">Quiz {payload[0].payload.quizID}</p>
                                        {payload.map((entry, i) => (
                                          <div key={i} className="flex justify-between items-center gap-4 text-xs">
                                            <span className="flex items-center gap-1.5 text-slate-500">
                                              <span className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: entry.color }} />
                                              {entry.name}:
                                            </span>
                                            <span className="font-bold text-slate-800 font-mono">{entry.value}</span>
                                          </div>
                                        ))}
                                      </div>
                                    );
                                  }
                                  return null;
                                }}
                              />
                              <Legend style={{ fontSize: 11 }} />
                              {selectedStatsKeys.includes('student_count') && (
                                <Line type="monotone" dataKey="student_count" name="Student Count" stroke="#3b82f6" strokeWidth={2} activeDot={{ r: 6 }} />
                              )}
                              {selectedStatsKeys.includes('attempt_count') && (
                                <Line type="monotone" dataKey="attempt_count" name="Total Attempts" stroke="#8b5cf6" strokeWidth={2} />
                              )}
                              {selectedStatsKeys.includes('attempt_rate') && (
                                <Line type="monotone" dataKey="attempt_rate" name="Attempts / Student" stroke="#10b981" strokeWidth={2} />
                              )}
                              {selectedStatsKeys.includes('mean_grade') && (
                                <Line type="monotone" dataKey="mean_grade" name="Mean Grade" stroke="#f59e0b" strokeWidth={2} />
                              )}
                              {selectedStatsKeys.includes('grade_variance') && (
                                <Line type="monotone" dataKey="grade_variance" name="Grade Variance" stroke="#ef4444" strokeWidth={2} />
                              )}
                              {selectedStatsKeys.includes('mean_highest_grade') && (
                                <Line type="monotone" dataKey="mean_highest_grade" name="Mean Highest" stroke="#ec4899" strokeWidth={2} />
                              )}
                            </LineChart>
                          </ResponsiveContainer>
                        </div>
                      </div>
                    )}

                    {/* Section 6: Merged List of Users and Files (Attempts Table) */}
                    {showMergedList && (
                      <div className="bg-white border border-slate-200 rounded-2xl p-5 shadow-xs">
                        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-4">
                          <div>
                            <h3 className="font-bold text-slate-900 text-base">Merged List of Users and Files</h3>
                            <p className="text-slate-500 text-xs mt-0.5">
                              This table combines all uploaded quiz files into one searchable log view. Each row represents a finished attempt.
                            </p>
                          </div>

                          {/* Student Search box */}
                          <div className="relative w-full md:w-72 shrink-0">
                            <Search className="absolute left-3.5 top-1/2 -translate-y-1/2 text-slate-400 w-4 h-4" />
                            <input
                              type="text"
                              placeholder="Search by student name, ID, email..."
                              value={searchQuery}
                              onChange={(e) => {
                                setSearchQuery(e.target.value);
                                setCurrentPage(1); // Reset to page 1 on search
                              }}
                              className="w-full bg-slate-50 border border-slate-250 focus:border-blue-500 focus:bg-white rounded-xl pl-9.5 pr-4 py-1.5 text-xs text-slate-800 transition shadow-2xs"
                            />
                          </div>
                        </div>

                        {/* Records stats label */}
                        <div className="text-[11px] text-slate-400 mb-3 flex items-center justify-between">
                          <p>
                            Showing <span className="font-bold text-slate-700 font-mono">{filteredAttempts.length > 0 ? startIndex + 1 : 0}</span> to{' '}
                            <span className="font-bold text-slate-700 font-mono">{Math.min(filteredAttempts.length, startIndex + itemsPerPage)}</span> of{' '}
                            <span className="font-bold text-slate-700 font-mono">{filteredAttempts.length}</span> attempts
                          </p>
                        </div>

                        {/* Responsive attempts table */}
                        {filteredAttempts.length > 0 ? (
                          <div className="overflow-x-auto border border-slate-150 rounded-xl">
                            <table className="w-full text-left border-collapse text-xs">
                              <thead>
                                <tr className="bg-slate-50 border-b border-slate-200 text-slate-500 font-bold">
                                  <th className="p-3">Quiz</th>
                                  <th className="p-3">Student Name</th>
                                  <th className="p-3">Student ID</th>
                                  <th className="p-3">Email Address</th>
                                  <th className="p-3">Started On</th>
                                  <th className="p-3">Time Taken</th>
                                  <th className="p-3">Grade / 10</th>
                                </tr>
                              </thead>
                              <tbody>
                                {paginatedAttempts.map((attempt, index) => {
                                  const quizColor = getQuizColor(attempt.quizID);
                                  const studentName = `${attempt.firstname} ${attempt.surname}`.trim() || 'Anonymized Student';
                                  
                                  return (
                                    <tr key={index} className="border-b border-slate-100 hover:bg-slate-50 transition">
                                      <td className="p-3">
                                        <span
                                          className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-[10px] font-bold text-white shadow-3xs"
                                          style={{ backgroundColor: quizColor }}
                                        >
                                          Q{attempt.quizID}
                                        </span>
                                      </td>
                                      <td className="p-3 font-semibold text-slate-900">{studentName}</td>
                                      <td className="p-3 text-slate-500 font-mono">{attempt.student_id}</td>
                                      <td className="p-3 text-slate-500">{attempt.email || '-'}</td>
                                      <td className="p-3 text-slate-500">{attempt.start_date || '-'}</td>
                                      <td className="p-3 text-slate-500 font-mono flex items-center gap-1">
                                        <Clock className="w-3.5 h-3.5 text-slate-300" />
                                        {formatDuration(attempt.time_taken)}
                                      </td>
                                      <td className="p-3 font-bold text-slate-800 font-mono">
                                        <span className="bg-slate-50 border border-slate-150 text-slate-700 px-2 py-0.5 rounded shadow-3xs">
                                          {attempt.grade}
                                        </span>
                                      </td>
                                    </tr>
                                  );
                                })}
                              </tbody>
                            </table>
                          </div>
                        ) : (
                          <div className="p-10 text-center bg-slate-50 border border-dashed border-slate-200 rounded-xl text-slate-400 text-xs flex flex-col items-center justify-center gap-1">
                            <Search className="w-5 h-5 text-slate-300" />
                            No matches found for search query.
                          </div>
                        )}

                        {/* Pagination Footer controls */}
                        {totalPages > 1 && (
                          <div className="flex items-center justify-between pt-4">
                            <button
                              onClick={() => setCurrentPage(prev => Math.max(1, prev - 1))}
                              disabled={currentPage === 1}
                              className="text-xs bg-white border border-slate-200 hover:border-slate-300 disabled:opacity-45 disabled:hover:border-slate-200 font-medium px-3.5 py-1.5 rounded-lg text-slate-700 transition"
                            >
                              Previous
                            </button>
                            <span className="text-xs text-slate-400">
                              Page <strong className="text-slate-700 font-mono">{currentPage}</strong> of <strong className="text-slate-700 font-mono">{totalPages}</strong>
                            </span>
                            <button
                              onClick={() => setCurrentPage(prev => Math.min(totalPages, prev + 1))}
                              disabled={currentPage === totalPages}
                              className="text-xs bg-white border border-slate-200 hover:border-slate-300 disabled:opacity-45 disabled:hover:border-slate-200 font-medium px-3.5 py-1.5 rounded-lg text-slate-700 transition"
                            >
                              Next
                            </button>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              </motion.div>
            )
          )}
        </AnimatePresence>
      </main>
    </div>
  );
}
