export interface Attempt {
  surname: string;
  firstname: string;
  email: string;
  student_id: string;
  quizID: number;
  state: string;
  start_date: string; // Display string, e.g., "15 September 2024, 10:14 AM"
  end_date: string;   // Display string, e.g., "15 September 2024, 10:30 AM"
  time_taken: number | null; // seconds
  grade: number;
  fileName: string;
}

export interface QuizStats {
  quizID: number;
  student_count?: number;
  attempt_rate?: number;
  mean_grade?: number;
  grade_variance?: number;
  mean_highest_grade?: number;
  attempt_count?: number;
}

export interface ResponseLevelRow {
  student_id: string;
  studentName: string;
  email: string;
  quizID: number;
  question: string; // e.g. "Q1"
  grade: number;     // e.g. 1.00 (or scaled)
  maxGrade: number;  // e.g. 1.00
  responseStatus: 'correct' | 'incorrect' | 'syntax_error' | 'invalid' | 'blank';
  responseText: string;
  extractedPRTs: Record<string, { score: number; status: string }>; // prtName -> { score, status }
}

export interface PRTLevelRow {
  quizKey: string;
  quizID: number;
  student: string;
  student_id: string;
  question: string;
  prtName: string;
  prtScore: number | null;
  responseStatus: string;
  grade: number;
}

export interface QuestionMetricsSummary {
  question: string;
  attempts: number;
  avgScore: number;
  percentCorrect: number;
  percentIncorrect: number;
  percentValid: number;
  percentInvalid: number;
  syntaxErrorCount: number;
  syntaxErrorPercent: number;
}

export interface PRTPassRate {
  question: string;
  prtName: string;
  attempts: number;
  passRate: number;
}

export interface RepeatedAttemptsSummary {
  question: string;
  totalRepeatedWrongCount: number;
  mostFrequentWrongAnswer: string;
}
