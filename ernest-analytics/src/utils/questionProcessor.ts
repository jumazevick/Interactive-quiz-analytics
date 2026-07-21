import { ResponseLevelRow, PRTLevelRow, QuestionMetricsSummary, PRTPassRate, RepeatedAttemptsSummary } from '../types';

/**
 * Parses raw Moodle response rows into a clean, structured ResponseLevelRow array
 */
export function buildResponseLevelTable(
  rawData: any[],
  headers: string[],
  quizID: number
): ResponseLevelRow[] {
  if (!rawData || rawData.length === 0) return [];

  // Find all question numbers in the headers
  const questionNumbers = new Set<number>();
  headers.forEach(h => {
    let match = h.match(/^(?:Q|Question|q)\.?\s*(\d+)/i);
    if (match) {
      questionNumbers.add(parseInt(match[1], 10));
    }
    match = h.match(/^Response\s*(\d+)/i);
    if (match) {
      questionNumbers.add(parseInt(match[1], 10));
    }
  });

  const sortedQNums = Array.from(questionNumbers).sort((a, b) => a - b);

  // If no question columns detected, fallback to finding grade-like columns Q. 1, Q. 2 or Response columns
  if (sortedQNums.length === 0) {
    headers.forEach(h => {
      const match = h.match(/Q\.\s*(\d+)/i) || h.match(/Response\s*(\d+)/i);
      if (match) {
        sortedQNums.push(parseInt(match[1], 10));
      }
    });
  }

  const result: ResponseLevelRow[] = [];

  rawData.forEach((row, rowIndex) => {
    // Only include finished attempts as per general pipeline
    const stateVal = (row['State'] ?? row['state'] ?? '').toString().trim().toLowerCase();
    if (stateVal !== 'finished' && stateVal !== '') {
      // In some custom exports, State might be blank but valid, but let's stick to 'finished' if state exists
      if (row['State'] && stateVal !== 'finished') return;
    }

    // Extract student identifiers
    const surname = row['Surname'] ?? row['Last name'] ?? '';
    const firstname = row['First name'] ?? row['Firstname'] ?? '';
    const email = row['Email address'] ?? row['Email'] ?? '';
    
    let student_id = '';
    if (email && email.toString().trim() !== '') {
      student_id = email.toString().trim();
    } else if (row['anonymized_full_name'] && row['anonymized_full_name'].toString().trim() !== '') {
      student_id = row['anonymized_full_name'].toString().trim();
    } else {
      student_id = `student_${rowIndex}`;
    }

    const studentName = `${firstname} ${surname}`.trim() || 'Anonymized Student';

    sortedQNums.forEach(qNum => {
      const qKey = `Q${qNum}`;

      // 1. Find grade column for this question
      // Match: "Q1 /1.00", "Q. 1 /1.00", "Q1 grade", "q1:grade"
      const gradeCol = headers.find(h => {
        const hClean = h.trim();
        return (
          new RegExp(`^(?:Q|Question|q)\\.?\\s*0*${qNum}\\s*(?:\\/|$)`, 'i').test(hClean) ||
          new RegExp(`^(?:Q|Question|q)\\.?\\s*0*${qNum}_grade`, 'i').test(hClean) ||
          new RegExp(`^q0*${qNum}:grade`, 'i').test(hClean)
        );
      });

      // 2. Find Response text column
      // Match: "Response 1", "Q1 response", "q1_response", "q1:response"
      const responseCol = headers.find(h => {
        const hClean = h.trim();
        return (
          new RegExp(`^response\\s*0*${qNum}$`, 'i').test(hClean) ||
          new RegExp(`^(?:Q|Question|q)\\.?\\s*0*${qNum}\\s*response`, 'i').test(hClean) ||
          new RegExp(`^q0*${qNum}_response`, 'i').test(hClean) ||
          new RegExp(`^q0*${qNum}:response`, 'i').test(hClean)
        );
      });

      const responseText = responseCol ? (row[responseCol] ?? '').toString().trim() : '';

      // Parse responseText for PRTs and statuses
      const parsedPRTsFromText: Record<string, { score: number; status: string }> = {};
      let hasPRTsInText = false;
      let textPrtScoreSum = 0;
      let textPrtCount = 0;
      let textHasSyntaxError = false;
      let textHasInvalid = false;

      if (responseText) {
        // Split by semicolon
        const parts = responseText.split(';');
        parts.forEach((part: string) => {
          const match = part.match(/^\s*(prt\w+)\s*:\s*(.+)$/i);
          if (match) {
            const prtName = match[1].toLowerCase();
            const prtVal = match[2].trim();
            hasPRTsInText = true;
            
            let prtScore = 0;
            let prtStatus = 'incorrect';

            if (prtVal === '!') {
              textHasSyntaxError = true;
              prtStatus = 'invalid';
            } else {
              const scoreMatch = prtVal.match(/#\s*=\s*([\d.]+)/);
              if (scoreMatch) {
                prtScore = parseFloat(scoreMatch[1]);
                prtStatus = prtScore >= 0.99 ? 'correct' : (prtScore > 0 ? 'partially_correct' : 'incorrect');
                textPrtScoreSum += prtScore;
                textPrtCount++;
              } else {
                const lowerVal = prtVal.toLowerCase();
                if (lowerVal.includes('correct') || lowerVal.includes('true') || lowerVal.includes('pass')) {
                  prtScore = 1.0;
                  prtStatus = 'correct';
                  textPrtScoreSum += 1.0;
                  textPrtCount++;
                } else if (lowerVal.includes('incorrect') || lowerVal.includes('false') || lowerVal.includes('fail')) {
                  prtScore = 0.0;
                  prtStatus = 'incorrect';
                  textPrtCount++;
                }
              }
            }

            parsedPRTsFromText[prtName] = { score: prtScore, status: prtStatus };
          }

          if (part.toLowerCase().includes('[invalid]')) {
            textHasInvalid = true;
          }
        });
      }

      let grade = 0;
      let maxGrade = 1.00;

      if (gradeCol) {
        // Parse max grade from header, e.g. "Q1 /10.00" -> 10.00
        const maxMatch = gradeCol.match(/\/\s*(\d+(?:\.\d+)?)/);
        if (maxMatch) {
          maxGrade = parseFloat(maxMatch[1]);
        }
        const val = parseFloat(row[gradeCol]);
        if (!isNaN(val)) {
          grade = val;
        }
      } else if (hasPRTsInText && textPrtCount > 0) {
        grade = (textPrtScoreSum / textPrtCount) * maxGrade;
      }

      // 3. Find Question state/status column
      // Match: "Q1 state", "q1_state", "q1:state"
      const statusCol = headers.find(h => {
        const hClean = h.trim();
        return (
          new RegExp(`^(?:Q|Question|q)\\.?\\s*0*${qNum}\\s*state`, 'i').test(hClean) ||
          new RegExp(`^q0*${qNum}_state`, 'i').test(hClean) ||
          new RegExp(`^q0*${qNum}:state`, 'i').test(hClean)
        );
      });

      const rawStatusStr = statusCol ? (row[statusCol] ?? '').toString().trim().toLowerCase() : '';

      // Classify response status
      let responseStatus: 'correct' | 'incorrect' | 'syntax_error' | 'invalid' | 'blank' = 'incorrect';

      if (rawStatusStr.includes('syntax') || rawStatusStr.includes('error') || textHasSyntaxError) {
        responseStatus = 'syntax_error';
      } else if (rawStatusStr.includes('invalid') || textHasInvalid) {
        responseStatus = 'invalid';
      } else if (responseText === '') {
        responseStatus = 'blank';
      } else if (grade >= maxGrade * 0.9) {
        responseStatus = 'correct';
      } else {
        responseStatus = 'incorrect';
      }

      // 4. Extracted PRTs (Potential Response Trees)
      // Look for headers like: q1:prt1, q1:prt2, Q1: prt1, Q1_prt1
      const extractedPRTs: Record<string, { score: number; status: string }> = {};

      const prtCols = headers.filter(h => {
        const hClean = h.trim();
        return (
          new RegExp(`^q0*${qNum}:prt\\w+`, 'i').test(hClean) ||
          new RegExp(`^(?:Q|Question|q)\\.?\\s*0*${qNum}\\s*[:_]\\s*prt\\w+`, 'i').test(hClean)
        );
      });

      if (prtCols.length > 0) {
        prtCols.forEach(col => {
          // Extract prtName
          const nameMatch = col.match(/prt\w+/i);
          const prtName = nameMatch ? nameMatch[0].toLowerCase() : 'prt1';

          const prtValStr = (row[col] ?? '').toString().trim();
          let prtScore = 0;
          let prtStatus = 'incorrect';

          const parsedScore = parseFloat(prtValStr);
          if (!isNaN(parsedScore)) {
            prtScore = parsedScore;
            prtStatus = prtScore >= 0.5 ? 'correct' : 'incorrect';
          } else {
            // Textual status
            const valLower = prtValStr.toLowerCase();
            if (valLower === 'correct' || valLower === 'true' || valLower === 'pass') {
              prtScore = 1.0;
              prtStatus = 'correct';
            } else if (valLower === 'incorrect' || valLower === 'false' || valLower === 'fail') {
              prtScore = 0.0;
              prtStatus = 'incorrect';
            } else {
              prtScore = 0.0;
              prtStatus = 'unknown';
            }
          }

          extractedPRTs[prtName] = { score: prtScore, status: prtStatus };
        });
      } else if (hasPRTsInText) {
        Object.assign(extractedPRTs, parsedPRTsFromText);
      } else {
        // Fallback: Create one default PRT mapped to the question grade
        extractedPRTs['prt1'] = {
          score: maxGrade > 0 ? grade / maxGrade : 0,
          status: responseStatus === 'correct' ? 'correct' : 'incorrect'
        };
      }

      result.push({
        student_id,
        studentName,
        email,
        quizID,
        question: qKey,
        grade,
        maxGrade,
        responseStatus,
        responseText,
        extractedPRTs
      });
    });
  });

  return result;
}

/**
 * Explodes ResponseLevelRows to a tidy PRTLevelRow array where each row is (Student, Question, PRT)
 */
export function buildPRTLevelTable(responseLevelRows: ResponseLevelRow[]): PRTLevelRow[] {
  const result: PRTLevelRow[] = [];

  responseLevelRows.forEach(row => {
    const prtKeys = Object.keys(row.extractedPRTs);

    if (prtKeys.length === 0) {
      // Questions with no PRTs remain with null PRT values so attempt counts remain accurate
      result.push({
        quizKey: `Quiz ${row.quizID}`,
        quizID: row.quizID,
        student: row.studentName,
        student_id: row.student_id,
        question: row.question,
        prtName: '-',
        prtScore: null,
        responseStatus: row.responseStatus,
        grade: row.grade
      });
    } else {
      prtKeys.forEach(prtName => {
        const prtData = row.extractedPRTs[prtName];
        result.push({
          quizKey: `Quiz ${row.quizID}`,
          quizID: row.quizID,
          student: row.studentName,
          student_id: row.student_id,
          question: row.question,
          prtName,
          prtScore: prtData.score,
          responseStatus: prtData.status,
          grade: row.grade
        });
      });
    }
  });

  return result;
}

/**
 * Computes Question-Level Summary Metrics
 */
export function buildQuestionMetricsSummary(responseLevelRows: ResponseLevelRow[]): QuestionMetricsSummary[] {
  const grouped: Record<string, ResponseLevelRow[]> = {};
  responseLevelRows.forEach(row => {
    if (!grouped[row.question]) grouped[row.question] = [];
    grouped[row.question].push(row);
  });

  const sortedQuestions = Object.keys(grouped).sort((a, b) => {
    const numA = parseInt(a.replace(/\D/g, ''), 10) || 0;
    const numB = parseInt(b.replace(/\D/g, ''), 10) || 0;
    return numA - numB;
  });

  return sortedQuestions.map(q => {
    const qRows = grouped[q];
    const attempts = qRows.length;

    if (attempts === 0) {
      return {
        question: q,
        attempts: 0,
        avgScore: 0,
        percentCorrect: 0,
        percentIncorrect: 0,
        percentValid: 0,
        percentInvalid: 0,
        syntaxErrorCount: 0,
        syntaxErrorPercent: 0
      };
    }

    const totalScore = qRows.reduce((sum, r) => sum + (r.maxGrade > 0 ? (r.grade / r.maxGrade) * 10 : 0), 0);
    const avgScore = totalScore / attempts;

    const correctCount = qRows.filter(r => r.responseStatus === 'correct').length;
    const incorrectCount = qRows.filter(r => r.responseStatus === 'incorrect').length;
    const syntaxErrorCount = qRows.filter(r => r.responseStatus === 'syntax_error').length;
    const invalidCount = qRows.filter(r => r.responseStatus === 'invalid' || r.responseStatus === 'syntax_error').length;
    const blankCount = qRows.filter(r => r.responseStatus === 'blank').length;

    const percentCorrect = (correctCount / attempts) * 100;
    const percentIncorrect = (incorrectCount / attempts) * 100;
    const percentInvalid = (invalidCount / attempts) * 100;
    // Valid responses are correct + incorrect (anything not invalid, syntax error, or blank)
    const validCount = attempts - invalidCount - blankCount;
    const percentValid = (validCount / attempts) * 100;

    const syntaxErrorPercent = (syntaxErrorCount / attempts) * 100;

    return {
      question: q,
      attempts,
      avgScore: parseFloat(avgScore.toFixed(2)),
      percentCorrect: parseFloat(percentCorrect.toFixed(2)),
      percentIncorrect: parseFloat(percentIncorrect.toFixed(2)),
      percentValid: parseFloat(percentValid.toFixed(2)),
      percentInvalid: parseFloat(percentInvalid.toFixed(2)),
      syntaxErrorCount,
      syntaxErrorPercent: parseFloat(syntaxErrorPercent.toFixed(2))
    };
  });
}

/**
 * Computes PRT Pass Rates
 */
export function buildPRTPassRates(prtLevelRows: PRTLevelRow[]): PRTPassRate[] {
  const grouped: Record<string, PRTLevelRow[]> = {};
  prtLevelRows.forEach(row => {
    if (row.prtName === '-') return;
    const key = `${row.question}_${row.prtName}`;
    if (!grouped[key]) grouped[key] = [];
    grouped[key].push(row);
  });

  return Object.keys(grouped).sort().map(key => {
    const pRows = grouped[key];
    const attempts = pRows.length;
    if (attempts === 0) {
      return { question: key.split('_')[0], prtName: key.split('_')[1], attempts: 0, passRate: 0 };
    }

    // A PRT attempt passes if score is >= 0.5 (on normalized 0-1 scale) or status is 'correct'
    const passes = pRows.filter(r => r.prtScore !== null && (r.prtScore >= 0.5 || r.responseStatus === 'correct')).length;
    const passRate = (passes / attempts) * 100;

    const parts = key.split('_');
    return {
      question: parts[0],
      prtName: parts[1],
      attempts,
      passRate: parseFloat(passRate.toFixed(2))
    };
  });
}

/**
 * Ranks most difficult questions based on lowest average score and percent correct (attempts as tie-breaker)
 */
export function rankMostDifficultQuestions(
  questionMetricsSummary: QuestionMetricsSummary[],
  topN = 10
): QuestionMetricsSummary[] {
  return [...questionMetricsSummary]
    .sort((a, b) => {
      if (a.avgScore !== b.avgScore) {
        return a.avgScore - b.avgScore; // lowest first
      }
      if (a.percentCorrect !== b.percentCorrect) {
        return a.percentCorrect - b.percentCorrect; // lowest first
      }
      return b.attempts - a.attempts; // higher attempts as tie-breaker (more reliable bad trend)
    })
    .slice(0, topN);
}

/**
 * Computes repeated wrong answers per question
 */
export function buildRepeatedAttemptsSummary(responseLevelRows: ResponseLevelRow[]): RepeatedAttemptsSummary[] {
  const grouped: Record<string, ResponseLevelRow[]> = {};
  responseLevelRows.forEach(row => {
    if (!grouped[row.question]) grouped[row.question] = [];
    grouped[row.question].push(row);
  });

  const sortedQuestions = Object.keys(grouped).sort((a, b) => {
    const numA = parseInt(a.replace(/\D/g, ''), 10) || 0;
    const numB = parseInt(b.replace(/\D/g, ''), 10) || 0;
    return numA - numB;
  });

  return sortedQuestions.map(q => {
    const qRows = grouped[q];
    // Filter wrong answers (incorrect, invalid, or syntax error) with non-empty text
    const wrongAnswers = qRows
      .filter(r => (r.responseStatus === 'incorrect' || r.responseStatus === 'syntax_error' || r.responseStatus === 'invalid') && r.responseText !== '')
      .map(r => r.responseText);

    // Count frequencies
    const freq: Record<string, number> = {};
    wrongAnswers.forEach(ans => {
      freq[ans] = (freq[ans] || 0) + 1;
    });

    let totalRepeatedWrongCount = 0;
    let mostFrequentWrongAnswer = '-';
    let maxFreq = 0;

    Object.entries(freq).forEach(([ans, count]) => {
      if (count > 1) {
        // Any wrong answer submitted more than once is a repeat
        totalRepeatedWrongCount += count;
      }
      if (count > maxFreq) {
        maxFreq = count;
        mostFrequentWrongAnswer = ans;
      }
    });

    // Format most frequent answer with its frequency count
    if (maxFreq > 1) {
      mostFrequentWrongAnswer = `"${mostFrequentWrongAnswer}" (×${maxFreq})`;
    } else if (maxFreq === 1) {
      mostFrequentWrongAnswer = `"${mostFrequentWrongAnswer}" (×1)`;
    }

    return {
      question: q,
      totalRepeatedWrongCount,
      mostFrequentWrongAnswer
    };
  });
}
