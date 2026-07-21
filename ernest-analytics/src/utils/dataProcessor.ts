import { Attempt } from '../types';

/**
 * Regex-based helper to parse standard Moodle durations, e.g., "12 mins 5 secs"
 */
export function parseTime(timeStr: string | null | undefined): number | null {
  if (!timeStr) return null;
  const lowerStr = timeStr.toLowerCase().trim();

  // Look for Weeks, Days, Hours, Mins, Secs
  const weeksMatch = lowerStr.match(/(\d+)\s*week/);
  const daysMatch = lowerStr.match(/(\d+)\s*day/);
  const hoursMatch = lowerStr.match(/(\d+)\s*hour/);
  const minsMatch = lowerStr.match(/(\d+)\s*(min|m)/);
  const secsMatch = lowerStr.match(/(\d+)\s*(sec|s)/);

  let seconds = 0;
  let matchedAny = false;

  if (weeksMatch) {
    seconds += parseInt(weeksMatch[1], 10) * 7 * 24 * 60 * 60;
    matchedAny = true;
  }
  if (daysMatch) {
    seconds += parseInt(daysMatch[1], 10) * 24 * 60 * 60;
    matchedAny = true;
  }
  if (hoursMatch) {
    seconds += parseInt(hoursMatch[1], 10) * 60 * 60;
    matchedAny = true;
  }
  if (minsMatch) {
    seconds += parseInt(minsMatch[1], 10) * 60;
    matchedAny = true;
  }
  if (secsMatch) {
    seconds += parseInt(secsMatch[1], 10);
    matchedAny = true;
  }

  if (matchedAny) {
    return seconds;
  }

  // Fallback: is it already a raw number?
  const num = parseFloat(lowerStr);
  return isNaN(num) ? null : num;
}

/**
 * Parses dates of format: "20 September 2024 10:14 AM" or ISO strings
 */
export function parseDateTime(value: string | null | undefined): Date | null {
  if (!value) return null;
  const clean = value.replace(/,/g, '').trim();

  // Try standard browser parser
  const d = new Date(clean);
  if (!isNaN(d.getTime())) {
    return d;
  }

  // Custom parser for "20 September 2024 10:14 AM"
  const parts = clean.split(/\s+/);
  if (parts.length >= 4) {
    const day = parseInt(parts[0], 10);
    const monthStr = parts[1].toLowerCase();
    const year = parseInt(parts[2], 10);
    const timeStr = parts[3];
    const ampm = parts[4] ? parts[4].toUpperCase() : '';

    const months: Record<string, number> = {
      january: 0, feb: 1, february: 1, mar: 2, march: 2, apr: 3, april: 3,
      may: 4, jun: 5, june: 5, jul: 6, july: 6, aug: 7, august: 7,
      sep: 8, september: 8, oct: 9, october: 9, nov: 10, november: 10,
      dec: 11, december: 11, jan: 0, apr_: 3, jul_: 6, aug_: 7, sept: 8
    };

    const month = months[monthStr];
    if (month !== undefined && !isNaN(day) && !isNaN(year)) {
      const timeParts = timeStr.split(':');
      let hours = parseInt(timeParts[0] || '0', 10);
      const minutes = parseInt(timeParts[1] || '0', 10);

      if (ampm === 'PM' && hours < 12) hours += 12;
      if (ampm === 'AM' && hours === 12) hours = 0;

      return new Date(year, month, day, hours, minutes);
    }
  }

  return null;
}

/**
 * Normalizes grades based on "Grade/10.00" column name matching
 */
export function processRawRow(
  row: any,
  keys: string[],
  quizID: number,
  index: number,
  fileName: string
): Attempt {
  // Rename 'Last name' to 'Surname' if needed
  const surname = row['Surname'] ?? row['Last name'] ?? '';
  const firstname = row['First name'] ?? row['Firstname'] ?? '';
  const email = row['Email address'] ?? row['Email'] ?? '';
  const state = row['State'] ?? '';

  // Look for a grade column
  const gradeKey = keys.find(k => /^Grade\/\d+(\.\d+)?/i.test(k.trim()));
  let rawGrade = NaN;
  let maxGradeValue = 10;

  if (gradeKey) {
    const match = gradeKey.match(/Grade\/(\d+(\.\d+)?)/i);
    if (match) {
      maxGradeValue = parseFloat(match[1]);
    }
    rawGrade = parseFloat(row[gradeKey]);
  } else {
    // Fallbacks
    const fallbackGradeKey = keys.find(k => /^grade$/i.test(k.trim()));
    if (fallbackGradeKey) {
      rawGrade = parseFloat(row[fallbackGradeKey]);
    }
  }

  let grade = isNaN(rawGrade) ? 0 : rawGrade;
  if (maxGradeValue !== 10 && maxGradeValue > 0) {
    grade = (grade / maxGradeValue) * 10;
  }

  // student_id assignment matching Python logic
  let student_id = '';
  if (email && email.trim() !== '') {
    student_id = email;
  } else if (row['anonymized_full_name'] && row['anonymized_full_name'].trim() !== '') {
    student_id = row['anonymized_full_name'];
  } else {
    student_id = `row_${index}`;
  }

  const startedOnStr = row['Started on'] ?? '';
  const completedStr = row['Completed'] ?? '';
  const timeTakenStr = row['Time taken'] ?? '';

  return {
    surname,
    firstname,
    email,
    student_id,
    quizID,
    state,
    start_date: startedOnStr,
    end_date: completedStr,
    time_taken: parseTime(timeTakenStr),
    grade: parseFloat(grade.toFixed(2)),
    fileName
  };
}

/**
 * Calculates variance of an array of numbers
 */
export function calculateVariance(values: number[], mean: number): number {
  if (values.length <= 1) return 0;
  const sumOfSquares = values.reduce((acc, v) => acc + Math.pow(v - mean, 2), 0);
  return parseFloat((sumOfSquares / (values.length - 1)).toFixed(2));
}

/**
 * Computes custom stats on filtered attempts list
 */
export function computeQuizStats(attempts: Attempt[], selectedStats: string[]): any[] {
  // Group attempts by quizID
  const grouped: Record<number, Attempt[]> = {};
  attempts.forEach(a => {
    if (!grouped[a.quizID]) grouped[a.quizID] = [];
    grouped[a.quizID].push(a);
  });

  const quizIDs = Object.keys(grouped).map(Number).sort((a, b) => a - b);

  return quizIDs.map(qid => {
    const quizAttempts = grouped[qid];
    const stats: any = { quizID: qid };

    // Unique students
    const uniqueStudents = Array.from(new Set(quizAttempts.map(a => a.student_id)));

    if (selectedStats.includes('student_count')) {
      stats.student_count = uniqueStudents.length;
    }

    if (selectedStats.includes('attempt_count')) {
      stats.attempt_count = quizAttempts.length;
    }

    // Mean grade & grade variance
    const grades = quizAttempts.map(a => a.grade);
    const meanGrade = grades.length > 0 ? grades.reduce((a, b) => a + b, 0) / grades.length : 0;

    if (selectedStats.includes('mean_grade')) {
      stats.mean_grade = parseFloat(meanGrade.toFixed(2));
    }

    if (selectedStats.includes('grade_variance')) {
      stats.grade_variance = calculateVariance(grades, meanGrade);
    }

    // Average highest grade per student
    if (selectedStats.includes('mean_highest_grade')) {
      const highestGrades = uniqueStudents.map(sid => {
        const studentAttempts = quizAttempts.filter(a => a.student_id === sid);
        return Math.max(...studentAttempts.map(a => a.grade));
      });
      const meanHighest = highestGrades.length > 0 ? highestGrades.reduce((a, b) => a + b, 0) / highestGrades.length : 0;
      stats.mean_highest_grade = parseFloat(meanHighest.toFixed(2));
    }

    // Attempt rate: average attempts per student
    if (selectedStats.includes('attempt_rate')) {
      const rate = uniqueStudents.length > 0 ? quizAttempts.length / uniqueStudents.length : 0;
      stats.attempt_rate = parseFloat(rate.toFixed(2));
    }

    return stats;
  });
}

/**
 * Formats seconds into human-readable duration
 */
export function formatDuration(seconds: number | null): string {
  if (seconds === null) return '-';
  if (seconds < 60) return `${seconds}s`;
  const mins = Math.floor(seconds / 60);
  const secs = seconds % 60;
  if (mins < 60) return `${mins}m ${secs}s`;
  const hours = Math.floor(mins / 60);
  const remMins = mins % 60;
  return `${hours}h ${remMins}m`;
}

/**
 * Calculates Pearson Correlation Coefficient between two arrays
 */
export function calculateCorrelation(x: number[], y: number[]): number {
  const n = x.length;
  if (n <= 1) return 0;
  const meanX = x.reduce((a, b) => a + b, 0) / n;
  const meanY = y.reduce((a, b) => a + b, 0) / n;

  let num = 0;
  let denX = 0;
  let denY = 0;

  for (let i = 0; i < n; i++) {
    const diffX = x[i] - meanX;
    const diffY = y[i] - meanY;
    num += diffX * diffY;
    denX += diffX * diffX;
    denY += diffY * diffY;
  }

  if (denX === 0 || denY === 0) return 0;
  return num / Math.sqrt(denX * denY);
}

export interface BoxPlotData {
  quizID: number;
  min: number;
  q1: number;
  median: number;
  q3: number;
  max: number;
  mean: number;
  points: { grade: number; jitter: number }[];
}

/**
 * Computes box-plot statistics (min, Q1, median, Q3, max, mean) for each quiz
 */
export function computeBoxPlotStats(attempts: Attempt[]): BoxPlotData[] {
  const grouped: Record<number, Attempt[]> = {};
  attempts.forEach(a => {
    if (!grouped[a.quizID]) grouped[a.quizID] = [];
    grouped[a.quizID].push(a);
  });

  return Object.keys(grouped).map(Number).sort((a, b) => a - b).map(qid => {
    const list = grouped[qid].map(a => a.grade).sort((x, y) => x - y);
    const n = list.length;
    if (n === 0) {
      return { quizID: qid, min: 0, q1: 0, median: 0, q3: 0, max: 0, mean: 0, points: [] };
    }

    const min = list[0];
    const max = list[n - 1];
    const mean = list.reduce((s, v) => s + v, 0) / n;

    const getPercentile = (p: number) => {
      const idx = (n - 1) * p;
      const base = Math.floor(idx);
      const rest = idx - base;
      if (list[base + 1] !== undefined) {
        return list[base] + rest * (list[base + 1] - list[base]);
      }
      return list[base];
    };

    const q1 = getPercentile(0.25);
    const median = getPercentile(0.5);
    const q3 = getPercentile(0.75);

    // Points for jitter strip plot
    const points = list.map((val, idx) => {
      // Use a deterministic-ish jitter based on index to avoid flickering during renders
      const pseudoRandom = Math.sin(idx + qid) * 0.15;
      return {
        grade: val,
        jitter: pseudoRandom
      };
    });

    return {
      quizID: qid,
      min: parseFloat(min.toFixed(2)),
      q1: parseFloat(q1.toFixed(2)),
      median: parseFloat(median.toFixed(2)),
      q3: parseFloat(q3.toFixed(2)),
      max: parseFloat(max.toFixed(2)),
      mean: parseFloat(mean.toFixed(2)),
      points
    };
  });
}

/**
 * Computes Gaussian Kernel Density Estimation for started-on times
 */
export function computeKDE(
  attempts: Attempt[],
  quizIDs: number[]
): { dateLabel: string; dateNum: number; [key: string]: number | string }[] {
  const dataPoints = attempts
    .map(a => {
      const dt = parseDateTime(a.start_date);
      return { quizID: a.quizID, time: dt ? dt.getTime() : null };
    })
    .filter((p): p is { quizID: number; time: number } => p.time !== null && !isNaN(p.time));

  if (dataPoints.length === 0) return [];

  const times = dataPoints.map(p => p.time);
  const minTime = Math.min(...times);
  const maxTime = Math.max(...times);

  // Pad time bounds
  const range = maxTime - minTime;
  const padding = range === 0 ? 12 * 60 * 60 * 1000 : range * 0.08;
  const startTime = minTime - padding;
  const endTime = maxTime + padding;

  // Generate 50 steps
  const gridSteps = 40;
  const step = (endTime - startTime) / (gridSteps - 1);
  const grid: { dateLabel: string; dateNum: number; [key: string]: number | string }[] = [];

  // Bandwidth (Silverman's rule of thumb)
  const stdDev = (arr: number[]) => {
    const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
    return Math.sqrt(arr.reduce((acc, v) => acc + Math.pow(v - mean, 2), 0) / arr.length);
  };
  const sd = stdDev(times) || 12 * 60 * 60 * 1000;
  const h = 0.9 * sd * Math.pow(times.length, -0.2) || 1;

  const gaussianKernel = (z: number) => {
    return Math.exp(-0.5 * Math.pow(z, 2)) / Math.sqrt(2 * Math.PI);
  };

  for (let i = 0; i < gridSteps; i++) {
    const x = startTime + i * step;
    const dateObj = new Date(x);
    // Short localized display string
    const dateLabel = dateObj.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });

    const item: { dateLabel: string; dateNum: number; [key: string]: number | string } = {
      dateLabel,
      dateNum: x
    };

    quizIDs.forEach(qid => {
      const quizTimes = dataPoints.filter(p => p.quizID === qid).map(p => p.time);
      if (quizTimes.length === 0) {
        item[`Quiz ${qid}`] = 0;
        return;
      }
      let sum = 0;
      quizTimes.forEach(y => {
        sum += gaussianKernel((x - y) / h);
      });
      // Density, scaled to avoid sub-million decimals in chart labels
      const density = (sum / (quizTimes.length * h)) * 1e8; 
      item[`Quiz ${qid}`] = parseFloat(density.toFixed(4));
    });

    grid.push(item);
  }

  return grid;
}
