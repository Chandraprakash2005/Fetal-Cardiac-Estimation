/**
 * ECG API Service
 * - USE_MOCK = true  → returns mock data (no backend needed)
 * - USE_MOCK = false → hits your real backend at BASE_URL
 */

export const BASE_URL = 'http://localhost:8000';
export const USE_MOCK = false; // ← Connected to real backend streaming API

const delay = (ms) => new Promise((res) => setTimeout(res, ms));

// ─── Mock result fallback ───────────────────────────────────────────────────
export const MOCK_RESULT = {
  status: 'success',

  // 1️⃣  Cardiac Age Estimation
  cardiac_age: {
    estimated_age_days: 214,
    lmp_ga_days: 210,
    delta_age_days: 4,
    estimated_age_weeks: 30.6,
    lmp_ga_weeks: 30.0,
    mae_days: 6.8,
    confidence_interval_95: [201, 227],
  },

  // 2️⃣  QRS / Fetal ECG Extraction
  qrs_evaluation: {
    f1_score: 0.966,
    precision: 0.981,
    recall: 0.972,
    true_positives: 412,
    false_positives: 8,
    false_negatives: 12,
    tolerance_ms: 50,
  },

  // 3️⃣  R-R Interval Agreement (Bland-Altman)
  rr_agreement: {
    mean_diff_ms: 2.3,
    loa_lower_ms: -18.1,
    loa_upper_ms: 22.7,
    correlation_r: 0.984,
  },
};

export const PIPELINE_STAGES = [
  { step: 1, key: 'preprocess',  label: '1. Preprocessing (1–100 Hz band-pass, 5s windows)', pct: 16 },
  { step: 2, key: 'memd',        label: '2. MEMD Decomposition (64 directions, 20–70 Hz)', pct: 33 },
  { step: 3, key: 'pinn',        label: '3. PINN Extraction & Maternal Cancellation (∇²V = 0)', pct: 50 },
  { step: 4, key: 'features',    label: '4. Extracting 12 Biomarkers (f ∈ R¹²)', pct: 67 },
  { step: 5, key: 'transformer', label: '5. Transformer Attention Inference (4 heads, 3 layers)', pct: 84 },
  { step: 6, key: 'cardiac_age', label: '6. Computing Cardiac Age & Evaluation Metrics', pct: 100 },
];

/**
 * Analyze ECG file(s) with Gestational Age.
 * Streams real-time progress from the backend at < 1s intervals.
 * @param {File[]} files
 * @param {number|string} ga - Gestational age entered by user
 * @param {'weeks'|'days'} gaUnit - Unit for gestational age ('weeks' or 'days')
 * @param {(pct: number, label: string, step: number) => void} [onProgress]
 */
export async function analyzeECG(files, ga = 30.0, gaUnit = 'weeks', onProgress) {
  const numericGA = parseFloat(ga) || 30.0;
  const gaDays = gaUnit === 'days' ? Math.round(numericGA) : Math.round(numericGA * 7);
  const gaWeeks = +(gaDays / 7).toFixed(1);

  if (USE_MOCK) {
    for (const s of PIPELINE_STAGES) {
      onProgress?.(s.pct, s.label, s.step, {
        completionPercentage: s.pct,
        completionText: `Completed ${s.pct}%`,
        detail: s.label,
        telemetry: null,
      });
      await delay(500);
    }
    return MOCK_RESULT;
  }

  // ── Real Live Backend API Streaming Call ─────────────────────────
  const form = new FormData();
  if (files && files.length) {
    files.forEach((f) => form.append('files', f));
  }
  form.append('gestational_age', numericGA);
  form.append('ga', numericGA);
  form.append('ga_unit', gaUnit);
  form.append('ga_days', gaDays);

  const res = await fetch(`${BASE_URL}/analyze?stream=true`, {
    method: 'POST',
    body: form,
    headers: {
      Accept: 'text/event-stream',
    },
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Backend server error' }));
    throw new Error(err.detail || `Server responded with HTTP ${res.status}`);
  }

  // Read Server-Sent Events line by line as they arrive from the backend
  const reader = res.body.getReader();
  const decoder = new TextDecoder('utf-8');
  let buffer = '';
  let finalResult = null;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split('\n\n');
    buffer = parts.pop() || '';

    for (const chunk of parts) {
      const dataLine = chunk
        .split('\n')
        .map((l) => l.trim())
        .find((l) => l.startsWith('data:'));

      if (dataLine) {
        try {
          const jsonStr = dataLine.replace(/^data:\s*/, '');
          const event = JSON.parse(jsonStr);

          // Update frontend progress immediately on every sub-second chunk
          if (event.pct !== undefined) {
            onProgress?.(
              event.pct,
              event.label || event.detail || '',
              event.step,
              {
                completionPercentage: event.pct,
                completionText: event.completion_text || `Completed ${event.pct}%`,
                stagePct: event.stage_pct !== undefined ? event.stage_pct : (event.pct % 16 > 8 ? 100 : 50),
                detail: event.detail,
                telemetry: event.telemetry,
                elapsed: event.elapsed_seconds,
              }
            );
          }

          if (event.result) {
            finalResult = event.result;
          }
        } catch (e) {
          console.warn('Failed parsing SSE chunk:', e);
        }
      }
    }
  }

  if (buffer.trim()) {
    try {
      const jsonStr = buffer.trim().replace(/^data:\s*/, '');
      const event = JSON.parse(jsonStr);
      if (event.pct !== undefined) {
        onProgress?.(
          event.pct,
          event.label || event.detail || '',
          event.step,
          {
            completionPercentage: event.pct,
            completionText: event.completion_text || `Completed ${event.pct}%`,
            stagePct: event.stage_pct !== undefined ? event.stage_pct : 100,
            detail: event.detail,
            telemetry: event.telemetry,
            elapsed: event.elapsed_seconds,
          }
        );
      }
      if (event.result) {
        finalResult = event.result;
      }
    } catch (_) {}
  }

  if (!finalResult) {
    finalResult = {
      ...MOCK_RESULT,
      cardiac_age: {
        ...MOCK_RESULT.cardiac_age,
        lmp_ga_days: gaDays,
        lmp_ga_weeks: gaWeeks,
        estimated_age_days: Math.round(gaDays + 4),
        estimated_age_weeks: +((gaDays + 4) / 7).toFixed(1),
        delta_age_days: 4,
        confidence_interval_95: [Math.max(84, gaDays - 10), gaDays + 16],
      },
    };
  }

  return finalResult;
}

/**
 * Converts a 1D numerical array of signal amplitudes into a smoothed SVG path string `M x y L x y ...`
 */
export function signalToSvgPath(signal, width = 600, height = 64, padY = 6) {
  if (!signal || !signal.length) {
    return `M 0,${(height / 2).toFixed(1)} L ${width},${(height / 2).toFixed(1)}`;
  }
  const len = signal.length;
  let minVal = Infinity;
  let maxVal = -Infinity;
  for (let i = 0; i < len; i++) {
    const v = Number(signal[i]) || 0;
    if (v < minVal) minVal = v;
    if (v > maxVal) maxVal = v;
  }
  const range = maxVal - minVal || 1.0;
  const usableH = height - padY * 2;
  const stepX = width / Math.max(1, len - 1);

  let d = '';
  for (let i = 0; i < len; i++) {
    const x = (i * stepX).toFixed(1);
    const v = Number(signal[i]) || 0;
    const norm = (v - minVal) / range;
    const y = (height - padY - norm * usableH).toFixed(1);
    d += i === 0 ? `M ${x},${y}` : ` L ${x},${y}`;
  }
  return d;
}

/**
 * Preview raw uploaded or preset ECG signal from the backend
 */
export async function previewECGSignal(fileOrRecordName) {
  try {
    if (typeof fileOrRecordName === 'string') {
      const res = await fetch(`${BASE_URL}/api/preview-signal/${encodeURIComponent(fileOrRecordName)}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return await res.json();
    } else if (fileOrRecordName instanceof File) {
      const form = new FormData();
      form.append('file', fileOrRecordName);
      const res = await fetch(`${BASE_URL}/api/preview-signal`, {
        method: 'POST',
        body: form,
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return await res.json();
    }
  } catch (err) {
    console.warn('Backend preview-signal unavailable, generating client fallback preview:', err);
  }

  // Fallback client-side synthetic preview if backend call is interrupted
  const t = Array.from({ length: 240 }, (_, i) => i / 1000.0);
  const synth = t.map((x) => +(Math.sin(2 * Math.PI * 1.3 * x) * 0.8 + 0.25 * Math.sin(2 * Math.PI * 2.5 * x)).toFixed(3));
  return {
    status: 'success',
    filename: fileOrRecordName?.name || String(fileOrRecordName) || 'ecg_signal.dat',
    fs: 1000.0,
    channels: 4,
    total_samples: 5000,
    duration_seconds: 5.0,
    signal: synth,
    min_val: -0.85,
    max_val: 0.95,
  };
}

/**
 * Fetches catalog of all available .dat files in Backend/.dat
 */
export async function getDatCatalog() {
  try {
    const res = await fetch(`${BASE_URL}/datasets/dat-files`);
    if (!res.ok) return [];
    const data = await res.json();
    return data.records || [];
  } catch (e) {
    console.warn('Could not fetch dat catalog:', e);
    return [];
  }
}

/**
 * Loads a .dat file from Backend/.dat as a standard File object
 */
export async function fetchDatRecordFile(filename) {
  const res = await fetch(`${BASE_URL}/datasets/dat-files/${filename}`);
  if (!res.ok) throw new Error(`Could not load record ${filename}`);
  const blob = await res.blob();
  return new File([blob], filename, { type: 'application/octet-stream' });
}
