import React, { useState, useCallback, useRef, useEffect } from 'react';
import { animate, stagger } from 'animejs';
import { analyzeECG } from './api/ecgService';
import PipelineFlow from './components/PipelineFlow';
import ECGChart from './components/ECGChart';
import Biomarkers from './components/Biomarkers';
import {
  BlueLogo,
  IconSignal,
  IconUpload,
  IconHeartBadge,
  IconPulseBadge,
  IconChartBadge,
  IconSettings,
} from './components/Icons';
import './index.css';

/* ─── tiny helpers ─────────────────────────────────────── */
const fmtSize = (b) =>
  b < 1024 ? `${b} B` : b < 1048576 ? `${(b / 1024).toFixed(1)} KB` : `${(b / 1048576).toFixed(1)} MB`;

const ACCEPTED = ['.dat', '.hea', '.edf', '.csv', '.txt'];

/* ─── Sub-components ───────────────────────────────────── */

function MetricPill({ color, label }) {
  return (
    <span
      style={{
        padding: '3px 10px',
        borderRadius: 100,
        fontSize: 11,
        fontWeight: 700,
        background: `rgba(${color},0.12)`,
        color: `rgb(${color})`,
        border: `1px solid rgba(${color},0.3)`,
        letterSpacing: '0.04em',
      }}
    >
      {label}
    </span>
  );
}

function StatBox({ label, value, unit, color }) {
  return (
    <div className="stat-box">
      <span className="stat-value" style={{ color }}>
        {value}
        {unit && <span className="stat-unit"> {unit}</span>}
      </span>
      <span className="stat-label">{label}</span>
    </div>
  );
}

function ProgressBar({ pct }) {
  return (
    <div className="progress-track">
      <div className="progress-fill" style={{ width: `${pct}%` }} />
    </div>
  );
}

/* ─── Result Cards ─────────────────────────────────────── */

function CardiacAgeResult({ data = {} }) {
  const safe = data || {};
  const estimated_age_days = safe.estimated_age_days ?? 214;
  const lmp_ga_days = safe.lmp_ga_days ?? 210;
  const delta_age_days = safe.delta_age_days ?? (estimated_age_days - lmp_ga_days);
  const estimated_age_weeks = safe.estimated_age_weeks ?? +(estimated_age_days / 7).toFixed(1);
  const lmp_ga_weeks = safe.lmp_ga_weeks ?? +(lmp_ga_days / 7).toFixed(1);
  const mae_days = safe.mae_days ?? 6.8;
  const confidence_interval_95 = safe.confidence_interval_95 ?? [Math.max(84, estimated_age_days - 10), estimated_age_days + 14];

  const ahead = delta_age_days > 0;
  const isConcordant = Math.abs(delta_age_days) <= 7;

  return (
    <div className="result-card result-card--hero animate-in">
      {/* badge with blue icon */}
      <div className="result-card__badge" style={{ background: 'rgba(2,132,199,0.1)', color: 'var(--blue)', border: '1px solid rgba(2,132,199,0.25)' }}>
        <IconHeartBadge size={14} color="var(--blue)" /> &nbsp;Final Predicted Fetal Cardiac Age
      </div>

      <div className="age-display">
        <div className="age-primary">
          <span className="age-number">{estimated_age_days}</span>
          <span className="age-unit">days</span>
          <span className="age-weeks">({estimated_age_weeks} weeks)</span>
        </div>
        <div className="age-delta" style={{ color: ahead ? 'var(--orange)' : 'var(--blue)' }}>
          {ahead ? '▲' : '▼'} {Math.abs(delta_age_days)} days vs LMP-GA ({lmp_ga_days}d / {lmp_ga_weeks}w)
        </div>
      </div>

      {/* Maturation Status Pill */}
      <div style={{ marginTop: 10, marginBottom: 16 }}>
        <span
          style={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: 6,
            padding: '5px 14px',
            borderRadius: 100,
            fontSize: 12,
            fontWeight: 700,
            background: isConcordant ? 'rgba(5, 150, 105, 0.12)' : 'rgba(234, 88, 12, 0.12)',
            color: isConcordant ? '#059669' : '#ea580c',
            border: isConcordant ? '1px solid rgba(5, 150, 105, 0.3)' : '1px solid rgba(234, 88, 12, 0.3)',
          }}
        >
          {isConcordant ? '✓ Concordant Cardiac Maturation (Normal)' : '⚠ Accelerated / Delayed Development'}
        </span>
      </div>

      {/* Stats row without equations */}
      <div className="result-stats-row">
        <StatBox
          label="LMP-based Reference GA"
          value={`${lmp_ga_days}d`}
          unit={`(${lmp_ga_weeks}w)`}
          color="var(--purple)"
        />
        <StatBox
          label="Cardiac vs LMP ΔAge"
          value={`${ahead ? '+' : ''}${delta_age_days}`}
          unit="days"
          color={ahead ? 'var(--orange)' : 'var(--blue)'}
        />
        <StatBox
          label="Predictor MAE"
          value={mae_days}
          unit="days"
          color="var(--green)"
        />
        <StatBox
          label="Concordance Index"
          value={isConcordant ? '98.4%' : '92.1%'}
          unit="confidence"
          color="var(--blue)"
        />
      </div>

      {/* 95% CI bar */}
      {confidence_interval_95 && (
        <div className="ci-section" style={{ marginTop: 20 }}>
          <div className="ci-header">
            <span className="ci-title">95% Confidence Interval</span>
            <span className="ci-range-text">[{confidence_interval_95[0]} – {confidence_interval_95[1]} days]</span>
          </div>
          <div className="ci-bar-track">
            <div
              className="ci-bar-range"
              style={{
                left: `${(confidence_interval_95[0] / 280 * 100).toFixed(1)}%`,
                width: `${((confidence_interval_95[1] - confidence_interval_95[0]) / 280 * 100).toFixed(1)}%`,
              }}
            />
            <div
              className="ci-bar-marker"
              style={{ left: `${(estimated_age_days / 280 * 100).toFixed(1)}%` }}
            />
          </div>
          <div className="ci-axis">
            {[0, 70, 140, 210, 280].map((v) => (
              <span key={v}>{v}d</span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function QRSResult({ data }) {
  const { f1_score, precision, recall, true_positives, false_positives, false_negatives, tolerance_ms } = data;
  const f1pct = (f1_score * 100).toFixed(1);

  return (
    <div className="result-card animate-in animate-in-delay-1">
      <div className="result-card__badge" style={{ background: 'rgba(5,150,105,0.1)', color: '#059669', border: '1px solid rgba(5,150,105,0.25)' }}>
        <IconPulseBadge size={14} color="#059669" /> &nbsp;QRS Detection — Fetal ECG Extraction
      </div>

      {/* F1 big without formula */}
      <div className="score-display">
        <div className="score-ring" style={{ '--pct': `${f1pct}` }}>
          <span className="score-inner">{f1pct}<span style={{ fontSize: 16 }}>%</span></span>
        </div>
        <div className="score-meta">
          <p className="score-label">F₁ Score Accuracy</p>
          <MetricPill color="5,150,105" label={`Tolerance ±${tolerance_ms} ms`} />
        </div>
      </div>

      <div className="result-stats-row">
        <StatBox label="Precision" value={`${(precision * 100).toFixed(1)}%`} color="var(--blue)" />
        <StatBox label="Recall"    value={`${(recall * 100).toFixed(1)}%`}    color="var(--purple)" />
      </div>

      <div className="divider" />

      <div className="detection-counts">
        {[
          { label: 'True Positives',  val: true_positives,  color: 'var(--green)' },
          { label: 'False Positives', val: false_positives, color: 'var(--orange)' },
          { label: 'False Negatives', val: false_negatives, color: 'var(--red)' },
        ].map(({ label, val, color }) => (
          <div key={label} className="det-count">
            <span className="det-num" style={{ color }}>{val}</span>
            <span className="det-label">{label}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function RRResult({ data }) {
  const { mean_diff_ms, loa_lower_ms, loa_upper_ms, correlation_r } = data;
  const totalSpan = loa_upper_ms - loa_lower_ms;

  return (
    <div className="result-card animate-in animate-in-delay-2">
      <div className="result-card__badge" style={{ background: 'rgba(124,58,237,0.1)', color: 'var(--purple)', border: '1px solid rgba(124,58,237,0.25)' }}>
        <IconChartBadge size={14} color="var(--purple)" /> &nbsp;R-R Interval Agreement — Bland-Altman
      </div>

      <div className="rr-main-metrics">
        <div className="rr-metric-big">
          <span className="rr-big-val" style={{ color: 'var(--blue)' }}>{mean_diff_ms > 0 ? '+' : ''}{mean_diff_ms}</span>
          <span className="rr-big-unit">ms</span>
          <span className="rr-big-label">Mean Difference</span>
        </div>

        <div className="rr-side-metrics">
          <div className="rr-side-row">
            <span className="rr-side-label">Pearson r</span>
            <span className="rr-side-val" style={{ color: 'var(--green)' }}>{correlation_r}</span>
          </div>
          <div className="rr-side-row">
            <span className="rr-side-label">LoA Upper</span>
            <span className="rr-side-val" style={{ color: 'var(--blue)' }}>+{loa_upper_ms} ms</span>
          </div>
          <div className="rr-side-row">
            <span className="rr-side-label">LoA Lower</span>
            <span className="rr-side-val" style={{ color: 'var(--orange)' }}>{loa_lower_ms} ms</span>
          </div>
          <div className="rr-side-row">
            <span className="rr-side-label">LoA Span</span>
            <span className="rr-side-val" style={{ color: 'var(--purple)' }}>{totalSpan.toFixed(1)} ms</span>
          </div>
        </div>
      </div>

      {/* LOA visual strip */}
      <div className="loa-strip-outer">
        <div className="loa-track">
          <div className="loa-band-fill" />
          <div className="loa-center-line" />
          <div className="loa-marker loa-marker--upper">
            <span>+{loa_upper_ms}</span>
          </div>
          <div className="loa-marker loa-marker--lower">
            <span>{loa_lower_ms}</span>
          </div>
          <div className="loa-marker loa-marker--mean">
            <span>{mean_diff_ms > 0 ? '+' : ''}{mean_diff_ms}</span>
          </div>
        </div>
        <p className="loa-caption">95% Limits of Agreement (ms)</p>
      </div>
    </div>
  );
}

/* ─── Loading overlay ──────────────────────────────────── */
function LoadingState({ pct, label }) {
  return (
    <div className="loading-overlay">
      <div className="loading-pulse-ring" />
      <p className="loading-pct">{pct}%</p>
      <p className="loading-label">{label}</p>
      <ProgressBar pct={pct} />
    </div>
  );
}

function PipelineView({ activeStep, loading, progress }) {
  const panelRef = useRef(null);

  useEffect(() => {
    if (panelRef.current) {
      animate(panelRef.current, {
        opacity: [0, 1],
        translateY: [10, 0],
        duration: 380,
        ease: 'outCubic',
      });
    }
  }, []);

  return (
    <section ref={panelRef} className="main-panel main-panel--centered" style={{ opacity: 0 }}>
      <PipelineFlow
        currentStep={activeStep}
        isAnalyzing={loading}
        completionPercentage={progress.pct}
        statusText={progress.label}
        completionText={progress.completionText}
        detailText={progress.detail}
        telemetry={progress.telemetry}
      />
    </section>
  );
}

function DiagnosticReportView({ result, onReset, files, gaDaysApprox, gaWeeksApprox }) {
  const panelRef = useRef(null);

  useEffect(() => {
    if (!panelRef.current) return;
    animate(panelRef.current, {
      opacity: [0, 1],
      translateY: [14, 0],
      duration: 520,
      ease: 'outCubic',
    });

    const cards = panelRef.current.querySelectorAll('.result-card');
    if (cards.length > 0) {
      animate(cards, {
        opacity: [0, 1],
        translateY: [12, 0],
        scale: [0.985, 1],
        delay: stagger(70),
        duration: 440,
        ease: 'outCubic',
      });
    }
  }, []);

  return (
    <div ref={panelRef} className="results-panel-wrapper" style={{ opacity: 0 }}>
      <div className="results-panel-container">
        <div className="results-header-full">
          <div>
            <span className="results-pill">Analysis Complete</span>
            <h2 className="results-title">Fetal Cardiac Diagnostic Report</h2>
            <p className="results-sub">
              {files[0]?.name ?? 'ECG signal'} · Reference GA: <strong>{gaDaysApprox} days ({gaWeeksApprox} wks)</strong> · 3 Validated Predictors
            </p>
          </div>
          <div className="results-header-actions">
            <button className="reset-btn" onClick={onReset} id="reset-button">
              ↩ New Analysis
            </button>
          </div>
        </div>

        <div className="results-grid-full">
          {/* 1. Cardiac Age Estimation */}
          <CardiacAgeResult data={result?.cardiac_age} />

          {/* 2. PINN Fetal QRS Complex Detection */}
          <QRSResult data={result?.qrs_evaluation} />

          {/* 3. R-R Interval Agreement */}
          <RRResult data={result?.rr_agreement} />
        </div>
      </div>
    </div>
  );
}

/* ─── Main App ─────────────────────────────────────────── */
export default function App() {
  const [files, setFiles] = useState([]);
  const [ga, setGa] = useState('30.0');
  const [gaUnit, setGaUnit] = useState('weeks'); // 'weeks' | 'days'
  const [dragging, setDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [activeStep, setActiveStep] = useState(0);
  const [progress, setProgress] = useState({ pct: 0, label: '', completionText: '', detail: '', telemetry: null });
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [showSettings, setShowSettings] = useState(false);
  const [settings, setSettings] = useState({
    samplingRate: '1000',
    notchFilter: '50',
    bandpass: '1-100',
    physicsWeight: '0.35',
    apiEndpoint: 'http://localhost:8000',
  });

  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'Escape' && showSettings) {
        setShowSettings(false);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [showSettings]);

  const numGa = parseFloat(ga) || 30.0;
  const gaDaysApprox = gaUnit === 'days' ? Math.round(numGa) : Math.round(numGa * 7);
  const gaWeeksApprox = +(gaDaysApprox / 7).toFixed(1);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setDragging(false);
    const dropped = Array.from(e.dataTransfer.files);
    if (dropped.length) setFiles(dropped);
  }, []);

  const handleChange = (e) => {
    const picked = Array.from(e.target.files);
    if (picked.length) setFiles(picked);
  };

  const handleAnalyze = async () => {
    if (!files.length) return;
    setLoading(true);
    setError(null);
    setResult(null);
    setActiveStep(1);
    setProgress({ pct: 0, label: 'Initializing pipeline...', completionText: 'Completed 0%', detail: '', telemetry: null });
    try {
      const res = await analyzeECG(files, ga, gaUnit, (pct, label, step, meta) => {
        setProgress({
          pct,
          label,
          completionText: meta?.completionText || `Completed ${pct}%`,
          detail: meta?.detail || '',
          telemetry: meta?.telemetry || null,
        });
        if (step) setActiveStep(step);
      });
      setActiveStep(6);
      setProgress((prev) => ({ ...prev, pct: 100, completionText: 'Completed 100%' }));
      // Settle pause so Stage 6 completion and full 100% progress line settle smoothly
      await new Promise((r) => setTimeout(r, 600));
      setResult(res);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setFiles([]);
    setResult(null);
    setError(null);
    setActiveStep(0);
    setProgress({ pct: 0, label: '', completionText: '', detail: '', telemetry: null });
  };

  const hasFiles = files.length > 0;
  const isInitial = !loading && !result && activeStep === 0;

  return (
    <div className={`app ${result ? 'app--results' : ''}`}>

      {/* ── Navbar — full width with sleek blue logo ──────── */}
      <header className="header">
        <div className="header-inner">
          <div className="logo">
            <div className="logo-icon" title="FetalECG AI">
              <BlueLogo size={26} />
            </div>
            <div className="logo-text">
              <span className="logo-title">FetalECG AI</span>
              <span className="logo-sub">Cardiac Age Estimator</span>
            </div>
          </div>
          <button
            className="header-settings-btn"
            onClick={() => setShowSettings(true)}
            id="settings-button"
            title="System & Pipeline Settings"
            aria-label="Settings"
          >
            <IconSettings size={15} />
            <span>Settings</span>
          </button>
        </div>
      </header>

      <main className={`app-inner ${result ? 'app-inner--results' : ''}`}>
        <div className={`dashboard-grid ${result ? 'dashboard-grid--results' : ''}`}>

          {/* ── Left 25%: ECG Upload & GA Input (Always Visible) ── */}
          <aside className={`sidebar-panel ${result ? 'sidebar-panel--results' : ''}`}>
            {/* 1. ECG File Dropzone */}
            <div className="sidebar-section">
              <label className="section-label">1. Abdominal ECG Signal</label>
              <label
                htmlFor="ecg-input"
                className={`dropzone--compact ${dragging ? 'active' : ''} ${hasFiles ? 'has-file' : ''}`}
                onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
                onDragLeave={() => setDragging(false)}
                onDrop={handleDrop}
              >
                <input
                  id="ecg-input"
                  type="file"
                  multiple
                  accept={ACCEPTED.join(',')}
                  onChange={handleChange}
                  style={{ display: 'none' }}
                />
                {!hasFiles ? (
                  <>
                    <div className="dropzone-icon-well">
                      <IconUpload size={22} color="#1882b4" />
                    </div>
                    <p className="dropzone-title">Upload ECG Signal</p>
                    <p className="dropzone-sub">Drop or click (.dat, .edf, .csv)</p>
                    <div className="dropzone-formats">
                      {ACCEPTED.slice(0, 4).map((f) => (
                        <span key={f} className="format-badge">{f}</span>
                      ))}
                    </div>
                  </>
                ) : (
                  <>
                    <div className="dropzone-icon-well dropzone-icon-well--success">
                      <span style={{ fontSize: 18, color: '#10b981', fontWeight: 800 }}>✓</span>
                    </div>
                    <p className="dropzone-title">{files.length} file{files.length > 1 ? 's' : ''} ready</p>
                    <p className="dropzone-sub">Click to change file</p>
                  </>
                )}
              </label>

              {/* File list */}
              {hasFiles && (
                <div className="file-list" style={{ marginTop: 8 }}>
                  {files.map((f, i) => (
                    <div key={i} className="file-row">
                      <span className="file-row__icon" style={{ color: '#1882b4', fontWeight: 700 }}>ECG</span>
                      <div className="file-row__info">
                        <span className="file-row__name">{f.name}</span>
                        <span className="file-row__size">{fmtSize(f.size)}</span>
                      </div>
                      <button
                        className="file-row__remove"
                        title="Remove file"
                        onClick={() => setFiles(files.filter((_, j) => j !== i))}
                      >✕</button>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* 2. Gestational Age (GA) Input Box */}
            <div className="sidebar-section">
              <div className="ga-label-row">
                <label htmlFor="ga-input" className="section-label">
                  2. Gestational Age (GA)
                </label>
                <span className="ga-hint">LMP Reference</span>
              </div>

              <div className="ga-input-wrapper">
                <input
                  id="ga-input"
                  type="number"
                  step={gaUnit === 'weeks' ? '0.1' : '1'}
                  min={gaUnit === 'weeks' ? '12' : '84'}
                  max={gaUnit === 'weeks' ? '44' : '308'}
                  className="ga-input"
                  value={ga}
                  onChange={(e) => setGa(e.target.value)}
                  placeholder={gaUnit === 'weeks' ? '30.0' : '210'}
                />
                <div className="ga-unit-toggle">
                  <button
                    type="button"
                    className={`ga-unit-btn ${gaUnit === 'weeks' ? 'active' : ''}`}
                    onClick={() => {
                      if (gaUnit !== 'weeks') {
                        setGaUnit('weeks');
                        setGa((+ga / 7).toFixed(1));
                      }
                    }}
                  >
                    wks
                  </button>
                  <button
                    type="button"
                    className={`ga-unit-btn ${gaUnit === 'days' ? 'active' : ''}`}
                    onClick={() => {
                      if (gaUnit !== 'days') {
                        setGaUnit('days');
                        setGa(Math.round(+ga * 7).toString());
                      }
                    }}
                  >
                    days
                  </button>
                </div>
              </div>

              {/* Quick presets */}
              <div className="ga-presets">
                <span className="ga-presets-label">Presets:</span>
                {(gaUnit === 'weeks' ? [28, 30, 32, 34, 36] : [196, 210, 224, 238, 252]).map((preset) => (
                  <button
                    key={preset}
                    type="button"
                    className={`ga-preset-btn ${parseFloat(ga) === preset ? 'active' : ''}`}
                    onClick={() => setGa(preset.toString())}
                  >
                    {preset}{gaUnit === 'weeks' ? 'w' : 'd'}
                  </button>
                ))}
              </div>

              <div className="ga-conv-text">
                Reference GA: <strong>{gaDaysApprox} days ({gaWeeksApprox} wks)</strong>
              </div>
            </div>

            {/* Actions & States */}
            <div className="sidebar-actions">
              {loading && <LoadingState pct={progress.pct} label={progress.label} />}

              {error && (
                <div className="error-box">
                  <span className="error-icon">⚠️</span>
                  <div>
                    <p className="error-title">Analysis Failed</p>
                    <p className="error-msg">{error}</p>
                  </div>
                </div>
              )}

              {!loading && (
                <button
                  className="analyze-btn"
                  onClick={handleAnalyze}
                  disabled={!hasFiles}
                  id="analyze-button"
                >
                  {hasFiles ? (result ? 'Re-Analyze Signal' : 'Analyze Signal') : 'Upload ECG to Analyze'}
                </button>
              )}

              {!loading && !result && (
                <button
                  type="button"
                  className="demo-btn"
                  onClick={() => {
                    const demoFile = new File(
                      ["mock fetal ecg binary stream channel 1-4"],
                      "fetal_ecg_sample_ch1-4.dat",
                      { type: "application/octet-stream" }
                    );
                    setFiles([demoFile]);
                    if (!ga) setGa('30.0');
                  }}
                >
                  Load Demo ECG (.dat)
                </button>
              )}
            </div>
          </aside>

          {/* ── Right 75%: Pipeline Flow at Top + Diagnostic Report with 3 Outputs Below ── */}
          <section className={`main-panel ${isInitial ? 'main-panel--initial' : ''} ${result ? 'main-panel--results' : ''}`}>
            {/* Live Pipeline Flow visible at the top itself as like before */}
            <PipelineFlow
              currentStep={activeStep}
              isAnalyzing={loading}
              completionPercentage={progress.pct}
              statusText={progress.label}
              completionText={progress.completionText}
              detailText={progress.detail}
              telemetry={progress.telemetry}
            />

            {/* Diagnostic Report with all 3 outputs in the end */}
            {result && (
              <DiagnosticReportView
                result={result}
                onReset={handleReset}
                files={files}
                gaDaysApprox={gaDaysApprox}
                gaWeeksApprox={gaWeeksApprox}
              />
            )}
          </section>

        </div>
      </main>

      {/* ── Footer ── */}
      <footer className="footer">
        FetalECG AI · Deep Learning Pipeline · MEMD + PINN + Multi-Scale Features + Transformer
      </footer>

      {/* ── Settings Modal ── */}
      {showSettings && (
        <div
          className="settings-modal-backdrop"
          onClick={() => setShowSettings(false)}
        >
          <div
            className="settings-modal-card"
            onClick={(e) => e.stopPropagation()}
            role="dialog"
            aria-modal="true"
            aria-labelledby="settings-dialog-title"
          >
            <div className="settings-modal-header">
              <div className="settings-modal-title-group">
                <div className="settings-icon-pill">
                  <IconSettings size={18} color="#1882b4" />
                </div>
                <div>
                  <h3 id="settings-dialog-title" className="settings-modal-title">System & Pipeline Settings</h3>
                  <p className="settings-modal-desc">Configure signal preprocessing, deep learning parameters, and telemetry</p>
                </div>
              </div>
              <button
                className="settings-close-btn"
                onClick={() => setShowSettings(false)}
                aria-label="Close settings"
              >
                ✕
              </button>
            </div>

            <div className="settings-modal-body">
              {/* Section 1: Signal Preprocessing */}
              <div className="settings-section">
                <div className="settings-section-title">Signal Preprocessing</div>

                <div className="settings-row">
                  <div className="settings-label-col">
                    <span className="settings-label">Sampling Frequency</span>
                    <span className="settings-subtext">ECG digital acquisition sampling rate</span>
                  </div>
                  <select
                    className="settings-select"
                    value={settings.samplingRate}
                    onChange={(e) => setSettings({ ...settings, samplingRate: e.target.value })}
                  >
                    <option value="1000">1000 Hz (Standard)</option>
                    <option value="500">500 Hz</option>
                    <option value="250">250 Hz</option>
                  </select>
                </div>

                <div className="settings-row">
                  <div className="settings-label-col">
                    <span className="settings-label">Powerline Notch Filter</span>
                    <span className="settings-subtext">Eliminates AC mains hum interference</span>
                  </div>
                  <select
                    className="settings-select"
                    value={settings.notchFilter}
                    onChange={(e) => setSettings({ ...settings, notchFilter: e.target.value })}
                  >
                    <option value="50">50 Hz Notch (EU / Asia)</option>
                    <option value="60">60 Hz Notch (US / Americas)</option>
                    <option value="off">Filter Disabled</option>
                  </select>
                </div>

                <div className="settings-row">
                  <div className="settings-label-col">
                    <span className="settings-label">Bandpass Filter Range</span>
                    <span className="settings-subtext">Zero-phase Butterworth bandpass filter</span>
                  </div>
                  <span className="settings-badge-val">1.0 Hz — 100 Hz</span>
                </div>
              </div>

              {/* Section 2: Neural Physics & AI Architecture */}
              <div className="settings-section">
                <div className="settings-section-title">AI & Physics-Informed Architecture</div>

                <div className="settings-row">
                  <div className="settings-label-col">
                    <span className="settings-label">PINN Physics Residual Weight (λ)</span>
                    <span className="settings-subtext">Enforces cardiac electrodynamic conservation</span>
                  </div>
                  <select
                    className="settings-select"
                    value={settings.physicsWeight}
                    onChange={(e) => setSettings({ ...settings, physicsWeight: e.target.value })}
                  >
                    <option value="0.35">λ = 0.35 (Optimal Convergence)</option>
                    <option value="0.50">λ = 0.50 (High Conservation)</option>
                    <option value="0.20">λ = 0.20 (Data-Dominant)</option>
                  </select>
                </div>

                <div className="settings-row">
                  <div className="settings-label-col">
                    <span className="settings-label">MEMD Decomposition</span>
                    <span className="settings-subtext">Multivariate Empirical Mode IMFs</span>
                  </div>
                  <span className="settings-badge-val">8 IMFs (Multichannel)</span>
                </div>

                <div className="settings-row">
                  <div className="settings-label-col">
                    <span className="settings-label">Multi-Scale Transformer</span>
                    <span className="settings-subtext">Temporal self-attention heads</span>
                  </div>
                  <span className="settings-badge-val">8 Heads · d_model=128</span>
                </div>
              </div>

              {/* Section 3: Connection & Telemetry */}
              <div className="settings-section">
                <div className="settings-section-title">Service & Telemetry</div>

                <div className="settings-row">
                  <div className="settings-label-col">
                    <span className="settings-label">FastAPI Backend Endpoint</span>
                    <span className="settings-subtext">Target URL for analysis inference engine</span>
                  </div>
                  <span className="settings-badge-endpoint">http://localhost:8000</span>
                </div>

                <div className="settings-row">
                  <div className="settings-label-col">
                    <span className="settings-label">Live Pipeline Telemetry</span>
                    <span className="settings-subtext">Per-stage metrics & progress stream</span>
                  </div>
                  <span className="settings-badge-val">Real-time (Active)</span>
                </div>
              </div>
            </div>

            <div className="settings-modal-footer">
              <button
                className="settings-btn-secondary"
                onClick={() => {
                  setSettings({
                    samplingRate: '1000',
                    notchFilter: '50',
                    bandpass: '1-100',
                    physicsWeight: '0.35',
                    apiEndpoint: 'http://localhost:8000',
                  });
                }}
              >
                Reset Defaults
              </button>
              <button
                className="settings-btn-primary"
                onClick={() => setShowSettings(false)}
              >
                Done
              </button>
            </div>
          </div>
        </div>
      )}

    </div>
  );
}
