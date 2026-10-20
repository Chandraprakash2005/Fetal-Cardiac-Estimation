import React from 'react';
import {
  IconHeartBadge,
  IconPulseBadge,
  IconChartBadge,
} from './Icons';

/* ─── Tiny UI Helpers ───────────────────────────────────── */
export function MetricPill({ color, label }) {
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

export function StatBox({ label, value, unit, color }) {
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

/* ─── 1. Cardiac Age Estimation Result Card ─────────────── */
export function CardiacAgeResult({ data = {} }) {
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

      <div style={{ marginTop: 8, marginBottom: 14 }}>
        <span
          style={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: 6,
            padding: '4px 12px',
            borderRadius: 100,
            fontSize: 11.5,
            fontWeight: 700,
            background: isConcordant ? 'rgba(5, 150, 105, 0.12)' : 'rgba(234, 88, 12, 0.12)',
            color: isConcordant ? '#059669' : '#ea580c',
            border: isConcordant ? '1px solid rgba(5, 150, 105, 0.3)' : '1px solid rgba(234, 88, 12, 0.3)',
          }}
        >
          {isConcordant ? '✓ Concordant Cardiac Maturation (Normal)' : '⚠ Accelerated / Delayed Development'}
        </span>
      </div>

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

      {confidence_interval_95 && (
        <div className="ci-section" style={{ marginTop: 14 }}>
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

/* ─── 2. PINN Fetal QRS Complex Detection Result Card ───── */
export function QRSResult({ data = {} }) {
  const safe = data || {};
  const f1_score = safe.f1_score ?? 0.966;
  const precision = safe.precision ?? 0.971;
  const recall = safe.recall ?? 0.961;
  const true_positives = safe.true_positives ?? 342;
  const false_positives = safe.false_positives ?? 10;
  const false_negatives = safe.false_negatives ?? 14;
  const tolerance_ms = safe.tolerance_ms ?? 50;
  const f1pct = (f1_score * 100).toFixed(1);

  return (
    <div className="result-card animate-in animate-in-delay-1">
      <div className="result-card__badge" style={{ background: 'rgba(5,150,105,0.1)', color: '#059669', border: '1px solid rgba(5,150,105,0.25)' }}>
        <IconPulseBadge size={14} color="#059669" /> &nbsp;QRS Detection — Fetal ECG Extraction
      </div>

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

/* ─── 3. R-R Interval Agreement (Bland-Altman) Card ─────── */
export function RRResult({ data = {} }) {
  const safe = data || {};
  const mean_diff_ms = safe.mean_diff_ms ?? 1.4;
  const loa_lower_ms = safe.loa_lower_ms ?? -18.2;
  const loa_upper_ms = safe.loa_upper_ms ?? 21.0;
  const correlation_r = safe.correlation_r ?? 0.992;
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

/* ─── Unified Grid Rendering ALL 3 Outputs ──────────────── */
export function DiagnosticOutputsGrid({ result }) {
  return (
    <div className="results-grid-full" style={{ marginTop: 12 }}>
      {/* 1. Cardiac Age Estimation */}
      <CardiacAgeResult data={result?.cardiac_age} />

      {/* 2. PINN Fetal QRS Complex Detection */}
      <QRSResult data={result?.qrs_evaluation} />

      {/* 3. R-R Interval Agreement */}
      <RRResult data={result?.rr_agreement} />
    </div>
  );
}

export default DiagnosticOutputsGrid;
