import React from 'react';

function ScoreBadge({ value, label, color }) {
  const isPercent = typeof value === 'number' && value <= 1;
  const display   = isPercent ? `${(value * 100).toFixed(1)}%` : value;
  return (
    <div className="eval-score-card">
      <p className="eval-score-value" style={{ color }}>{display}</p>
      <p className="eval-score-label">{label}</p>
    </div>
  );
}

function BlandAltmanSummary({ rr }) {
  if (!rr) return null;
  return (
    <div className="bland-altman-block">
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 14 }}>
        <p style={{ fontSize: 13, fontWeight: 700, color: 'var(--text-primary)' }}>R-R Interval Agreement</p>
        <span className="ecg-chip ecg-chip--cyan">Bland-Altman</span>
      </div>

      <div style={{ display: 'flex', gap: 16, marginBottom: 16 }}>
        <div className="ba-metric">
          <span className="ba-label">Mean Difference</span>
          <span className="ba-value cyan">{rr.mean_diff_ms > 0 ? '+' : ''}{rr.mean_diff_ms} ms</span>
        </div>
        <div className="ba-metric">
          <span className="ba-label">Limits of Agreement</span>
          <span className="ba-value green">[{rr.loa_lower_ms}, +{rr.loa_upper_ms}] ms</span>
        </div>
        <div className="ba-metric">
          <span className="ba-label">Correlation</span>
          <span className="ba-value purple">r = {rr.bland_altman_r}</span>
        </div>
      </div>

      {/* Visual LOA strip */}
      <div className="loa-strip-wrapper">
        <div className="loa-strip">
          <div className="loa-mean-line" style={{ left: '50%' }} title="Mean difference" />
          <div className="loa-band" style={{ left: '15%', right: '15%' }} />
          <span className="loa-text loa-lower">{rr.loa_lower_ms} ms</span>
          <span className="loa-text loa-upper">+{rr.loa_upper_ms} ms</span>
          <span className="loa-text loa-center">Mean = {rr.mean_diff_ms} ms</span>
        </div>
        <p className="loa-caption">Limits of Agreement (95%)</p>
      </div>
    </div>
  );
}

function QRSEvaluation({ qrs }) {
  if (!qrs) return null;
  return (
    <div className="qrs-eval-block">
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 14 }}>
        <p style={{ fontSize: 13, fontWeight: 700, color: 'var(--text-primary)' }}>QRS Detection</p>
        <span className="ecg-chip ecg-chip--green">Tolerance ±{qrs.tolerance_ms} ms</span>
      </div>
      <div className="qrs-metrics-row">
        {[
          { label: 'True Positives',  value: qrs.true_positives,  color: 'var(--green)' },
          { label: 'False Positives', value: qrs.false_positives, color: 'var(--orange)' },
          { label: 'False Negatives', value: qrs.false_negatives, color: 'var(--red)' },
        ].map(({ label, value, color }) => (
          <div key={label} className="qrs-metric">
            <span className="qrs-value" style={{ color }}>{value}</span>
            <span className="qrs-label">{label}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function Evaluation({ qrs_evaluation, rr_agreement, evaluation }) {
  return (
    <div>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 24 }}>
        <div className="card-icon green" style={{ width: 40, height: 40, borderRadius: 10, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 16, fontWeight: 800, color: '#059669', background: '#ecfdf5', border: '1px solid #a7f3d0' }}>MET</div>
        <div>
          <p className="card-title">Evaluation Metrics</p>
          <p className="card-subtitle">Extraction · R-R Agreement · Cardiac Age Error</p>
        </div>
      </div>

      {/* Top Score Cards */}
      <div className="eval-scores-row">
        <ScoreBadge
          value={evaluation?.extraction_f1}
          label="F₁ Score (QRS Detection)"
          color="var(--cyan)"
        />
        <ScoreBadge
          value={`${evaluation?.rr_mean_diff_ms ?? '–'} ms`}
          label="R-R Mean Difference"
          color="var(--purple)"
        />
        <ScoreBadge
          value={`${evaluation?.age_mae_days ?? '–'} days`}
          label="Cardiac Age MAE"
          color="var(--green)"
        />
      </div>

      <div className="divider" />
      <QRSEvaluation qrs={qrs_evaluation} />

      <div className="divider" />
      <BlandAltmanSummary rr={rr_agreement} />
    </div>
  );
}
