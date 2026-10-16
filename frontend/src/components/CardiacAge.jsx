import React from 'react';

function GaugeBar({ value, max, color }) {
  const pct = Math.min((value / max) * 100, 100);
  return (
    <div className="gauge-track">
      <div className="gauge-fill" style={{ width: `${pct}%`, background: color }} />
    </div>
  );
}

export default function CardiacAge({ cardiac_age, vae, evaluation }) {
  if (!cardiac_age) return null;

  const { estimated_age_days, lmp_ga_days, delta_age_days, transformer } = cardiac_age;
  const estWeeks = (estimated_age_days / 7).toFixed(1);
  const lmpWeeks = (lmp_ga_days / 7).toFixed(1);
  const isAhead  = delta_age_days > 0;
  const deltaAbs = Math.abs(delta_age_days);

  const metrics = [
    { label: 'Estimated Cardiac Age', value: estimated_age_days, unit: 'days', sub: `${estWeeks} weeks`, color: 'var(--cyan)' },
    { label: 'LMP-based GA',          value: lmp_ga_days,        unit: 'days', sub: `${lmpWeeks} weeks`, color: 'var(--purple)' },
    { label: 'ΔAge = CA − GA',        value: `${isAhead ? '+' : '−'}${deltaAbs}`, unit: 'days', sub: isAhead ? 'Cardiac age ahead' : 'Cardiac age behind', color: isAhead ? 'var(--orange)' : 'var(--cyan)' },
    { label: 'Age MAE',               value: evaluation?.age_mae_days ?? '–', unit: 'days', sub: '95% CI shown below', color: 'var(--green)' },
  ];

  return (
    <div>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 24 }}>
        <div className="card-icon red" style={{ width: 40, height: 40, borderRadius: 10, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 20 }}>❤️</div>
        <div>
          <p className="card-title">Cardiac Age Estimation</p>
          <p className="card-subtitle">ΔAge = CA − GA · MAE loss · AdamW</p>
        </div>
      </div>

      {/* Main metrics */}
      <div className="ca-metrics-grid">
        {metrics.map((m) => (
          <div key={m.label} className="ca-metric-card">
            <p className="ca-metric-label">{m.label}</p>
            <p className="ca-metric-value" style={{ color: m.color }}>
              {m.value}<span className="ca-metric-unit"> {m.unit}</span>
            </p>
            <p className="ca-metric-sub">{m.sub}</p>
          </div>
        ))}
      </div>

      {/* 95% CI bar */}
      {evaluation?.age_ci_95 && (
        <div className="ci-block">
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 8 }}>
            <span style={{ fontSize: 12, color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.08em' }}>
              95% Confidence Interval
            </span>
            <span style={{ fontSize: 13, fontFamily: 'JetBrains Mono', color: 'var(--cyan)' }}>
              [{evaluation.age_ci_95[0]} – {evaluation.age_ci_95[1]} days]
            </span>
          </div>
          <div className="ci-bar-wrapper">
            <div className="ci-bar">
              <div
                className="ci-range"
                style={{
                  left: `${((evaluation.age_ci_95[0] / 280) * 100).toFixed(1)}%`,
                  width: `${(((evaluation.age_ci_95[1] - evaluation.age_ci_95[0]) / 280) * 100).toFixed(1)}%`,
                }}
              />
              <div
                className="ci-marker"
                style={{ left: `${((estimated_age_days / 280) * 100).toFixed(1)}%` }}
                title={`Estimated: ${estimated_age_days} days`}
              />
            </div>
            <div className="ci-labels">
              <span>0</span><span>70 days</span><span>140 days</span><span>210 days</span><span>280 days</span>
            </div>
          </div>
        </div>
      )}

      <div className="divider" />

      {/* Transformer config */}
      {transformer && (
        <div>
          <p style={{ fontSize: 12, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.1em', color: 'var(--text-muted)', marginBottom: 12 }}>
            Transformer Architecture
          </p>
          <div className="transformer-grid">
            {[
              { k: 'Attention', v: transformer.attention },
              { k: 'Heads', v: transformer.heads },
              { k: 'Layers', v: transformer.layers },
              { k: 'd_model', v: transformer.d_model },
              { k: 'Loss', v: transformer.loss },
              { k: 'Optimizer', v: transformer.optimizer },
            ].map(({ k, v }) => (
              <div key={k} className="transformer-row">
                <span className="transformer-key">{k}</span>
                <span className="transformer-val">{v}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* VAE config */}
      {vae && (
        <>
          <div className="divider" />
          <p style={{ fontSize: 12, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.1em', color: 'var(--text-muted)', marginBottom: 12 }}>
            Contrastive VAE
          </p>
          <div className="transformer-grid">
            {[
              { k: 'Input space', v: vae.input_space },
              { k: 'Latent space', v: vae.latent_space },
              { k: 'Loss', v: vae.loss },
              { k: 'τ (tau)', v: vae.tau },
              { k: 'GA input', v: 'Not used' },
            ].map(({ k, v }) => (
              <div key={k} className="transformer-row">
                <span className="transformer-key">{k}</span>
                <span className="transformer-val">{v}</span>
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  );
}
