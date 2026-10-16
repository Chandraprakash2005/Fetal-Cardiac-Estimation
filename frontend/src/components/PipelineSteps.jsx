import React from 'react';

const STEPS = [
  {
    id: 1,
    label: 'Preprocessing',
    sub: '1–100 Hz bandpass filter',
    color: 'cyan',
  },
  {
    id: 2,
    label: 'MEMD',
    sub: '64 directions decomposition',
    color: 'purple',
  },
  {
    id: 3,
    label: 'PINN',
    sub: 'Maternal cancellation',
    color: 'cyan',
  },
  {
    id: 4,
    label: '12 Biomarkers',
    sub: 'Morphological features',
    color: 'green',
  },
  {
    id: 5,
    label: 'Contrastive VAE',
    sub: 'Latent embedding representation',
    color: 'purple',
  },
  {
    id: 6,
    label: 'Transformer',
    sub: '4-head self-attention',
    color: 'cyan',
  },
  {
    id: 7,
    label: 'Cardiac Age',
    sub: 'Fetal age prediction',
    color: 'red',
  },
];

const COLOR_MAP = {
  cyan:   { bg: 'var(--cyan-dim)',   border: 'rgba(64,200,255,0.3)',  text: 'var(--cyan)'   },
  purple: { bg: 'var(--purple-dim)', border: 'rgba(168,85,247,0.3)', text: 'var(--purple)' },
  green:  { bg: 'var(--green-dim)',  border: 'rgba(0,229,160,0.3)',  text: 'var(--green)'  },
  red:    { bg: 'var(--red-dim)',    border: 'rgba(255,77,106,0.3)', text: 'var(--red)'    },
};

export default function PipelineSteps({ activeStep = 0 }) {
  return (
    <div className="pipeline-wrapper">
      {/* Section Label */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 20 }}>
        <div className="card-icon cyan" style={{ width: 36, height: 36, borderRadius: 8, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 13, fontWeight: 700, background: '#eaf4fa', color: '#1882b4' }}>FLOW</div>
        <div>
          <p className="card-title">Processing Pipeline</p>
          <p className="card-subtitle">Fetal ECG Extraction → Cardiac Age Estimation</p>
        </div>
      </div>

      <div className="pipeline-row">
        {STEPS.map((step, idx) => {
          const c = COLOR_MAP[step.color];
          const isActive = idx < activeStep;
          const isCurrent = idx === activeStep - 1;

          return (
            <React.Fragment key={step.id}>
              <div className={`pipeline-step ${isActive ? 'pipeline-step--done' : ''} ${isCurrent ? 'pipeline-step--current' : ''}`}>
                <div
                  className="pipeline-icon"
                  style={{
                    background: isActive ? c.bg : 'rgba(255,255,255,0.03)',
                    border: `1px solid ${isActive ? c.border : 'var(--border)'}`,
                    color: isActive ? c.text : 'var(--text-muted)',
                    boxShadow: isCurrent ? `0 0 16px ${c.border}` : 'none',
                    fontWeight: 700,
                  }}
                >
                  {isActive ? '✓' : step.id}
                </div>
                <p className="pipeline-label" style={{ color: isActive ? 'var(--text-primary)' : 'var(--text-muted)' }}>
                  {step.label}
                </p>
                <p className="pipeline-sub">{step.sub}</p>
              </div>
              {idx < STEPS.length - 1 && (
                <div className={`pipeline-arrow ${isActive ? 'pipeline-arrow--done' : ''}`}>›</div>
              )}
            </React.Fragment>
          );
        })}
      </div>
    </div>
  );
}
