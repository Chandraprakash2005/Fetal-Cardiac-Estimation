import React from 'react';

const TYPE_COLOR = {
  morphological: { text: 'var(--cyan)',   bg: 'var(--cyan-dim)',   label: 'Morphological' },
  dynamic:       { text: 'var(--purple)', bg: 'var(--purple-dim)', label: 'Dynamic / HRV' },
};

export default function Biomarkers({ biomarkers }) {
  if (!biomarkers?.length) return null;

  const morphological = biomarkers.filter((b) => b.type === 'morphological');
  const dynamic       = biomarkers.filter((b) => b.type === 'dynamic');

  return (
    <div>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 20 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <div className="card-icon purple" style={{ width: 36, height: 36, borderRadius: 8, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 16 }}>📈</div>
          <div>
            <p className="card-title">12 Biomarkers</p>
            <p className="card-subtitle">f ∈ R¹² — morphological + dynamic features</p>
          </div>
        </div>
        <div style={{ display: 'flex', gap: 8 }}>
          <span className="ecg-chip ecg-chip--cyan">6 Morphological</span>
          <span className="ecg-chip ecg-chip--purple">6 Dynamic</span>
        </div>
      </div>

      {/* Groups */}
      {[{ label: 'Morphological Features', items: morphological, type: 'morphological' },
        { label: 'Dynamic / HRV Features', items: dynamic, type: 'dynamic' }
      ].map(({ label, items, type }) => {
        const c = TYPE_COLOR[type];
        return (
          <div key={type} style={{ marginBottom: 20 }}>
            <p style={{ fontSize: 11, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.1em', color: c.text, marginBottom: 12 }}>
              {label}
            </p>
            <div className="biomarker-grid">
              {items.map((b) => (
                <div key={b.id} className="biomarker-card" style={{ borderColor: c.text.replace('var(', 'rgba(').replace(')', ',0.2)') }}>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 6 }}>
                    <span className="biomarker-id" style={{ color: c.text }}>#{b.id}</span>
                    <span className="biomarker-type-tag" style={{ background: c.bg, color: c.text }}>
                      {c.label}
                    </span>
                  </div>
                  <p className="biomarker-name">{b.name}</p>
                  <p className="biomarker-value" style={{ color: c.text }}>
                    {b.value}
                    <span className="biomarker-unit">{b.unit}</span>
                  </p>
                  {b.normal_range && (
                    <p className="biomarker-range">Normal: {b.normal_range} {b.unit}</p>
                  )}
                </div>
              ))}
            </div>
          </div>
        );
      })}
    </div>
  );
}
