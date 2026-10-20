import React, { useState, useEffect, useRef } from 'react';
import { animate } from 'animejs';
import {
  BlueLogo,
  IconPreprocess,
  IconMEMD,
  IconPINN,
  IconFeature,
  IconTransformer,
} from './Icons';
import StageVisualizer from './StageVisualizer';

export const LINEAR_STEPS = [
  {
    id: 1,
    title: 'Preprocess',
    short: 'Preprocess',
    Icon: IconPreprocess,
    sub: '1–100 Hz Bandpass',
  },
  {
    id: 2,
    title: 'MEMD',
    short: 'MEMD',
    Icon: IconMEMD,
    sub: '64 Multiscale IMFs',
  },
  {
    id: 3,
    title: 'PINN',
    short: 'PINN',
    Icon: IconPINN,
    sub: 'Maternal Cancellation',
  },
  {
    id: 4,
    title: '12 Feature',
    short: '12 Feature',
    Icon: IconFeature,
    sub: 'Biomarkers Extraction',
  },
  {
    id: 5,
    title: 'Transformer',
    short: 'Transformer',
    Icon: IconTransformer,
    sub: 'Multi-Head Attention',
  },
  {
    id: 6,
    title: 'Cardiac Age',
    short: 'Cardiac Age',
    Icon: BlueLogo, // Animated blue heart ECG logo for stage 6
    sub: 'Clinical Estimation',
  },
];

/* ─── Blue Water Liquid Conduit Connector between Adjacent Squares ─── */
function AdjacentWaveConnector({ isCompleted = false, isProcessing = false, progressPct = 0 }) {
  const targetPct = isCompleted ? 100 : isProcessing ? Math.max(0, Math.min(100, progressPct)) : 0;
  const [smoothPct, setSmoothPct] = useState(targetPct);
  const animRef = useRef(null);
  const pctObj = useRef({ val: targetPct });

  useEffect(() => {
    animRef.current?.pause?.();
    animRef.current = animate(pctObj.current, {
      val: targetPct,
      duration: isCompleted ? 500 : 700,
      ease: 'outCubic',
      onUpdate: () => {
        setSmoothPct(pctObj.current.val);
      },
    });
    return () => animRef.current?.pause?.();
  }, [targetPct, isCompleted]);

  return (
    <div
      className={`adj-wave-connector ${isProcessing ? 'adj-wave-connector--processing' : ''} ${isCompleted ? 'adj-wave-connector--completed' : ''}`}
      title={isCompleted ? 'Water flow complete (100%)' : isProcessing ? `Water streaming: ${Math.round(smoothPct)}%` : 'Standby conduit'}
    >
      {/* 1. Neumorphic Carved Channel Trench Background */}
      <div className="water-channel-trench" />

      {/* 2. Flowing Blue Water Liquid Stream */}
      <div
        className="water-stream-fill"
        style={{ width: `${smoothPct.toFixed(1)}%` }}
      >
        {/* Continuous Flowing Water Current Waves & Ripples */}
        <div className="water-current-ripples" />

        {/* Top Water Surface Meniscus / Glossy Sheen */}
        <div className="water-surface-sheen" />

        {/* Active Leading Water Surge Droplet */}
        {isProcessing && smoothPct > 0 && smoothPct < 99 && (
          <div className="water-leading-drop" />
        )}
      </div>
    </div>
  );
}

/* ─── Creative Precision Tech HUD Reticle with Anime.js Fade & Scale ─── */
function CreativeTechFrame({ isVisible }) {
  const frameRef = useRef(null);
  const animRef = useRef(null);
  const isFirstRender = useRef(true);

  useEffect(() => {
    if (!frameRef.current) return;
    animRef.current?.pause?.();

    if (isVisible) {
      animRef.current = animate(frameRef.current, {
        opacity: [0, 1],
        scale: [0.86, 1],
        duration: 360,
        ease: 'outBack(1.3)',
      });
    } else if (!isFirstRender.current) {
      animRef.current = animate(frameRef.current, {
        opacity: [1, 0],
        scale: [1, 0.86],
        duration: 220,
        ease: 'inQuad',
      });
    }
    isFirstRender.current = false;
    return () => animRef.current?.pause?.();
  }, [isVisible]);

  return (
    <svg
      ref={frameRef}
      className="linear-node__tech-svg"
      viewBox="0 0 74 74"
      aria-hidden="true"
      style={{
        opacity: isVisible ? 1 : 0,
        transform: isVisible ? 'scale(1)' : 'scale(0.86)',
        pointerEvents: 'none',
      }}
    >
      {/* 4 Precision HUD Corner Targeting Brackets */}
      <path className="tech-corner tech-corner--tl" d="M 1,14 L 1,7 A 6,6 0 0 1 7,1 L 14,1" />
      <path className="tech-corner tech-corner--tr" d="M 60,1 L 67,1 A 6,6 0 0 1 73,7 L 73,14" />
      <path className="tech-corner tech-corner--br" d="M 73,60 L 73,67 A 6,6 0 0 1 67,73 L 60,73" />
      <path className="tech-corner tech-corner--bl" d="M 14,73 L 7,73 A 6,6 0 0 1 1,67 L 1,60" />

      {/* Crisp technical micro-dash track rim */}
      <rect
        className="tech-track-rim"
        x="6"
        y="6"
        width="62"
        height="62"
        rx="17"
        ry="17"
      />

      {/* Orbiting solid blue precision tracker beacon (Zero Glow) */}
      <rect
        className="tech-orbital-beacon"
        x="6"
        y="6"
        width="62"
        height="62"
        rx="17"
        ry="17"
        pathLength="100"
      />
    </svg>
  );
}

/* ─── Smooth Completed Checkmark Badge with Anime.js Spring ─── */
function NodeCheckmark({ isVisible }) {
  const checkRef = useRef(null);
  const animRef = useRef(null);
  const isFirstRender = useRef(true);

  useEffect(() => {
    if (!checkRef.current) return;
    animRef.current?.pause?.();

    if (isVisible) {
      animRef.current = animate(checkRef.current, {
        opacity: [0, 1],
        scale: [0, 1.25, 1],
        rotate: [-20, 0],
        duration: 350,
        ease: 'outBack(1.6)',
      });
    } else if (!isFirstRender.current) {
      animRef.current = animate(checkRef.current, {
        opacity: [1, 0],
        scale: [1, 0],
        duration: 180,
        ease: 'inQuad',
      });
    }
    isFirstRender.current = false;
    return () => animRef.current?.pause?.();
  }, [isVisible]);

  return (
    <span
      ref={checkRef}
      className="linear-node__check"
      style={{
        opacity: isVisible ? 1 : 0,
        transform: isVisible ? 'none' : 'scale(0)',
        pointerEvents: 'none',
      }}
    >
      ✓
    </span>
  );
}

/* ─── Smooth Node Live Progress Pill with Anime.js ─── */
function NodeLivePct({ isVisible, pct }) {
  const pctRef = useRef(null);
  const animRef = useRef(null);
  const numAnimRef = useRef(null);
  const isFirstRender = useRef(true);
  const numObj = useRef({ val: pct });
  const [displayVal, setDisplayVal] = useState(pct);

  useEffect(() => {
    numAnimRef.current?.pause?.();
    numAnimRef.current = animate(numObj.current, {
      val: pct,
      duration: 650,
      ease: 'outCubic',
      onUpdate: () => {
        setDisplayVal(Math.round(numObj.current.val));
      },
    });
    return () => numAnimRef.current?.pause?.();
  }, [pct]);

  useEffect(() => {
    if (!pctRef.current) return;
    animRef.current?.pause?.();

    if (isVisible) {
      animRef.current = animate(pctRef.current, {
        opacity: [0, 1],
        scale: [0.75, 1],
        translateY: [4, 0],
        duration: 300,
        ease: 'outBack(1.4)',
      });
    } else if (!isFirstRender.current) {
      animRef.current = animate(pctRef.current, {
        opacity: [1, 0],
        scale: [1, 0.75],
        duration: 180,
        ease: 'inQuad',
      });
    }
    isFirstRender.current = false;
    return () => animRef.current?.pause?.();
  }, [isVisible]);

  return (
    <div
      ref={pctRef}
      className="linear-node__live-pct"
      style={{
        opacity: isVisible ? 1 : 0,
        transform: isVisible ? 'none' : 'scale(0.75)',
        pointerEvents: 'none',
      }}
    >
      {displayVal}%
    </div>
  );
}

export default function PipelineFlow({
  currentStep = 0,
  isAnalyzing = false,
  completionPercentage = 0,
  statusText = '',
  completionText = '',
  detailText = '',
  telemetry = null,
  result = null,
  onReset = null,
}) {
  const [selectedId, setSelectedId] = useState(1);
  const [hasUserSelected, setHasUserSelected] = useState(false);
  const total = LINEAR_STEPS.length;
  const isFinished = currentStep >= total && !isAnalyzing;
  const isInitial = currentStep === 0 && !isAnalyzing && !isFinished;

  const [isExporting, setIsExporting] = useState(false);
  const [exportNotice, setExportNotice] = useState(null);

  // Smooth numerical count-up interpolation with Anime.js
  const pctObjRef = useRef({ val: 0 });
  const [displayPct, setDisplayPct] = useState(completionPercentage || 0);

  useEffect(() => {
    const target = completionPercentage || 0;
    const anim = animate(pctObjRef.current, {
      val: target,
      duration: 440,
      ease: 'outQuad',
      onUpdate: () => {
        setDisplayPct(Math.round(pctObjRef.current.val));
      },
    });
    return () => anim?.pause?.();
  }, [completionPercentage]);

  // Auto-switch visualizer to currently processing stage during live analysis, or stage 6 on finish
  useEffect(() => {
    if (isAnalyzing) {
      const derived =
        completionPercentage >= 84 ? 6 :
        completionPercentage >= 67 ? 5 :
        completionPercentage >= 50 ? 4 :
        completionPercentage >= 33 ? 3 :
        completionPercentage >= 16 ? 2 : 1;
      setSelectedId(currentStep > 0 ? currentStep : derived);
    } else if (isFinished && result) {
      setSelectedId(6);
    }
  }, [isAnalyzing, currentStep, completionPercentage, isFinished, result]);

  // Export Clinical Diagnostic Report as PDF and save directly to /reports folder
  const handleExportPDF = async () => {
    try {
      setIsExporting(true);
      const payload = result || {
        cardiac_age: {
          estimated_age_days: 214,
          lmp_ga_days: 210,
          delta_age_days: 4,
          estimated_age_weeks: 30.6,
          lmp_ga_weeks: 30.0,
          mae_days: 6.8,
          confidence_interval_95: [204, 228],
        },
      };

      const resp = await fetch('http://localhost:8000/api/export-pdf', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (!resp.ok) {
        throw new Error(`Export failed with HTTP ${resp.status}`);
      }

      const reportPath = resp.headers.get('X-Report-Path') || 'reports/Fetal_Cardiac_Diagnostic_Report.pdf';
      const filename = reportPath.split(/[/\\]/).pop() || 'Fetal_Cardiac_Diagnostic_Report.pdf';
      const blob = await resp.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(url);

      setExportNotice({
        type: 'success',
        msg: `PDF Report saved in /reports/ folder (${filename}) and downloaded!`,
      });
      setTimeout(() => setExportNotice(null), 6000);
    } catch (err) {
      console.error('PDF export failed:', err);
      setExportNotice({
        type: 'error',
        msg: 'PDF export failed. Make sure backend service is active.',
      });
      setTimeout(() => setExportNotice(null), 5000);
    } finally {
      setIsExporting(false);
    }
  };

  return (
    <div className={`linear-pipeline-card animate-in ${isAnalyzing ? 'linear-pipeline-card--analyzing' : ''} ${isInitial ? 'linear-pipeline-card--initial' : ''}`}>
      {/* ── Top Bar with Status and Top-Right Export as PDF Button ──── */}
      <div className="pipeline-top-bar">
        <div className="pipeline-top-status">
          <span className={`pipeline-status-dot ${isAnalyzing ? 'pipeline-status-dot--active' : isFinished ? 'pipeline-status-dot--done' : ''}`} />
          <span className="pipeline-top-title">
            {isAnalyzing
              ? `Processing DL Pipeline · ${displayPct}% Completed`
              : isFinished
              ? 'Fetal Cardiac Diagnostic Report · Ready'
              : 'Fetal Cardiac Ageing Clock Pipeline · Standby'}
          </span>
        </div>

        <button
          type="button"
          className="export-pdf-btn"
          onClick={handleExportPDF}
          disabled={isExporting}
          title="Export Clinical Diagnostic Report as PDF and store in reports folder"
        >
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
            <polyline points="14 2 14 8 20 8" />
            <line x1="12" y1="18" x2="12" y2="12" />
            <polyline points="9 15 12 18 15 15" />
          </svg>
          <span>{isExporting ? 'Exporting PDF...' : 'Export as PDF'}</span>
        </button>
      </div>

      {exportNotice && (
        <div className={`export-notice-toast export-notice-toast--${exportNotice.type}`}>
          <span className="export-notice-icon">{exportNotice.type === 'success' ? '✓' : '⚠️'}</span>
          <span>{exportNotice.msg}</span>
        </div>
      )}

      {/* ── Continuous Horizontal Flow with Waves Connecting Adjacent Squares ──── */}
      <div className="linear-track-wrapper">
        <div className="linear-nodes-row">
          {(() => {
            const STAGE_CONFIG = [
              { id: 1, start: 0,  mid: 10, end: 16 },
              { id: 2, start: 16, mid: 25, end: 33 },
              { id: 3, start: 33, mid: 42, end: 50 },
              { id: 4, start: 50, mid: 58, end: 67 },
              { id: 5, start: 67, mid: 78, end: 84 },
              { id: 6, start: 84, mid: 92, end: 100 },
            ];

            // Derive which stage is actively executing
            let activeProcessingStage = 1;
            if (isFinished) {
              activeProcessingStage = 6;
            } else if (isAnalyzing) {
              const matched = STAGE_CONFIG.find((cfg) => completionPercentage >= cfg.start && completionPercentage < cfg.end);
              activeProcessingStage = matched ? matched.id : (currentStep > 0 && currentStep <= 6 ? currentStep : 1);
            }

            // Active stage progress: 0% at start, 50% (Dead Center) at mid, 100% at end
            const currentCfg = STAGE_CONFIG[activeProcessingStage - 1] || STAGE_CONFIG[0];
            let activeStagePct = 50;
            if (isFinished) {
              activeStagePct = 100;
            } else if (isAnalyzing) {
              if (completionPercentage <= currentCfg.mid) {
                const ratio = Math.max(0, (completionPercentage - currentCfg.start) / Math.max(1, currentCfg.mid - currentCfg.start));
                activeStagePct = Math.round(ratio * 50);
              } else {
                const ratio = Math.max(0, (completionPercentage - currentCfg.mid) / Math.max(1, currentCfg.end - currentCfg.mid));
                activeStagePct = Math.round(50 + ratio * 50);
              }
              activeStagePct = Math.min(100, Math.max(15, activeStagePct));
            }

            return LINEAR_STEPS.map((step, idx) => {
              const cfg = STAGE_CONFIG[idx] || { start: 0, mid: 10, end: 16 };
              const isCompleted = isFinished || completionPercentage >= cfg.end;
              const isProcessing = isAnalyzing && (activeProcessingStage === step.id) && !isCompleted;
              const isSelected = !isInitial && selectedId === step.id;
              const isHighlighted = isProcessing || isSelected;
              const isPending = !isFinished && !isCompleted && !isProcessing;
              const isLastStep = step.id === 6;
              const StepIcon = step.Icon;

              // Connector idx (0 to 4) is located directly in FRONT of Stage (idx + 1).
              // Conduits behind the active stage are strictly 100% solid completed (no trailing droplet).
              // Only the conduit NEXT to the running stage flows forward into the next square.
              const isConnectorBehind = isFinished || completionPercentage >= cfg.end;
              const isConnectorCurrent = isAnalyzing && (activeProcessingStage === idx + 1);

              const connectorCompleted = isConnectorBehind;
              const connectorProcessing = !isFinished && isConnectorCurrent;

              // Forward streaming water percentage: 50% at midpoint, 100% at completion
              let segPct = 0;
              if (connectorCompleted) {
                segPct = 100;
              } else if (connectorProcessing) {
                segPct = activeStagePct;
              }

              return (
                <React.Fragment key={step.id}>
                  {/* 1. Square Node Process */}
                  <div
                    className={`linear-node ${isLastStep ? 'linear-node--last' : ''} ${isCompleted ? 'linear-node--completed' : ''} ${isProcessing ? 'linear-node--active linear-node--processing' : ''} ${isSelected ? 'linear-node--selected' : ''} ${isHighlighted ? 'linear-node--highlighted' : ''} ${isPending ? 'linear-node--pending' : ''}`}
                    onClick={() => {
                      setSelectedId(step.id);
                      setHasUserSelected(true);
                    }}
                    role="button"
                    tabIndex={0}
                    title={`Click to inspect Stage ${step.id}: ${step.title}`}
                  >
                    {/* Node square circle / icon container */}
                    <div
                      className={`linear-node__circle ${isLastStep ? 'linear-node__circle--last' : ''}`}
                      style={{ '--node-idx': step.id }}
                      title={`${step.title} — ${step.sub}`}
                    >
                      <span className="linear-node__icon">
                        <StepIcon
                          size={isLastStep ? (isHighlighted ? 32 : 28) : (isHighlighted ? 30 : 26)}
                          color={isHighlighted ? '#2563eb' : isCompleted ? '#1882b4' : '#1882b4'}
                        />
                      </span>

                      {/* Creative Precision Tech HUD Reticle with Anime.js smooth fade & scale */}
                      <CreativeTechFrame isVisible={isHighlighted} />

                      {/* Live real-time completion percentage pill on the active node */}
                      <NodeLivePct isVisible={isProcessing} pct={activeStagePct} />

                      {/* Completed checkmark badge (smoothly springs into view with Anime.js) */}
                      <NodeCheckmark isVisible={isCompleted && !isHighlighted} />
                    </div>

                    {/* Process Name & Subtitle below the square */}
                    <div className="linear-node__text">
                      <span className="linear-node__name">{step.title}</span>
                      <span className="linear-node__sub">{step.sub}</span>
                    </div>
                  </div>

                  {/* 2. Wave Connecting this Square to the Adjacent Next Square */}
                  {idx < total - 1 && (
                    <AdjacentWaveConnector
                      isCompleted={connectorCompleted}
                      isProcessing={connectorProcessing}
                      progressPct={segPct}
                    />
                  )}
                </React.Fragment>
              );
            });
          })()}
        </div>
      </div>

      {/* ── Active Stage Output Visualizer (visible when selected, analyzing, or finished) ──── */}
      {(!isInitial || hasUserSelected || isAnalyzing || isFinished || !!result) && (
        <StageVisualizer
          activeStep={selectedId}
          isAnalyzing={isAnalyzing && currentStep === selectedId}
          completionPercentage={completionPercentage}
          telemetry={telemetry}
          result={result}
          onReset={onReset}
        />
      )}
    </div>
  );
}
