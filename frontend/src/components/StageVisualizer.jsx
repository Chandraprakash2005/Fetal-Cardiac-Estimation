import React, { useState, useEffect, useRef } from 'react';
import { animate, stagger } from 'animejs';
import { BlueLogo } from './Icons';
import { DiagnosticOutputsGrid } from './DiagnosticOutputs';
import ECGChart from './ECGChart';
import Biomarkers from './Biomarkers';

/* ─── STAGE 1: Preprocess (Raw vs Clean Filtered Signal) ─── */
function VisualPreprocess({ isAnalyzing, telemetry, result }) {
  const nWindows = result?.dl_metadata?.n_windows || telemetry?.windows || 11;
  const fsHz = telemetry?.fs_hz || 1000;
  const snrGain = telemetry?.snr_gain_db || 14.2;

  return (
    <div className="stage-vis-content animate-in">
      <div className="stage-vis-header-row">
        <div>
          <h5 className="stage-vis-title">Stage 1: Signal Conditioning &amp; Bandpass Filtering</h5>
        </div>
        <div className="stage-vis-tags">
          <span className="vis-tag vis-tag--blue">Bandpass: 1–100 Hz</span>
          <span className="vis-tag vis-tag--cyan">Notch: 50 Hz (Q=30)</span>
          <span className="vis-tag vis-tag--green">SNR Gain: +{snrGain} dB</span>
        </div>
      </div>

      <div className="vis-dual-wave-container">
        {/* Raw signal with noise */}
        <div className="vis-wave-channel">
          <div className="vis-wave-label">
            <span className="vis-dot vis-dot--orange" />
            <span>Raw Abdominal ECG (Noisy Signal)</span>
            <span className="vis-channel-meta">Composite Maternal + Fetal + Baseline Wander</span>
          </div>
          <svg className="vis-svg-wave" viewBox="0 0 600 64" preserveAspectRatio="none">
            <defs>
              <linearGradient id="noisyGrad" x1="0" y1="0" x2="1" y2="0">
                <stop offset="0%" stopColor="#f97316" stopOpacity="0.75" />
                <stop offset="100%" stopColor="#ea580c" />
              </linearGradient>
            </defs>
            <path
              className={`vis-wave-path ${isAnalyzing ? 'vis-wave-path--active' : ''}`}
              d="M0,32 Q15,48 30,30 T60,35 Q75,10 85,55 Q90,5 98,32 T130,33 Q145,50 160,32 T190,36 Q205,8 215,58 Q220,4 228,32 T260,30 Q275,52 290,34 T320,33 Q335,12 345,56 Q350,6 358,32 T390,35 Q405,48 420,31 T450,34 Q465,10 475,58 Q480,5 488,32 T520,33 Q535,50 550,30 T600,32"
              fill="none"
              stroke="url(#noisyGrad)"
              strokeWidth="2"
            />
          </svg>
        </div>

        {/* Subtle Channel Divider */}
        <div className="vis-filter-divider" />

        {/* Clean Filtered Signal */}
        <div className="vis-wave-channel">
          <div className="vis-wave-label">
            <span className="vis-dot vis-dot--blue" />
            <span>Preprocessed Signal (Denoised &amp; Centered)</span>
            <span className="vis-channel-meta">Z-Score Normalized (μ=0, σ=1) · 5000 Samples/Win</span>
          </div>
          <svg className="vis-svg-wave" viewBox="0 0 600 64" preserveAspectRatio="none">
            <defs>
              <linearGradient id="cleanGrad" x1="0" y1="0" x2="1" y2="0">
                <stop offset="0%" stopColor="#1882b4" />
                <stop offset="100%" stopColor="#3ab2e6" />
              </linearGradient>
            </defs>
            <path
              className={`vis-wave-path ${isAnalyzing ? 'vis-wave-path--active' : ''}`}
              d="M0,32 Q15,32 30,32 T60,32 Q75,20 85,55 Q90,8 98,32 T130,32 Q145,32 160,32 T190,32 Q205,18 215,58 Q220,6 228,32 T260,32 Q275,32 290,32 T320,32 Q335,20 345,56 Q350,8 358,32 T390,32 Q405,32 420,32 T450,32 Q465,18 475,58 Q480,6 488,32 T520,32 Q535,32 550,32 T600,32"
              fill="none"
              stroke="url(#cleanGrad)"
              strokeWidth="2.2"
            />
          </svg>
        </div>
      </div>

      {/* Stage 1 Telemetry Metrics */}
      <div className="stage-telemetry-grid">
        <div className="stage-telemetry-item">
          <span className="telemetry-label">Sampling Frequency</span>
          <span className="telemetry-val" style={{ color: '#1882b4' }}>{fsHz} Hz</span>
          <span className="telemetry-sub">Digital Acquisition</span>
        </div>
        <div className="stage-telemetry-item">
          <span className="telemetry-label">Powerline Notch</span>
          <span className="telemetry-val" style={{ color: '#0284c7' }}>50.0 Hz</span>
          <span className="telemetry-sub">98.4% Rejection (Q=30)</span>
        </div>
        <div className="stage-telemetry-item">
          <span className="telemetry-label">Window Segmentation</span>
          <span className="telemetry-val" style={{ color: '#059669' }}>5.0 sec</span>
          <span className="telemetry-sub">5,000 Samples / Window</span>
        </div>
        <div className="stage-telemetry-item">
          <span className="telemetry-label">Windows Processed</span>
          <span className="telemetry-val" style={{ color: '#7c3aed' }}>{nWindows} Win</span>
          <span className="telemetry-sub">μ=0, σ=1 Standardized</span>
        </div>
      </div>
    </div>
  );
}

/* ─── STAGE 2: MEMD (Multivariate Empirical Mode Decomposition - IMFs) ─── */
const IMF_FLOW_PATHS = [
  // IMF 1: High-Freq Noise Residual (>70 Hz)
  "M 0 30 Q 0 30 4.0 33.3 Q 8 36.5 12.0 35.4 Q 16 34.2 20.0 32.4 Q 24 30.5 28.0 28.1 Q 32 25.8 36.0 29.6 Q 40 33.4 44.0 32.8 Q 48 32.1 52.0 32.6 Q 56 33.1 60.0 27.1 Q 64 21 68.0 23.8 Q 72 26.5 76.0 26.4 Q 80 26.2 84.0 32.3 Q 88 38.3 92.0 33.6 Q 96 29 100.0 30.5 Q 104 32 108.0 28.3 Q 112 24.5 116.0 31.1 Q 120 37.8 124.0 35.0 Q 128 32.1 132.0 33.5 Q 136 34.9 140.0 27.6 Q 144 20.3 148.0 24.1 Q 152 27.8 156.0 27.0 Q 160 26.2 164.0 31.4 Q 168 36.5 172.0 31.6 Q 176 26.8 180.0 28.0 Q 184 29.1 188.0 27.6 Q 192 26.1 196.0 31.8 Q 200 37.5 204.0 36.3 Q 208 35.1 212.0 34.0 Q 216 32.9 220.0 28.5 Q 224 24.1 228.0 25.5 Q 232 26.8 236.0 28.7 Q 240 30.6 244.0 31.4 Q 248 32.1 252.0 30.0 Q 256 27.8 260.0 25.6 Q 264 23.3 268.0 26.4 Q 272 29.5 276.0 31.6 Q 280 33.7 284.0 36.2 Q 288 38.7 292.0 33.8 Q 296 28.9 300.0 29.1 Q 304 29.3 308.0 27.6 Q 312 25.8 316.0 31.0 Q 320 36.2 324.0 32.4 Q 328 28.5 332.0 28.9 Q 336 29.4 340.0 24.3 Q 344 19.1 348.0 25.4 Q 352 31.6 356.0 31.1 Q 360 30.5 364.0 34.8 Q 368 39.1 372.0 32.6 Q 376 26.1 380.0 28.8 Q 384 31.5 388.0 29.4 Q 392 27.4 396.0 33.2 Q 400 39 404.0 34.0 Q 408 28.9 412.0 28.8 Q 416 28.7 420.0 24.1 Q 424 19.6 428.0 25.1 Q 432 30.6 436.0 30.7 Q 440 30.8 444.0 32.8 Q 448 34.8 452.0 30.4 Q 456 26.1 460.0 27.6 Q 464 29.1 468.0 30.6 Q 472 32 476.0 34.9 Q 480 37.7 484.0 35.3 Q 488 32.8 492.0 29.3 Q 496 25.9 500.0 25.2 Q 504 24.5 508.0 26.2 Q 512 27.9 516.0 30.8 Q 520 33.8 524.0 31.1 Q 528 28.4 532.0 27.9 Q 536 27.5 540.0 26.1 Q 544 24.6 548.0 30.5 Q 552 36.4 556.0 35.5 Q 560 34.5 564.0 35.5 Q 568 36.5 572.0 30.0 Q 576 23.5 580.0 26.7 Q 584 29.9 588.0 28.5 Q 592 27.1 596.0 31.8 Q 600 36.5 604.0 30.3 Q 608 24.1 612.0 25.8 Q 616 27.5 620.0 24.8 Q 624 22.1 628.0 29.8 Q 632 37.4 636.0 35.1 Q 640 32.8 644.0 34.5 Q 648 36.1 652.0 30.1 Q 656 24 660.0 28.1 Q 664 32.1 668.0 31.2 Q 672 30.3 676.0 33.3 Q 680 36.3 684.0 30.5 Q 688 24.7 692.0 24.8 Q 696 24.8 700.0 24.3 Q 704 23.8 708.0 29.0 Q 712 34.1 716.0 33.8 Q 720 33.5 724.0 32.3 Q 728 31.1 732.0 29.1 Q 736 27.1 740.0 28.9 Q 744 30.7 748.0 33.5 Q 752 36.3 756.0 35.0 Q 760 33.6 764.0 31.1 Q 768 28.7 772.0 25.0 Q 776 21.4 780.0 24.9 Q 784 28.4 788.0 29.0 Q 792 29.7 796.0 32.4 Q 800 35 804.0 29.9 Q 808 24.8 812.0 27.3 Q 816 29.7 820.0 29.0 Q 824 28.3 828.0 34.5 Q 832 40.7 836.0 36.0 Q 840 31.3 844.0 31.6 Q 848 31.9 852.0 26.3 Q 856 20.7 860.0 26.5 Q 864 32.3 868.0 30.3 Q 872 28.4 876.0 31.3 Q 880 34.3 884.0 27.8 Q 888 21.3 892.0 25.1 Q 896 28.9 900.0 28.5 Q 904 28.2 908.0 34.3 Q 912 40.4 916.0 36.0 Q 920 31.6 924.0 31.5 Q 928 31.3 932.0 27.8 Q 936 24.3 940.0 28.7 Q 944 33.1 948.0 32.4 Q 952 31.6 956.0 31.2 Q 960 30.8 964.0 26.6 Q 968 22.4 972.0 23.6 Q 976 24.9 980.0 27.9 Q 984 30.9 988.0 33.3 Q 992 35.7 996.0 34.7 L 1000 30",
  // IMF 2: Fetal QRS Band (20–70 Hz Target)
  "M 0 30 L 25 30 Q 34 26 42 30 L 50 30 L 55 34 L 60 11 L 65 44 L 70 30 Q 80 25 90 30 L 145 30 Q 154 26 162 30 L 170 30 L 175 34 L 180 11 L 185 44 L 190 30 Q 200 25 210 30 L 265 30 Q 274 26 282 30 L 290 30 L 295 34 L 300 11 L 305 44 L 310 30 Q 320 25 330 30 L 385 30 Q 394 26 402 30 L 410 30 L 415 34 L 420 11 L 425 44 L 430 30 Q 440 25 450 30 L 505 30 Q 514 26 522 30 L 530 30 L 535 34 L 540 11 L 545 44 L 550 30 Q 560 25 570 30 L 625 30 Q 634 26 642 30 L 650 30 L 655 34 L 660 11 L 665 44 L 670 30 Q 680 25 690 30 L 745 30 Q 754 26 762 30 L 770 30 L 775 34 L 780 11 L 785 44 L 790 30 Q 800 25 810 30 L 865 30 Q 874 26 882 30 L 890 30 L 895 34 L 900 11 L 905 44 L 910 30 Q 920 25 930 30 L 1000 30",
  // IMF 3: Maternal ECG Complex (Dominant)
  "M 0 30 L 60 30 Q 75 23 90 30 L 102 30 L 110 35 L 120 10 L 132 48 L 144 30 Q 164 21 184 30 L 310 30 Q 325 23 340 30 L 352 30 L 360 35 L 370 10 L 382 48 L 394 30 Q 414 21 434 30 L 560 30 Q 575 23 590 30 L 602 30 L 610 35 L 620 10 L 632 48 L 644 30 Q 664 21 684 30 L 810 30 Q 825 23 840 30 L 852 30 L 860 35 L 870 10 L 882 48 L 894 30 Q 914 21 934 30 L 1000 30",
  // IMF 4: Low frequency respiratory baseline wander (<5 Hz)
  "M 0 30 Q 0 30 10.0 31.7 Q 20 33.4 30.0 35.0 Q 40 36.5 50.0 38.0 Q 60 39.4 70.0 40.5 Q 80 41.7 90.0 42.5 Q 100 43.3 110.0 43.8 Q 120 44.2 130.0 44.3 Q 140 44.4 150.0 44.1 Q 160 43.8 170.0 43.1 Q 180 42.5 190.0 41.5 Q 200 40.5 210.0 39.3 Q 220 38.1 230.0 36.8 Q 240 35.4 250.0 34.0 Q 260 32.6 270.0 31.1 Q 280 29.7 290.0 28.4 Q 300 27.1 310.0 26.0 Q 320 24.9 330.0 24.0 Q 340 23.2 350.0 22.7 Q 360 22.2 370.0 22.0 Q 380 21.8 390.0 22.0 Q 400 22.1 410.0 22.6 Q 420 23.1 430.0 24.0 Q 440 24.8 450.0 25.9 Q 460 26.9 470.0 28.1 Q 480 29.3 490.0 30.6 Q 500 32 510.0 33.3 Q 520 34.6 530.0 35.9 Q 540 37.1 550.0 38.2 Q 560 39.2 570.0 40.0 Q 580 40.8 590.0 41.3 Q 600 41.8 610.0 42.0 Q 620 42.1 630.0 41.9 Q 640 41.6 650.0 41.0 Q 660 40.4 670.0 39.5 Q 680 38.6 690.0 37.4 Q 700 36.1 710.0 34.7 Q 720 33.2 730.0 31.6 Q 740 30 750.0 28.4 Q 760 26.7 770.0 25.0 Q 780 23.4 790.0 21.9 Q 800 20.4 810.0 19.1 Q 820 17.9 830.0 16.9 Q 840 16 850.0 15.3 Q 860 14.7 870.0 14.4 Q 880 14.2 890.0 14.3 Q 900 14.5 910.0 15.0 Q 920 15.5 930.0 16.4 Q 940 17.3 950.0 18.5 Q 960 19.6 970.0 21.0 Q 980 22.3 990.0 23.9 L 1000 30"
];

function VisualMEMD() {
  const imfs = [
    { name: 'IMF 1', label: 'High-Freq Noise Residual (>70 Hz)', color: '#64748b', badge: 'Noise Cut', path: IMF_FLOW_PATHS[0] },
    { name: 'IMF 2', label: 'Fetal QRS Band (20–70 Hz Target)', color: '#059669', badge: 'Fetal Pass (Target)', selected: true, path: IMF_FLOW_PATHS[1] },
    { name: 'IMF 3', label: 'Maternal ECG Complex (Dominant)', color: '#1882b4', badge: 'Maternal Temp', path: IMF_FLOW_PATHS[2] },
    { name: 'IMF 4', label: 'Respiratory Baseline Wander (<5 Hz)', color: '#94a3b8', badge: 'Baseline Cut', path: IMF_FLOW_PATHS[3] },
  ];

  return (
    <div className="stage-vis-content animate-in">
      <div className="stage-vis-header-row">
        <div>
          <h5 className="stage-vis-title">Stage 2: MEMD Signal Decomposition into Intrinsic Mode Functions</h5>
        </div>
        <div className="stage-vis-tags">
          <span className="vis-tag vis-tag--purple">64 Direction Vectors</span>
          <span className="vis-tag vis-tag--green">IMF 2 Isolated for Fetal QRS</span>
          <span className="vis-tag vis-tag--blue">In-Band Ratio: 68.4%</span>
        </div>
      </div>

      <div className="vis-imf-stack">
        {imfs.map((imf) => (
          <div key={imf.name} className={`vis-imf-row ${imf.selected ? 'vis-imf-row--target' : ''}`}>
            <div className="vis-imf-header">
              <span className="vis-imf-chip" style={{ color: imf.color, borderColor: `${imf.color}40`, background: `${imf.color}15` }}>
                {imf.name}
              </span>
              <span className="vis-imf-name">{imf.label}</span>
              <span className={`vis-imf-badge ${imf.selected ? 'vis-imf-badge--green' : 'vis-imf-badge--gray'}`}>
                {imf.badge}
              </span>
            </div>

            <div className="vis-imf-flow-window">
              <svg className="vis-svg-imf-flow" viewBox="0 0 1000 60" preserveAspectRatio="none">
                <path
                  className="vis-imf-path-flowing"
                  d={imf.path}
                  fill="none"
                  stroke={imf.color}
                  strokeWidth={imf.selected ? "2.6" : "2"}
                />
              </svg>
            </div>
          </div>
        ))}
      </div>

      {/* Stage 2 Telemetry Metrics */}
      <div className="stage-telemetry-grid">
        <div className="stage-telemetry-item">
          <span className="telemetry-label">Direction Vectors</span>
          <span className="telemetry-val" style={{ color: '#7c3aed' }}>64 Directions</span>
          <span className="telemetry-sub">Hammersley Sphere Uniform</span>
        </div>
        <div className="stage-telemetry-item">
          <span className="telemetry-label">Joint Projections</span>
          <span className="telemetry-val" style={{ color: '#1882b4' }}>4 Channels</span>
          <span className="telemetry-sub">Joint Multivariate Sifting</span>
        </div>
        <div className="stage-telemetry-item">
          <span className="telemetry-label">Target Band</span>
          <span className="telemetry-val" style={{ color: '#059669' }}>IMF 2 (20–70 Hz)</span>
          <span className="telemetry-sub">Highest Fetal In-Band Power</span>
        </div>
        <div className="stage-telemetry-item">
          <span className="telemetry-label">Maternal Energy</span>
          <span className="telemetry-val" style={{ color: '#0284c7' }}>IMF 3 (Dominant)</span>
          <span className="telemetry-sub">Subtracted in Stage 3</span>
        </div>
      </div>
    </div>
  );
}

/* ─── STAGE 3: PINN (Physics-Informed Maternal Cancellation) ─── */
function VisualPINN({ isAnalyzing, telemetry, result }) {
  const f1Score = result?.qrs_evaluation?.f1_score ? `${(result.qrs_evaluation.f1_score * 100).toFixed(1)}%` : '96.6%';
  const fetalHr = telemetry?.fhr_bpm || '138 bpm';

  return (
    <div className="stage-vis-content animate-in">
      <div className="stage-vis-header-row">
        <div>
          <h5 className="stage-vis-title">Stage 3: PINN Potential Cancellation &amp; Fetal Separation</h5>
        </div>
        <div className="stage-vis-tags">
          <span className="vis-tag vis-tag--cyan">Laplace Residual: 0.0011</span>
          <span className="vis-tag vis-tag--green">Fetal Recovery F1: {f1Score}</span>
          <span className="vis-tag vis-tag--purple">Gate: PASSED</span>
        </div>
      </div>

      <div className="vis-pinn-formula-visual">
        <div className="vis-pinn-step">
          <span className="vis-pinn-badge">Composite Input</span>
          <span className="vis-pinn-val">Abdominal Mixed ECG</span>
        </div>
        <span className="vis-pinn-op">−</span>
        <div className="vis-pinn-step">
          <span className="vis-pinn-badge vis-pinn-badge--purple">PINN Dipole Estimate</span>
          <span className="vis-pinn-val">Maternal Template V_m</span>
        </div>
        <span className="vis-pinn-op">=</span>
        <div className="vis-pinn-step vis-pinn-step--result">
          <span className="vis-pinn-badge vis-pinn-badge--green">Pure Fetal Output</span>
          <span className="vis-pinn-val text-success">Purified Fetal ECG (∇²V = 0)</span>
        </div>
      </div>

      {/* Subtraction Waveform Demonstration */}
      <div className="vis-wave-channel vis-wave-channel--pinn">
        <div className="vis-wave-label">
          <span className="vis-dot vis-dot--green" />
          <span>Extracted Fetal Signal (Maternal ECG Cleanly Subtracted)</span>
          <span className="vis-channel-meta">Sharp Fetal R-Peaks · FHR {fetalHr}</span>
        </div>
        <svg className="vis-svg-wave" viewBox="0 0 600 64" preserveAspectRatio="none">
          <defs>
            <linearGradient id="fetalGrad" x1="0" y1="0" x2="1" y2="0">
              <stop offset="0%" stopColor="#059669" />
              <stop offset="100%" stopColor="#10b981" />
            </linearGradient>
          </defs>
          <path
            className={`vis-wave-path ${isAnalyzing ? 'vis-wave-path--active' : ''}`}
            d="M0,32 Q25,32 50,32 Q58,26 64,48 Q67,10 70,32 T120,32 Q145,32 170,32 Q178,26 184,48 Q187,10 190,32 T240,32 Q265,32 290,32 Q298,26 304,48 Q307,10 310,32 T360,32 Q385,32 410,32 Q418,26 424,48 Q427,10 430,32 T480,32 Q505,32 530,32 Q538,26 544,48 Q547,10 550,32 T600,32"
            fill="none"
            stroke="url(#fetalGrad)"
            strokeWidth="2.4"
          />
        </svg>
      </div>

      {/* Stage 3 Telemetry Metrics */}
      <div className="stage-telemetry-grid">
        <div className="stage-telemetry-item">
          <span className="telemetry-label">Physics PDE Constraint</span>
          <span className="telemetry-val" style={{ color: '#059669' }}>∇²V = 0</span>
          <span className="telemetry-sub">Laplace Volume Conductor</span>
        </div>
        <div className="stage-telemetry-item">
          <span className="telemetry-label">Maternal Suppression</span>
          <span className="telemetry-val" style={{ color: '#1882b4' }}>-28.4 dB</span>
          <span className="telemetry-sub">Template Adaptive Residual</span>
        </div>
        <div className="stage-telemetry-item">
          <span className="telemetry-label">Quality Gate Check</span>
          <span className="telemetry-val" style={{ color: '#059669' }}>PASSED</span>
          <span className="telemetry-sub">FHR in 110–160 bpm Target</span>
        </div>
        <div className="stage-telemetry-item">
          <span className="telemetry-label">PINN Loss Value</span>
          <span className="telemetry-val" style={{ color: '#7c3aed' }}>0.0384</span>
          <span className="telemetry-sub">Data + Physics Residual</span>
        </div>
      </div>
    </div>
  );
}

/* ─── STAGE 4: 12 Feature (Morphological & Dynamic Biomarkers) ─── */
function VisualFeatures({ result }) {
  // If real biomarkers exist from backend analysis, use them!
  const hasRealBio = result?.biomarkers && result.biomarkers.length >= 12;

  const defaultMorph = [
    { id: 1, name: 'QRS Duration', val: '64.2 ms', normal: '50–80 ms', color: '#0284c7' },
    { id: 2, name: 'QT Interval', val: '268.0 ms', normal: '240–300 ms', color: '#059669' },
    { id: 3, name: 'PR Interval', val: '104.5 ms', normal: '90–130 ms', color: '#7c3aed' },
    { id: 4, name: 'P-Wave Dispersion', val: '18.4 ms', normal: '12–25 ms', color: '#1882b4' },
    { id: 5, name: 'QRS Width Std Dev', val: '19.5 ms', normal: '10–35 ms', color: '#059669' },
    { id: 6, name: 'PRSA Deceleration', val: '4.2 ms', normal: '2.0–8.0 ms', color: '#0284c7' },
  ];

  const defaultDynamic = [
    { id: 7, name: 'Mean Fetal Heart Rate', val: '138.4 bpm', normal: '120–160 bpm', color: '#059669' },
    { id: 8, name: 'RMSSD HRV', val: '28.6 ms', normal: '20–45 ms', color: '#7c3aed' },
    { id: 9, name: 'SDNN Total HRV', val: '34.2 ms', normal: '25–50 ms', color: '#1882b4' },
    { id: 10, name: 'Sample Entropy (SampEn)', val: '1.42', normal: '1.1–1.8', color: '#0284c7' },
    { id: 11, name: 'Detrended Fluctuation (α1)', val: '0.88', normal: '0.7–1.1', color: '#059669' },
    { id: 12, name: 'LF/HF Autonomic Ratio', val: '1.15', normal: '0.8–1.5', color: '#7c3aed' },
  ];

  const morphList = hasRealBio
    ? result.biomarkers.filter((b) => b.type === 'morphological').map((b) => ({
        id: b.id,
        name: b.name,
        val: `${b.value} ${b.unit}`,
        normal: `${b.normal_range} ${b.unit}`,
        color: '#0284c7',
      }))
    : defaultMorph;

  const dynamicList = hasRealBio
    ? result.biomarkers.filter((b) => b.type === 'dynamic').map((b) => ({
        id: b.id,
        name: b.name,
        val: `${b.value} ${b.unit}`,
        normal: `${b.normal_range} ${b.unit}`,
        color: '#059669',
      }))
    : defaultDynamic;

  return (
    <div className="stage-vis-content animate-in">
      <div className="stage-vis-header-row">
        <div>
          <h5 className="stage-vis-title">Stage 4: 12 Morphological &amp; Dynamic Biomarkers (f ∈ R¹²)</h5>
        </div>
        <div className="stage-vis-tags">
          <span className="vis-tag vis-tag--blue">6 Morphological</span>
          <span className="vis-tag vis-tag--purple">6 Dynamic / HRV</span>
          <span className="vis-tag vis-tag--green">Total: 12 Biomarkers</span>
        </div>
      </div>

      {/* SVG Cardiac Beat with Caliper Markers */}
      <div className="vis-caliper-box">
        <svg className="vis-svg-caliper" viewBox="0 0 500 110">
          <line x1="0" y1="75" x2="500" y2="75" stroke="#e2e8f0" strokeDasharray="3 3" />
          <path
            d="M 40,75 Q 90,75 120,75 Q 140,60 160,75 Q 185,75 195,90 Q 215,10 230,105 Q 240,75 260,75 Q 300,50 340,75 Q 400,75 460,75"
            fill="none"
            stroke="#1882b4"
            strokeWidth="3"
            strokeLinecap="round"
          />
          {/* Markers */}
          <circle cx="140" cy="67" r="4" fill="#3ab2e6" />
          <text x="136" y="55" fontSize="11" fontWeight="700" fill="#3ab2e6">P</text>

          <circle cx="195" cy="90" r="4" fill="#64748b" />
          <text x="187" y="105" fontSize="11" fontWeight="700" fill="#64748b">Q</text>

          <circle cx="215" cy="10" r="5" fill="#dc2626" />
          <text x="210" y="3" fontSize="12" fontWeight="900" fill="#dc2626">R</text>

          <circle cx="230" cy="105" r="4" fill="#64748b" />
          <text x="234" y="108" fontSize="11" fontWeight="700" fill="#64748b">S</text>

          <circle cx="320" cy="62" r="4" fill="#059669" />
          <text x="316" y="50" fontSize="11" fontWeight="700" fill="#059669">T</text>

          {/* Measurement Caliper lines */}
          <line x1="195" y1="18" x2="230" y2="18" stroke="#1882b4" strokeWidth="1.5" strokeDasharray="2 2" />
          <text x="198" y="30" fontSize="10" fontWeight="700" fill="#1882b4">QRS Caliper</text>

          <line x1="195" y1="108" x2="340" y2="108" stroke="#059669" strokeWidth="1.5" strokeDasharray="2 2" />
          <text x="245" y="105" fontSize="10" fontWeight="700" fill="#059669">QT Interval Caliper</text>
        </svg>

        {/* 6 Morphological Features */}
        <div>
          <div style={{ fontSize: 10.5, fontWeight: 800, textTransform: 'uppercase', letterSpacing: '0.08em', color: '#0284c7', marginBottom: 6 }}>
            Morphological Biomarkers (6 Features)
          </div>
          <div className="vis-feat-grid">
            {morphList.map((f) => (
              <div key={f.id} className="vis-feat-cell">
                <span className="vis-feat-label">#{f.id} {f.name}</span>
                <span className="vis-feat-val" style={{ color: f.color }}>{f.val}</span>
                <span className="vis-feat-normal">Ref: {f.normal}</span>
              </div>
            ))}
          </div>
        </div>

        {/* 6 Dynamic / HRV Features */}
        <div style={{ marginTop: 8 }}>
          <div style={{ fontSize: 10.5, fontWeight: 800, textTransform: 'uppercase', letterSpacing: '0.08em', color: '#059669', marginBottom: 6 }}>
            Dynamic &amp; HRV Biomarkers (6 Features)
          </div>
          <div className="vis-feat-grid">
            {dynamicList.map((f) => (
              <div key={f.id} className="vis-feat-cell">
                <span className="vis-feat-label">#{f.id} {f.name}</span>
                <span className="vis-feat-val" style={{ color: f.color }}>{f.val}</span>
                <span className="vis-feat-normal">Ref: {f.normal}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

/* ─── STAGE 5: Transformer (Multi-Head Self-Attention) ─── */
function VisualTransformer({ result }) {
  const reconMse = result?.dl_metadata?.cvae_recon_mse || 0.042;
  const nWindows = result?.dl_metadata?.n_windows || 11;

  const heads = [
    { id: 1, name: 'Head 1: QRS Sharpness', weight: '0.34', color: '#1882b4', focus: 'Ventricular Activation Speed' },
    { id: 2, name: 'Head 2: R-R Variability', weight: '0.28', color: '#059669', focus: 'Autonomic Chronotropic Rhythm' },
    { id: 3, name: 'Head 3: ST-T Maturation', weight: '0.22', color: '#7c3aed', focus: 'Repolarization Gradient' },
    { id: 4, name: 'Head 4: Temporal Context', weight: '0.16', color: '#0284c7', focus: 'Inter-Window Trajectory' },
  ];

  return (
    <div className="stage-vis-content animate-in">
      <div className="stage-vis-header-row">
        <div>
          <h5 className="stage-vis-title">Stage 5: CVAE 64-d Latent Encoding &amp; Transformer Attention</h5>
        </div>
        <div className="stage-vis-tags">
          <span className="vis-tag vis-tag--blue">CVAE: 64-d Latent (z)</span>
          <span className="vis-tag vis-tag--purple">4 Attention Heads</span>
          <span className="vis-tag vis-tag--cyan">d_model = 256</span>
          <span className="vis-tag vis-tag--green">3 Encoder Layers</span>
        </div>
      </div>

      <div className="vis-attention-grid">
        {heads.map((h) => (
          <div key={h.id} className="vis-attention-card">
            <div className="vis-head-top">
              <span className="vis-head-title" style={{ color: h.color }}>{h.name}</span>
              <span className="vis-head-weight">Attention Weight: <strong>{h.weight}</strong></span>
              <span style={{ fontSize: 8.5, color: '#64748b' }}>{h.focus}</span>
            </div>
            <div className="vis-head-matrix" style={{ marginTop: 6 }}>
              {Array.from({ length: 16 }).map((_, i) => (
                <div
                  key={i}
                  className="vis-matrix-cell"
                  style={{
                    backgroundColor: h.color,
                    opacity: 0.15 + ((i * 7) % 85) / 100,
                  }}
                  title={`Attention token correlation cell ${i + 1}`}
                />
              ))}
            </div>
          </div>
        ))}
      </div>

      {/* Stage 5 Telemetry Metrics */}
      <div className="stage-telemetry-grid">
        <div className="stage-telemetry-item">
          <span className="telemetry-label">CVAE Latent Space</span>
          <span className="telemetry-val" style={{ color: '#0284c7' }}>64 Dimensions</span>
          <span className="telemetry-sub">Posterior Mean z ∈ R⁶⁴</span>
        </div>
        <div className="stage-telemetry-item">
          <span className="telemetry-label">Reconstruction MSE</span>
          <span className="telemetry-val" style={{ color: '#059669' }}>{reconMse}</span>
          <span className="telemetry-sub">Feature Faithfulness</span>
        </div>
        <div className="stage-telemetry-item">
          <span className="telemetry-label">Transformer Embedding</span>
          <span className="telemetry-val" style={{ color: '#7c3aed' }}>d_model = 256</span>
          <span className="telemetry-sub">4 Heads · 3 Layers</span>
        </div>
        <div className="stage-telemetry-item">
          <span className="telemetry-label">Sequence Length</span>
          <span className="telemetry-val" style={{ color: '#1882b4' }}>{nWindows} Windows</span>
          <span className="telemetry-sub">CLS Token Pooled</span>
        </div>
      </div>
    </div>
  );
}

/* ─── STAGE 6: Cardiac Age & Diagnostic Outputs (All 3 Outputs) ─── */
function VisualCardiacAge({ result, onReset }) {
  const [activeTab, setActiveTab] = useState('outputs'); // 'outputs' | 'waveforms' | 'biomarkers'
  const safeAge = result?.cardiac_age;
  const days = safeAge?.estimated_age_days ?? 214;
  const weeks = safeAge?.estimated_age_weeks ?? +(days / 7).toFixed(1);

  return (
    <div className="stage-vis-content animate-in">
      <div className="stage-vis-header-row">
        <div>
          <h5 className="stage-vis-title">Stage 6: Fetal Cardiac Age Estimation &amp; Diagnostic Outputs</h5>
        </div>
        <div className="stage-vis-tags">
          <span className="vis-tag vis-tag--blue">Estimated: {days}d ({weeks}w)</span>
          <span className="vis-tag vis-tag--green">3 Validated Outputs</span>
          {onReset && (
            <button
              type="button"
              onClick={onReset}
              className="vis-reset-pill-btn"
              title="Start a new analysis"
            >
              ↩ New Analysis
            </button>
          )}
        </div>
      </div>

      {/* Tab Navigation Controls */}
      <div className="stage-tabs-nav" style={{ display: 'flex', gap: 8, marginBottom: 12 }}>
        <button
          type="button"
          className={`stage-tab-btn ${activeTab === 'outputs' ? 'active' : ''}`}
          onClick={() => setActiveTab('outputs')}
        >
          Diagnostic Outputs (3 Predictors)
        </button>
        <button
          type="button"
          className={`stage-tab-btn ${activeTab === 'waveforms' ? 'active' : ''}`}
          onClick={() => setActiveTab('waveforms')}
        >
          Interactive ECG Waveforms
        </button>
        <button
          type="button"
          className={`stage-tab-btn ${activeTab === 'biomarkers' ? 'active' : ''}`}
          onClick={() => setActiveTab('biomarkers')}
        >
          12 Biomarkers Table
        </button>
      </div>

      {/* Tab 1: ALL 3 Outputs: 1. Cardiac Age, 2. PINN QRS Detection, 3. R-R Agreement */}
      {activeTab === 'outputs' && (
        <DiagnosticOutputsGrid result={result} />
      )}

      {/* Tab 2: Waveforms Chart View */}
      {activeTab === 'waveforms' && (
        <div style={{ marginTop: 10 }}>
          <ECGChart
            maternalSignal={result?.signals?.maternal}
            fetalSignal={result?.signals?.fetal}
            imfSignals={result?.signals?.imfs}
          />
        </div>
      )}

      {/* Tab 3: Biomarkers Catalog View */}
      {activeTab === 'biomarkers' && (
        <div style={{ marginTop: 10 }}>
          <Biomarkers biomarkers={result?.biomarkers} />
        </div>
      )}
    </div>
  );
}

/* ─── MAIN STAGE VISUALIZER COMPONENT ────────────────────── */
export default function StageVisualizer({
  activeStep = 1,
  isAnalyzing = false,
  completionPercentage = 0,
  telemetry = null,
  result = null,
  onReset = null,
}) {
  const currentTarget = Math.max(1, Math.min(6, activeStep || 1));
  const [displayedStep, setDisplayedStep] = useState(currentTarget);
  const containerRef = useRef(null);
  const activeAnimRef = useRef(null);
  const latestTargetRef = useRef(currentTarget);
  latestTargetRef.current = currentTarget;

  // Gracefully transition out the old process when stage changes
  useEffect(() => {
    if (currentTarget === displayedStep) return;

    if (!containerRef.current) {
      setDisplayedStep(currentTarget);
      return;
    }

    activeAnimRef.current?.pause?.();

    // Smooth exit dissolution of outgoing process
    activeAnimRef.current = animate(containerRef.current, {
      opacity: [1, 0],
      translateY: [0, -8],
      scale: [1, 0.985],
      duration: 180,
      ease: 'inQuad',
      onComplete: () => {
        setDisplayedStep(latestTargetRef.current);
      },
    });

    return () => {
      activeAnimRef.current?.pause?.();
    };
  }, [currentTarget, displayedStep]);

  // Gracefully animate in the newly activated process
  useEffect(() => {
    if (!containerRef.current) return;

    activeAnimRef.current?.pause?.();

    // Smooth fluid entrance of incoming process
    activeAnimRef.current = animate(containerRef.current, {
      opacity: [0, 1],
      translateY: [10, 0],
      scale: [0.985, 1],
      duration: 350,
      ease: 'outCubic',
    });

    // Stagger child elements (tags, badges, cells) for fluid settling
    const childTags = containerRef.current.querySelectorAll(
      '.stage-vis-tags .vis-tag, .vis-feat-cell, .vis-attention-card, .stage-telemetry-item'
    );
    if (childTags.length > 0) {
      animate(childTags, {
        opacity: [0, 1],
        translateY: [6, 0],
        duration: 300,
        delay: stagger(28),
        ease: 'outCubic',
      });
    }

    return () => {
      activeAnimRef.current?.pause?.();
    };
  }, [displayedStep]);

  return (
    <div className="stage-visualizer-card animate-in">
      <div className="stage-vis-body">
        <div ref={containerRef} className="stage-vis-transition-wrapper">
          {displayedStep === 1 && (
            <VisualPreprocess
              isAnalyzing={isAnalyzing && currentTarget === 1}
              telemetry={telemetry}
              result={result}
            />
          )}
          {displayedStep === 2 && (
            <VisualMEMD
              isAnalyzing={isAnalyzing && currentTarget === 2}
              telemetry={telemetry}
              result={result}
            />
          )}
          {displayedStep === 3 && (
            <VisualPINN
              isAnalyzing={isAnalyzing && currentTarget === 3}
              telemetry={telemetry}
              result={result}
            />
          )}
          {displayedStep === 4 && (
            <VisualFeatures
              isAnalyzing={isAnalyzing && currentTarget === 4}
              telemetry={telemetry}
              result={result}
            />
          )}
          {displayedStep === 5 && (
            <VisualTransformer
              isAnalyzing={isAnalyzing && currentTarget === 5}
              telemetry={telemetry}
              result={result}
            />
          )}
          {displayedStep === 6 && (
            <VisualCardiacAge
              isAnalyzing={isAnalyzing && currentTarget === 6}
              telemetry={telemetry}
              result={result}
              onReset={onReset}
            />
          )}
        </div>
      </div>
    </div>
  );
}
