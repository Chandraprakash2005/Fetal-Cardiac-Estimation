import React, { useState, useEffect, useRef } from 'react';
import { animate, stagger } from 'animejs';
import { BlueLogo } from './Icons';

/* ─── STAGE 1: Preprocess (Raw vs Clean Filtered Signal) ─── */
function VisualPreprocess({ isAnalyzing }) {
  return (
    <div className="stage-vis-content animate-in">
      <div className="stage-vis-header-row">
        <div>
          <h5 className="stage-vis-title">Stage 1: Signal Conditioning &amp; Bandpass Filtering</h5>
          <p className="stage-vis-sub">1–100 Hz 4th-order Butterworth bandpass filter + 50 Hz notch filter removing maternal muscle noise &amp; baseline drift</p>
        </div>
        <div className="stage-vis-tags">
          <span className="vis-tag vis-tag--blue">Bandpass: 1–100 Hz</span>
          <span className="vis-tag vis-tag--green">SNR: +14.2 dB</span>
        </div>
      </div>

      <div className="vis-dual-wave-container">
        {/* Raw signal with noise */}
        <div className="vis-wave-channel">
          <div className="vis-wave-label">
            <span className="vis-dot vis-dot--orange" />
            <span>Raw Abdominal ECG (Noisy Signal)</span>
            <span className="vis-channel-meta">Maternal noise + Respiration wander</span>
          </div>
          <svg className="vis-svg-wave" viewBox="0 0 600 64" preserveAspectRatio="none">
            <defs>
              <linearGradient id="noisyGrad" x1="0" y1="0" x2="1" y2="0">
                <stop offset="0%" stopColor="#f97316" stopOpacity="0.7" />
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

        {/* Animated Filter Scan Bar */}
        <div className="vis-filter-scanner">
          <span className="vis-scanner-beam" />
          <span className="vis-scanner-text">Filtering: Zero-phase Forward &amp; Backward Filter</span>
        </div>

        {/* Clean Filtered Signal */}
        <div className="vis-wave-channel">
          <div className="vis-wave-label">
            <span className="vis-dot vis-dot--blue" />
            <span>Preprocessed Signal (Denoised &amp; Centered)</span>
            <span className="vis-channel-meta">Z-Score Normalized (μ=0, σ=1)</span>
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
    </div>
  );
}

/* ─── STAGE 2: MEMD (Multivariate Empirical Mode Decomposition - IMFs) ─── */
const IMF_FLOW_PATHS = [
  // IMF 1: High-Freq Noise Residual (>70 Hz) - Continuous smooth high-frequency oscillations (y: 19..41)
  "M 0 30 Q 0 30 4.0 33.3 Q 8 36.5 12.0 35.4 Q 16 34.2 20.0 32.4 Q 24 30.5 28.0 28.1 Q 32 25.8 36.0 29.6 Q 40 33.4 44.0 32.8 Q 48 32.1 52.0 32.6 Q 56 33.1 60.0 27.1 Q 64 21 68.0 23.8 Q 72 26.5 76.0 26.4 Q 80 26.2 84.0 32.3 Q 88 38.3 92.0 33.6 Q 96 29 100.0 30.5 Q 104 32 108.0 28.3 Q 112 24.5 116.0 31.1 Q 120 37.8 124.0 35.0 Q 128 32.1 132.0 33.5 Q 136 34.9 140.0 27.6 Q 144 20.3 148.0 24.1 Q 152 27.8 156.0 27.0 Q 160 26.2 164.0 31.4 Q 168 36.5 172.0 31.6 Q 176 26.8 180.0 28.0 Q 184 29.1 188.0 27.6 Q 192 26.1 196.0 31.8 Q 200 37.5 204.0 36.3 Q 208 35.1 212.0 34.0 Q 216 32.9 220.0 28.5 Q 224 24.1 228.0 25.5 Q 232 26.8 236.0 28.7 Q 240 30.6 244.0 31.4 Q 248 32.1 252.0 30.0 Q 256 27.8 260.0 25.6 Q 264 23.3 268.0 26.4 Q 272 29.5 276.0 31.6 Q 280 33.7 284.0 36.2 Q 288 38.7 292.0 33.8 Q 296 28.9 300.0 29.1 Q 304 29.3 308.0 27.6 Q 312 25.8 316.0 31.0 Q 320 36.2 324.0 32.4 Q 328 28.5 332.0 28.9 Q 336 29.4 340.0 24.3 Q 344 19.1 348.0 25.4 Q 352 31.6 356.0 31.1 Q 360 30.5 364.0 34.8 Q 368 39.1 372.0 32.6 Q 376 26.1 380.0 28.8 Q 384 31.5 388.0 29.4 Q 392 27.4 396.0 33.2 Q 400 39 404.0 34.0 Q 408 28.9 412.0 28.8 Q 416 28.7 420.0 24.1 Q 424 19.6 428.0 25.1 Q 432 30.6 436.0 30.7 Q 440 30.8 444.0 32.8 Q 448 34.8 452.0 30.4 Q 456 26.1 460.0 27.6 Q 464 29.1 468.0 30.6 Q 472 32 476.0 34.9 Q 480 37.7 484.0 35.3 Q 488 32.8 492.0 29.3 Q 496 25.9 500.0 25.2 Q 504 24.5 508.0 26.2 Q 512 27.9 516.0 30.8 Q 520 33.8 524.0 31.1 Q 528 28.4 532.0 27.9 Q 536 27.5 540.0 26.1 Q 544 24.6 548.0 30.5 Q 552 36.4 556.0 35.5 Q 560 34.5 564.0 35.5 Q 568 36.5 572.0 30.0 Q 576 23.5 580.0 26.7 Q 584 29.9 588.0 28.5 Q 592 27.1 596.0 31.8 Q 600 36.5 604.0 30.3 Q 608 24.1 612.0 25.8 Q 616 27.5 620.0 24.8 Q 624 22.1 628.0 29.8 Q 632 37.4 636.0 35.1 Q 640 32.8 644.0 34.5 Q 648 36.1 652.0 30.1 Q 656 24 660.0 28.1 Q 664 32.1 668.0 31.2 Q 672 30.3 676.0 33.3 Q 680 36.3 684.0 30.5 Q 688 24.7 692.0 24.8 Q 696 24.8 700.0 24.3 Q 704 23.8 708.0 29.0 Q 712 34.1 716.0 33.8 Q 720 33.5 724.0 32.3 Q 728 31.1 732.0 29.1 Q 736 27.1 740.0 28.9 Q 744 30.7 748.0 33.5 Q 752 36.3 756.0 35.0 Q 760 33.6 764.0 31.1 Q 768 28.7 772.0 25.0 Q 776 21.4 780.0 24.9 Q 784 28.4 788.0 29.0 Q 792 29.7 796.0 32.4 Q 800 35 804.0 29.9 Q 808 24.8 812.0 27.3 Q 816 29.7 820.0 29.0 Q 824 28.3 828.0 34.5 Q 832 40.7 836.0 36.0 Q 840 31.3 844.0 31.6 Q 848 31.9 852.0 26.3 Q 856 20.7 860.0 26.5 Q 864 32.3 868.0 30.3 Q 872 28.4 876.0 31.3 Q 880 34.3 884.0 27.8 Q 888 21.3 892.0 25.1 Q 896 28.9 900.0 28.5 Q 904 28.2 908.0 34.3 Q 912 40.4 916.0 36.0 Q 920 31.6 924.0 31.5 Q 928 31.3 932.0 27.8 Q 936 24.3 940.0 28.7 Q 944 33.1 948.0 32.4 Q 952 31.6 956.0 31.2 Q 960 30.8 964.0 26.6 Q 968 22.4 972.0 23.6 Q 976 24.9 980.0 27.9 Q 984 30.9 988.0 33.3 Q 992 35.7 996.0 34.7 L 1000 30",

  // IMF 2: Fetal QRS Band (20–70 Hz Target) - 8 Complete Fetal Beats with full R-peak (y:11) & S-dip (y:44)
  "M 0 30 L 25 30 Q 34 26 42 30 L 50 30 L 55 34 L 60 11 L 65 44 L 70 30 Q 80 25 90 30 L 145 30 Q 154 26 162 30 L 170 30 L 175 34 L 180 11 L 185 44 L 190 30 Q 200 25 210 30 L 265 30 Q 274 26 282 30 L 290 30 L 295 34 L 300 11 L 305 44 L 310 30 Q 320 25 330 30 L 385 30 Q 394 26 402 30 L 410 30 L 415 34 L 420 11 L 425 44 L 430 30 Q 440 25 450 30 L 505 30 Q 514 26 522 30 L 530 30 L 535 34 L 540 11 L 545 44 L 550 30 Q 560 25 570 30 L 625 30 Q 634 26 642 30 L 650 30 L 655 34 L 660 11 L 665 44 L 670 30 Q 680 25 690 30 L 745 30 Q 754 26 762 30 L 770 30 L 775 34 L 780 11 L 785 44 L 790 30 Q 800 25 810 30 L 865 30 Q 874 26 882 30 L 890 30 L 895 34 L 900 11 L 905 44 L 910 30 Q 920 25 930 30 L 1000 30",

  // IMF 3: Maternal ECG Complex (Dominant) - Complete Maternal complexes with tall R (y:10) & S (y:48)
  "M 0 30 L 60 30 Q 75 23 90 30 L 102 30 L 110 35 L 120 10 L 132 48 L 144 30 Q 164 21 184 30 L 310 30 Q 325 23 340 30 L 352 30 L 360 35 L 370 10 L 382 48 L 394 30 Q 414 21 434 30 L 560 30 Q 575 23 590 30 L 602 30 L 610 35 L 620 10 L 632 48 L 644 30 Q 664 21 684 30 L 810 30 Q 825 23 840 30 L 852 30 L 860 35 L 870 10 L 882 48 L 894 30 Q 914 21 934 30 L 1000 30",

  // IMF 4: Low frequency respiratory baseline wander (<5 Hz) - Smooth gentle respiration sine (y: 14..44)
  "M 0 30 Q 0 30 10.0 31.7 Q 20 33.4 30.0 35.0 Q 40 36.5 50.0 38.0 Q 60 39.4 70.0 40.5 Q 80 41.7 90.0 42.5 Q 100 43.3 110.0 43.8 Q 120 44.2 130.0 44.3 Q 140 44.4 150.0 44.1 Q 160 43.8 170.0 43.1 Q 180 42.5 190.0 41.5 Q 200 40.5 210.0 39.3 Q 220 38.1 230.0 36.8 Q 240 35.4 250.0 34.0 Q 260 32.6 270.0 31.1 Q 280 29.7 290.0 28.4 Q 300 27.1 310.0 26.0 Q 320 24.9 330.0 24.0 Q 340 23.2 350.0 22.7 Q 360 22.2 370.0 22.0 Q 380 21.8 390.0 22.0 Q 400 22.1 410.0 22.6 Q 420 23.1 430.0 24.0 Q 440 24.8 450.0 25.9 Q 460 26.9 470.0 28.1 Q 480 29.3 490.0 30.6 Q 500 32 510.0 33.3 Q 520 34.6 530.0 35.9 Q 540 37.1 550.0 38.2 Q 560 39.2 570.0 40.0 Q 580 40.8 590.0 41.3 Q 600 41.8 610.0 42.0 Q 620 42.1 630.0 41.9 Q 640 41.6 650.0 41.0 Q 660 40.4 670.0 39.5 Q 680 38.6 690.0 37.4 Q 700 36.1 710.0 34.7 Q 720 33.2 730.0 31.6 Q 740 30 750.0 28.4 Q 760 26.7 770.0 25.0 Q 780 23.4 790.0 21.9 Q 800 20.4 810.0 19.1 Q 820 17.9 830.0 16.9 Q 840 16 850.0 15.3 Q 860 14.7 870.0 14.4 Q 880 14.2 890.0 14.3 Q 900 14.5 910.0 15.0 Q 920 15.5 930.0 16.4 Q 940 17.3 950.0 18.5 Q 960 19.6 970.0 21.0 Q 980 22.3 990.0 23.9 L 1000 30"
];

function VisualMEMD() {
  const imfs = [
    { name: 'IMF 1', label: 'High-Freq Noise Residual (>70 Hz)', status: 'Filtered', color: '#64748b', badge: 'Noise Cut', path: IMF_FLOW_PATHS[0] },
    { name: 'IMF 2', label: 'Fetal QRS Band (20–70 Hz Target)', status: 'Isolated', color: '#059669', badge: 'Fetal Pass', selected: true, path: IMF_FLOW_PATHS[1] },
    { name: 'IMF 3', label: 'Maternal ECG Complex (Dominant)', status: 'Extracted', color: '#1882b4', badge: 'Maternal Temp', path: IMF_FLOW_PATHS[2] },
    { name: 'IMF 4', label: 'Respiratory Baseline Wander (<5 Hz)', status: 'Eliminated', color: '#94a3b8', badge: 'Baseline Cut', path: IMF_FLOW_PATHS[3] },
  ];

  return (
    <div className="stage-vis-content animate-in">
      <div className="stage-vis-header-row">
        <div>
          <h5 className="stage-vis-title">Stage 2: MEMD Signal Decomposition into Intrinsic Mode Functions</h5>
          <p className="stage-vis-sub">Multivariate EMD projects across 64 uniform directions to isolate IMF modes without mode-mixing</p>
        </div>
        <div className="stage-vis-tags">
          <span className="vis-tag vis-tag--purple">64 Directions</span>
          <span className="vis-tag vis-tag--green">IMF 2 Isolated for Fetal QRS</span>
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

            {/* Continuous flowing wave conveyor window */}
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
    </div>
  );
}

/* ─── STAGE 3: PINN (Physics-Informed Maternal Cancellation) ─── */
function VisualPINN({ isAnalyzing }) {
  return (
    <div className="stage-vis-content animate-in">
      <div className="stage-vis-header-row">
        <div>
          <h5 className="stage-vis-title">Stage 3: PINN Potential Cancellation &amp; Fetal Separation</h5>
          <p className="stage-vis-sub">Volume conduction Laplace potential field cancels high-amplitude maternal complexes from abdominal mixtures</p>
        </div>
        <div className="stage-vis-tags">
          <span className="vis-tag vis-tag--cyan">Maternal Cancellation: 99.1%</span>
          <span className="vis-tag vis-tag--green">Fetal Recovery F1: 96.6%</span>
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
          <span className="vis-pinn-val">Maternal Template</span>
        </div>
        <span className="vis-pinn-op">=</span>
        <div className="vis-pinn-step vis-pinn-step--result">
          <span className="vis-pinn-badge vis-pinn-badge--green">Pure Fetal Output</span>
          <span className="vis-pinn-val text-success">Purified Fetal ECG</span>
        </div>
      </div>

      {/* Subtraction Waveform Demonstration */}
      <div className="vis-wave-channel vis-wave-channel--pinn">
        <div className="vis-wave-label">
          <span className="vis-dot vis-dot--green" />
          <span>Extracted Fetal Signal (Maternal ECG Cleanly Subtracted)</span>
          <span className="vis-channel-meta">Sharp Fetal R-Peaks at ~145 bpm</span>
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
    </div>
  );
}

/* ─── STAGE 4: 12 Feature (Morphological Biomarkers Extraction) ─── */
function VisualFeatures() {
  const features = [
    { key: 'RR_interval',   label: 'RR Interval',       val: '422 ms',    normal: '350–500 ms', color: 'blue' },
    { key: 'FHR',           label: 'Fetal Heart Rate',  val: '142 bpm',   normal: '120–160 bpm', color: 'green' },
    { key: 'QRS_duration',  label: 'QRS Width',         val: '48 ms',     normal: '40–60 ms',   color: 'cyan' },
    { key: 'PR_interval',   label: 'PR Interval',       val: '104 ms',    normal: '90–120 ms',  color: 'purple' },
    { key: 'QT_interval',   label: 'QT Interval',       val: '210 ms',    normal: '180–240 ms', color: 'green' },
    { key: 'QTc_interval',  label: 'QTc (Bazett)',      val: '324 ms',    normal: '300–360 ms', color: 'blue' },
    { key: 'P_amplitude',   label: 'P Amplitude',       val: '0.09 mV',   normal: '0.05–0.15 mV', color: 'cyan' },
    { key: 'P_duration',    label: 'P Duration',        val: '42 ms',     normal: '35–55 ms',   color: 'purple' },
    { key: 'R_amplitude',   label: 'R-Peak Amplitude',  val: '0.86 mV',   normal: '0.6–1.2 mV', color: 'green' },
    { key: 'S_depth',       label: 'S-Wave Depth',      val: '-0.28 mV',  normal: '-0.15 to -0.4', color: 'blue' },
    { key: 'ST_elevation',  label: 'ST Elevation',      val: '0.04 mV',   normal: '< 0.08 mV',  color: 'cyan' },
    { key: 'T_amplitude',   label: 'T Amplitude',       val: '0.19 mV',   normal: '0.10–0.25 mV', color: 'green' },
  ];

  return (
    <div className="stage-vis-content animate-in">
      <div className="stage-vis-header-row">
        <div>
          <h5 className="stage-vis-title">Stage 4: 12 Morphological &amp; Dynamic Biomarkers</h5>
          <p className="stage-vis-sub">Fiducial P-Q-R-S-T wave point detection and interval extraction for gestational modeling</p>
        </div>
        <div className="stage-vis-tags">
          <span className="vis-tag vis-tag--green">12 Biomarkers Extracted</span>
          <span className="vis-tag vis-tag--blue">P-Q-R-S-T Calibrated</span>
        </div>
      </div>

      {/* SVG Cardiac Beat with Caliper Markers */}
      <div className="vis-caliper-box">
        <svg className="vis-svg-caliper" viewBox="0 0 500 120">
          {/* Baseline grid */}
          <line x1="0" y1="80" x2="500" y2="80" stroke="#e2e8f0" strokeDasharray="3 3" />

          {/* ECG Beat Path: P -> Q -> R -> S -> T */}
          <path
            d="M 40,80 Q 90,80 120,80 Q 140,65 160,80 Q 185,80 195,95 Q 215,10 230,110 Q 240,80 260,80 Q 300,55 340,80 Q 400,80 460,80"
            fill="none"
            stroke="#1882b4"
            strokeWidth="3"
            strokeLinecap="round"
          />

          {/* Points: P, Q, R, S, T */}
          <circle cx="140" cy="72" r="4" fill="#3ab2e6" />
          <text x="136" y="60" fontSize="12" fontWeight="700" fill="#3ab2e6">P</text>

          <circle cx="195" cy="95" r="4" fill="#64748b" />
          <text x="187" y="112" fontSize="12" fontWeight="700" fill="#64748b">Q</text>

          <circle cx="215" cy="10" r="5" fill="#dc2626" />
          <text x="210" y="3" fontSize="13" fontWeight="900" fill="#dc2626">R</text>

          <circle cx="230" cy="110" r="4" fill="#64748b" />
          <text x="234" y="115" fontSize="12" fontWeight="700" fill="#64748b">S</text>

          <circle cx="320" cy="67" r="4" fill="#059669" />
          <text x="316" y="55" fontSize="12" fontWeight="700" fill="#059669">T</text>

          {/* Measurement Caliper lines */}
          <line x1="195" y1="20" x2="230" y2="20" stroke="#1882b4" strokeWidth="1.5" strokeDasharray="2 2" />
          <text x="200" y="32" fontSize="10" fontWeight="700" fill="#1882b4">QRS: 48ms</text>

          <line x1="195" y1="118" x2="340" y2="118" stroke="#059669" strokeWidth="1.5" strokeDasharray="2 2" />
          <text x="250" y="115" fontSize="10" fontWeight="700" fill="#059669">QT: 210ms</text>
        </svg>

        {/* 6 key values grid */}
        <div className="vis-feat-grid">
          {features.map((f) => (
            <div key={f.key} className="vis-feat-cell">
              <span className="vis-feat-label">{f.label}</span>
              <span className="vis-feat-val" style={{ color: `var(--${f.color})` }}>{f.val}</span>
              <span className="vis-feat-normal">Ref: {f.normal}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

/* ─── STAGE 5: Transformer (Multi-Head Self-Attention) ─── */
function VisualTransformer() {
  const heads = [
    { id: 1, name: 'Head 1: QRS Sharpness', weight: '0.34', color: '#1882b4' },
    { id: 2, name: 'Head 2: R-R Variability', weight: '0.28', color: '#059669' },
    { id: 3, name: 'Head 3: ST Morphometry', weight: '0.22', color: '#7c3aed' },
    { id: 4, name: 'Head 4: Long-term Rhythm', weight: '0.16', color: '#0284c7' },
  ];

  return (
    <div className="stage-vis-content animate-in">
      <div className="stage-vis-header-row">
        <div>
          <h5 className="stage-vis-title">Stage 5: Multi-Head Self-Attention Transformer Encoder</h5>
          <p className="stage-vis-sub">Temporal token attention correlates beat-to-beat variability and feature vectors across cardiac cycles</p>
        </div>
        <div className="stage-vis-tags">
          <span className="vis-tag vis-tag--blue">4 Attention Heads</span>
          <span className="vis-tag vis-tag--purple">3 Transformer Layers</span>
        </div>
      </div>

      <div className="vis-attention-grid">
        {heads.map((h) => (
          <div key={h.id} className="vis-attention-card">
            <div className="vis-head-top">
              <span className="vis-head-title" style={{ color: h.color }}>{h.name}</span>
              <span className="vis-head-weight">Weight: {h.weight}</span>
            </div>
            <div className="vis-head-matrix">
              {Array.from({ length: 16 }).map((_, i) => (
                <div
                  key={i}
                  className="vis-matrix-cell"
                  style={{
                    backgroundColor: h.color,
                    opacity: 0.15 + ((i * 7) % 85) / 100,
                  }}
                />
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

/* ─── STAGE 6: Cardiac Age (Gestational Age Prediction) ─── */
function VisualCardiacAge() {
  return (
    <div className="stage-vis-content animate-in">
      <div className="stage-vis-header-row">
        <div>
          <h5 className="stage-vis-title">Stage 6: Fetal Cardiac Age Estimation &amp; Gestational Concordance</h5>
          <p className="stage-vis-sub">Transformer regressor predicts chronological cardiac maturity calibrated against LMP reference</p>
        </div>
        <div className="stage-vis-tags">
          <span className="vis-tag vis-tag--blue">Prediction: 212 days (30.3 wks)</span>
          <span className="vis-tag vis-tag--green">Concordance: Normal (Δ+2d)</span>
        </div>
      </div>

      <div className="vis-cardiac-gauge-container">
        {/* Growth Curve Chart */}
        <div className="vis-growth-chart">
          <svg className="vis-svg-growth" viewBox="0 0 500 130">
            {/* 95% Confidence Interval Band */}
            <path
              d="M 50,110 Q 250,65 450,20 L 450,45 Q 250,90 50,125 Z"
              fill="rgba(24, 130, 180, 0.12)"
            />

            {/* Mean developmental curve */}
            <path
              d="M 50,118 Q 250,78 450,32"
              fill="none"
              stroke="#1882b4"
              strokeWidth="2.5"
            />

            {/* Target LMP reference point */}
            <circle cx="270" cy="74" r="6" fill="#7c3aed" />
            <text x="250" y="60" fontSize="11" fontWeight="700" fill="#7c3aed">LMP: 210d</text>

            {/* Predicted Cardiac Age point */}
            <circle cx="275" cy="71" r="7" fill="#059669" />
            <text x="288" y="75" fontSize="12" fontWeight="800" fill="#059669">CA: 212d</text>
          </svg>

          <div className="vis-growth-legend">
            <span className="vis-legend-item"><span className="vis-dot vis-dot--purple" /> LMP Reference (210d)</span>
            <span className="vis-legend-item"><span className="vis-dot vis-dot--green" /> Estimated Cardiac Age (212d)</span>
            <span className="vis-legend-item"><span className="vis-dot vis-dot--blue" /> 95% CI Band [199 – 225d]</span>
          </div>
        </div>

        {/* Blue Heart Animated Icon Card */}
        <div className="vis-cardiac-icon-card">
          <div className="vis-cardiac-logo-wrapper">
            <BlueLogo size={36} />
          </div>
          <span className="vis-cardiac-result-val">212 <span style={{ fontSize: 13, fontWeight: 600 }}>days</span></span>
          <span className="vis-cardiac-result-sub">30.3 Gestational Weeks</span>
          <span className="vis-cardiac-badge text-success">✓ Normal Maturation</span>
        </div>
      </div>
    </div>
  );
}

/* ─── MAIN STAGE VISUALIZER COMPONENT ────────────────────── */
export default function StageVisualizer({
  activeStep = 1,
  isAnalyzing = false,
  completionPercentage = 0,
  telemetry = null,
}) {
  // Ensure step is within 1 to 6
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

    // Stagger child elements (tags, badges, cards) for fluid settling
    const childTags = containerRef.current.querySelectorAll(
      '.stage-vis-tags .vis-tag, .vis-feat-cell, .vis-attention-card'
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
      {/* ── Active Stage Visual Simulation (Smoothly transitioned with Anime.js) ── */}
      <div className="stage-vis-body">
        <div ref={containerRef} className="stage-vis-transition-wrapper">
          {displayedStep === 1 && <VisualPreprocess isAnalyzing={isAnalyzing && currentTarget === 1} />}
          {displayedStep === 2 && <VisualMEMD isAnalyzing={isAnalyzing && currentTarget === 2} />}
          {displayedStep === 3 && <VisualPINN isAnalyzing={isAnalyzing && currentTarget === 3} />}
          {displayedStep === 4 && <VisualFeatures isAnalyzing={isAnalyzing && currentTarget === 4} />}
          {displayedStep === 5 && <VisualTransformer isAnalyzing={isAnalyzing && currentTarget === 5} />}
          {displayedStep === 6 && <VisualCardiacAge isAnalyzing={isAnalyzing && currentTarget === 6} />}
        </div>
      </div>
    </div>
  );
}
