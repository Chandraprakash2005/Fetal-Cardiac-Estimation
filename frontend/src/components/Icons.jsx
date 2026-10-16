import React from 'react';

/**
 * Blue Heart + ECG Pulse Logo
 * Replaces the colorful anatomical heart emoji with a sleek, clinical blue vector emblem.
 */
export function BlueLogo({ size = 26, className = '' }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 32 32"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={`blue-logo-svg ${className}`}
      aria-label="FetalECG Blue Heart Logo"
    >
      <defs>
        <linearGradient id="blueLogoBg" x1="4" y1="4" x2="28" y2="28" gradientUnits="userSpaceOnUse">
          <stop stopColor="#1882b4" />
          <stop offset="1" stopColor="#3ab2e6" />
        </linearGradient>
      </defs>

      {/* Blue Heart Shape with subtle pulse */}
      <path
        className="blue-logo-heart"
        d="M16 27.2l-1.85-1.68C7.58 19.56 3.25 15.65 3.25 10.85 3.25 6.93 6.33 3.85 10.25 3.85c2.21 0 4.34 1.03 5.75 2.66 1.41-1.63 3.54-2.66 5.75-2.66 3.92 0 7 3.08 7 7 0 4.8-4.33 8.71-10.9 14.67L16 27.2z"
        fill="url(#blueLogoBg)"
      />

      {/* Crisp White Animated ECG Cardiogram Pulse Wave */}
      <path
        className="blue-logo-ecg"
        d="M5.5 15.5h4.2l1.8-4.5 3 9 2.5-6.5 1.5 3h6"
        stroke="#ffffff"
        strokeWidth="2.2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

/* ─── Blue Monochrome Pipeline Stage Icons with Micro-Animations ─── */

// 1. Preprocess: Bandpass filter / clean signal
export function IconPreprocess({ size = 22, color = '#1882b4', className = '' }) {
  return (
    <svg className={`anim-icon anim-icon--preprocess ${className}`} width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
      <path className="anim-wave-path" d="M2 12h3l2-5 3 10 3-7 2 4h7" />
      <circle className="anim-filter-dot" cx="20" cy="6" r="2" fill={color} stroke="none" />
    </svg>
  );
}

// 2. MEMD: Multivariate Empirical Mode Decomposition
export function IconMEMD({ size = 22, color = '#1882b4', className = '' }) {
  return (
    <svg className={`anim-icon anim-icon--memd ${className}`} width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
      <path className="anim-imf-wave1" d="M3 8c3-4 6-4 9 0s6 4 9 0" />
      <path className="anim-imf-wave2" d="M3 16c3-4 6-4 9 0s6 4 9 0" strokeOpacity="0.8" />
    </svg>
  );
}

// 3. PINN: Physics-Informed Neural Network (potential / cancellation)
export function IconPINN({ size = 22, color = '#1882b4', className = '' }) {
  return (
    <svg className={`anim-icon anim-icon--pinn ${className}`} width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
      <polygon className="anim-pinn-bolt" points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" fill="rgba(24, 130, 180, 0.18)" />
    </svg>
  );
}

// 4. 12 Feature: Morphological ECG features / dynamic equalizer
export function IconFeature({ size = 22, color = '#1882b4', className = '' }) {
  return (
    <svg className={`anim-icon anim-icon--feature ${className}`} width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
      <line className="anim-bar anim-bar-3" x1="18" y1="20" x2="18" y2="10" />
      <line className="anim-bar anim-bar-2" x1="12" y1="20" x2="12" y2="4" />
      <line className="anim-bar anim-bar-1" x1="6" y1="20" x2="6" y2="14" />
      <path d="M4 20h16" />
    </svg>
  );
}

// 5. CVAE: Contrastive Variational Autoencoder / latent embedding
export function IconCVAE({ size = 22, color = '#1882b4', className = '' }) {
  return (
    <svg className={`anim-icon anim-icon--cvae ${className}`} width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
      <g className="anim-cvae-group">
        <circle className="anim-cvae-node1" cx="6" cy="12" r="3" fill="rgba(24, 130, 180, 0.22)" />
        <circle className="anim-cvae-node2" cx="18" cy="12" r="3" fill="rgba(24, 130, 180, 0.22)" />
        <path d="M9 12h6" />
        <path d="M12 9v6" />
      </g>
    </svg>
  );
}

// 6. Transformer: Self-Attention / multi-head architecture
export function IconTransformer({ size = 22, color = '#1882b4', className = '' }) {
  return (
    <svg className={`anim-icon anim-icon--transformer ${className}`} width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
      <rect className="anim-trans-head1" x="4" y="4" width="6" height="6" rx="1.5" />
      <rect className="anim-trans-head2" x="14" y="4" width="6" height="6" rx="1.5" />
      <rect className="anim-trans-head3" x="9" y="14" width="6" height="6" rx="1.5" />
      <path className="anim-trans-link" d="M7 10v2a2 2 0 0 0 2 2h3m5-4v2a2 2 0 0 1-2 2h-3" />
    </svg>
  );
}

// 7. Cardiac Age: Estimation result / heart clock
export function IconCardiacAge({ size = 22, color = '#1882b4', className = '' }) {
  return (
    <svg className={`anim-icon anim-icon--cardiac ${className}`} width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
      <path className="anim-cardiac-heart" d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z" fill="rgba(24, 130, 180, 0.18)" />
      <polyline className="anim-clock-hand" points="12 10 12 14 15 15" />
    </svg>
  );
}

/* ─── UI Badges & Buttons Icons ─────────────── */

export function IconSignal({ size = 18, color = '#1882b4', className = '' }) {
  return (
    <svg className={`anim-icon anim-icon--signal ${className}`} width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M4 14.899A7 7 0 1 1 15.71 8h1.79a4.5 4.5 0 0 1 2.5 8.242" />
      <path className="anim-signal-arrow" d="M12 12v9" />
      <path className="anim-signal-arrow" d="m8 17 4 4 4-4" />
    </svg>
  );
}

export function IconUpload({ size = 22, color = '#1882b4', className = '' }) {
  return (
    <svg className={`anim-icon anim-icon--upload ${className}`} width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
      <polyline className="anim-upload-arrow" points="17 8 12 3 7 8" />
      <line className="anim-upload-arrow" x1="12" y1="3" x2="12" y2="15" />
    </svg>
  );
}

export function IconHeartBadge({ size = 16, color = '#1882b4', className = '' }) {
  return (
    <svg className={`anim-icon anim-icon--badge-heart ${className}`} width={size} height={size} viewBox="0 0 24 24" fill={color} stroke={color} strokeWidth="1" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 21.35l-1.45-1.32C5.4 15.36 2 12.28 2 8.5 2 5.42 4.42 3 7.5 3c1.74 0 3.41.81 4.5 2.09C13.09 3.81 14.76 3 16.5 3 19.58 3 22 5.42 22 8.5c0 3.78-3.4 6.86-8.55 11.54L12 21.35z" />
    </svg>
  );
}

export function IconPulseBadge({ size = 16, color = '#1882b4', className = '' }) {
  return (
    <svg className={`anim-icon anim-icon--badge-pulse ${className}`} width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
    </svg>
  );
}

export function IconChartBadge({ size = 16, color = '#1882b4', className = '' }) {
  return (
    <svg className={`anim-icon anim-icon--badge-chart ${className}`} width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M3 3v18h18" />
      <path className="anim-chart-line" d="m19 9-5 5-4-4-3 3" />
    </svg>
  );
}

export function IconSettings({ size = 16, color = 'currentColor', className = '' }) {
  return (
    <svg className={`anim-icon anim-icon--settings ${className}`} width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="3" />
      <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z" />
    </svg>
  );
}
