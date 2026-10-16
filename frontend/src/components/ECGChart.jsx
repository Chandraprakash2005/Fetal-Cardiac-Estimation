import React, { useEffect, useRef } from 'react';
import {
  Chart as ChartJS,
  CategoryScale, LinearScale, PointElement, LineElement,
  Title, Tooltip, Legend, Filler,
} from 'chart.js';
import { Line } from 'react-chartjs-2';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend, Filler);

function downsample(arr, maxPoints = 300) {
  if (!arr || arr.length <= maxPoints) return arr;
  const factor = Math.ceil(arr.length / maxPoints);
  return arr.filter((_, i) => i % factor === 0);
}

export default function ECGChart({ maternalSignal, fetalSignal, imfSignals }) {
  const samplingRate = 1000;

  const makeLabels = (data) =>
    data.map((_, i) => (i / samplingRate).toFixed(3));

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>

      {/* ── Maternal ECG ─────────────────────────────────── */}
      {maternalSignal && (
        <div className="ecg-chart-block">
          <div className="ecg-chart-header">
            <span className="ecg-chip ecg-chip--cyan">Channel: Abdominal ECG</span>
            <span className="ecg-chip-label">After PINN — Maternal Component</span>
          </div>
          <div className="chart-container" style={{ height: 160 }}>
            <Line
              data={{
                labels: makeLabels(downsample(maternalSignal)),
                datasets: [{
                  label: 'Maternal ECG (mV)',
                  data: downsample(maternalSignal),
                  borderColor: '#40c8ff',
                  borderWidth: 1.5,
                  pointRadius: 0,
                  tension: 0.3,
                  fill: false,
                }],
              }}
              options={chartOptions('Maternal ECG (mV)', '#40c8ff')}
            />
          </div>
        </div>
      )}

      {/* ── Fetal ECG ────────────────────────────────────── */}
      {fetalSignal && (
        <div className="ecg-chart-block">
          <div className="ecg-chart-header">
            <span className="ecg-chip ecg-chip--green">Channel: Fetal ECG</span>
            <span className="ecg-chip-label">Extracted via MEMD + PINN</span>
          </div>
          <div className="chart-container" style={{ height: 160 }}>
            <Line
              data={{
                labels: makeLabels(downsample(fetalSignal)),
                datasets: [{
                  label: 'Fetal ECG (mV)',
                  data: downsample(fetalSignal),
                  borderColor: '#00e5a0',
                  borderWidth: 1.5,
                  pointRadius: 0,
                  tension: 0.3,
                  fill: true,
                  backgroundColor: 'rgba(0,229,160,0.05)',
                }],
              }}
              options={chartOptions('Fetal ECG (mV)', '#00e5a0')}
            />
          </div>
        </div>
      )}

      {/* ── IMF Components ────────────────────────────────── */}
      {imfSignals && imfSignals.length > 0 && (
        <div className="ecg-chart-block">
          <div className="ecg-chart-header">
            <span className="ecg-chip ecg-chip--purple">MEMD Decomposition</span>
            <span className="ecg-chip-label">Top 3 IMF Components (20–70 Hz band)</span>
          </div>
          <div className="chart-container" style={{ height: 160 }}>
            <Line
              data={{
                labels: makeLabels(downsample(imfSignals[0])),
                datasets: imfSignals.map((sig, i) => ({
                  label: `IMF ${i + 1}`,
                  data: downsample(sig),
                  borderColor: ['#a855f7', '#40c8ff', '#00e5a0'][i],
                  borderWidth: 1.2,
                  pointRadius: 0,
                  tension: 0.3,
                  fill: false,
                })),
              }}
              options={chartOptions('Amplitude', '#a855f7')}
            />
          </div>
        </div>
      )}
    </div>
  );
}

function chartOptions(yLabel, color) {
  return {
    responsive: true,
    maintainAspectRatio: false,
    animation: { duration: 600 },
    plugins: {
      legend: {
        labels: {
          color: '#7aa3cc',
          font: { family: 'Inter', size: 11 },
          boxWidth: 12,
        },
      },
      tooltip: {
        backgroundColor: 'rgba(8,15,32,0.95)',
        borderColor: 'rgba(64,200,255,0.2)',
        borderWidth: 1,
        titleColor: '#e8f4ff',
        bodyColor: '#7aa3cc',
        titleFont: { family: 'JetBrains Mono', size: 11 },
        bodyFont: { family: 'JetBrains Mono', size: 11 },
      },
    },
    scales: {
      x: {
        ticks: {
          color: '#3d6480',
          font: { family: 'JetBrains Mono', size: 9 },
          maxTicksLimit: 10,
        },
        grid: { color: 'rgba(64,200,255,0.05)' },
        title: {
          display: true,
          text: 'Time (s)',
          color: '#3d6480',
          font: { family: 'Inter', size: 10 },
        },
      },
      y: {
        ticks: {
          color: '#3d6480',
          font: { family: 'JetBrains Mono', size: 9 },
        },
        grid: { color: 'rgba(64,200,255,0.05)' },
        title: {
          display: true,
          text: yLabel,
          color: '#3d6480',
          font: { family: 'Inter', size: 10 },
        },
      },
    },
  };
}
