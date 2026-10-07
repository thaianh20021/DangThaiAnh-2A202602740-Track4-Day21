/**
 * LiDAR-Camera Projection QA Studio - Frontend Application Logic
 */

// Application State
const state = {
  dataset: 'kitti',
  frame_id: '000011',
  yaw: 0.0,
  pitch: 0.0,
  roll: 0.0,
  tx: 0.0,
  ty: 0.0,
  tz: 0.0,
  show_boxes: true,
  show_points: true,
  max_depth: 50.0,
  point_radius: 2,
  isComparingBaseline: false,
  baselineImage: null,
  currentImage: null,
};

let debounceTimer = null;
let chartsInitialized = false;

// DOM Elements
const el = {
  datasetSelect: document.getElementById('dataset-select'),
  frameSelect: document.getElementById('frame-select'),
  sliderYaw: document.getElementById('slider-yaw'),
  sliderPitch: document.getElementById('slider-pitch'),
  sliderRoll: document.getElementById('slider-roll'),
  sliderTy: document.getElementById('slider-ty'),
  sliderTz: document.getElementById('slider-tz'),
  valYaw: document.getElementById('val-yaw'),
  valPitch: document.getElementById('val-pitch'),
  valRoll: document.getElementById('val-roll'),
  valTy: document.getElementById('val-ty'),
  valTz: document.getElementById('val-tz'),
  btnReset: document.getElementById('btn-reset-sliders'),
  chkBoxes: document.getElementById('chk-show-boxes'),
  chkPoints: document.getElementById('chk-show-points'),
  sliderMaxDepth: document.getElementById('slider-max-depth'),
  valMaxDepth: document.getElementById('val-max-depth'),
  liveImg: document.getElementById('live-canvas-img'),
  spinner: document.getElementById('loading-spinner'),
  btnToggleDiff: document.getElementById('btn-toggle-diff'),
  btnFullscreen: document.getElementById('btn-fullscreen'),
  
  // Telemetry Elements
  metricMeanShift: document.getElementById('metric-mean-shift'),
  metricP95Shift: document.getElementById('metric-p95-shift'),
  metricMaxShift: document.getElementById('metric-max-shift'),
  metricRetentionAll: document.getElementById('metric-retention-all'),
  metricRetNear: document.getElementById('metric-ret-near'),
  metricRetFar: document.getElementById('metric-ret-far'),
  metricEdgeScore: document.getElementById('metric-edge-score'),
  statusPill: document.getElementById('header-status-pill'),
  statusText: document.getElementById('status-text'),
  alertBanner: document.getElementById('system-alert-banner'),
  alertHeadline: document.getElementById('alert-headline'),
  alertDetail: document.getElementById('alert-detail'),
  alertIcon: document.getElementById('alert-icon'),
  
  // Containers
  objectsContainer: document.getElementById('objects-cards-container'),
  tabs: document.querySelectorAll('.tab-btn'),
  panes: document.querySelectorAll('.tab-pane'),
  presetBtns: document.querySelectorAll('.btn-preset'),
};

// Initialize Application
document.addEventListener('DOMContentLoaded', () => {
  setupEventListeners();
  triggerUpdate(true);
});

function setupEventListeners() {
  // Selectors
  el.datasetSelect.addEventListener('change', (e) => {
    state.dataset = e.target.value;
    updateFrameOptions();
    triggerUpdate(true);
  });

  el.frameSelect.addEventListener('change', (e) => {
    state.frame_id = e.target.value;
    triggerUpdate(true);
  });

  // Sliders with direct updates
  const bindSlider = (slider, valEl, key, formatFn, isMeter = false) => {
    slider.addEventListener('input', (e) => {
      const val = parseFloat(e.target.value);
      state[key] = isMeter ? val / 100.0 : val;
      valEl.textContent = formatFn(val);
      clearActivePresets();
      scheduleUpdate();
    });
  };

  bindSlider(el.sliderYaw, el.valYaw, 'yaw', v => `${v > 0 ? '+' : ''}${v.toFixed(1)}°`);
  bindSlider(el.sliderPitch, el.valPitch, 'pitch', v => `${v > 0 ? '+' : ''}${v.toFixed(1)}°`);
  bindSlider(el.sliderRoll, el.valRoll, 'roll', v => `${v > 0 ? '+' : ''}${v.toFixed(1)}°`);
  bindSlider(el.sliderTy, el.valTy, 'ty', v => `${v > 0 ? '+' : ''}${v} cm`, true);
  bindSlider(el.sliderTz, el.valTz, 'tz', v => `${v > 0 ? '+' : ''}${v} cm`, true);

  el.sliderMaxDepth.addEventListener('input', (e) => {
    state.max_depth = parseFloat(e.target.value);
    el.valMaxDepth.textContent = `${state.max_depth}m`;
    scheduleUpdate();
  });

  // Toggles
  el.chkBoxes.addEventListener('change', (e) => {
    state.show_boxes = e.target.checked;
    triggerUpdate(false);
  });

  el.chkPoints.addEventListener('change', (e) => {
    state.show_points = e.target.checked;
    triggerUpdate(false);
  });

  // Reset Button
  el.btnReset.addEventListener('click', () => {
    resetSliders();
    triggerUpdate(true);
  });

  // Preset Buttons
  el.presetBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      el.presetBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');

      const yaw = parseFloat(btn.dataset.yaw || 0);
      const pitch = parseFloat(btn.dataset.pitch || 0);
      const roll = parseFloat(btn.dataset.roll || 0);
      const ty = parseFloat(btn.dataset.ty || 0);
      const tz = parseFloat(btn.dataset.tz || 0);

      setSliderValues(yaw, pitch, roll, ty * 100, tz * 100);
      triggerUpdate(true);
    });
  });

  // Tab switching
  el.tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      const target = tab.dataset.tab;
      el.tabs.forEach(t => t.classList.remove('active'));
      el.panes.forEach(p => p.classList.remove('active'));
      tab.classList.add('active');
      document.getElementById(target).classList.add('active');

      if (target === 'tab-analytics' && !chartsInitialized) {
        initBenchmarkCharts();
      }
    });
  });

  // Toggle Diff
  el.btnToggleDiff.addEventListener('click', () => {
    if (!state.baselineImage) return;
    state.isComparingBaseline = !state.isComparingBaseline;
    if (state.isComparingBaseline) {
      el.liveImg.src = state.baselineImage;
      el.btnToggleDiff.textContent = 'Trở lại Drift hiện tại';
      el.btnToggleDiff.style.borderColor = 'var(--accent-cyan)';
    } else {
      el.liveImg.src = state.currentImage;
      el.btnToggleDiff.textContent = 'So sánh với Baseline (0°)';
      el.btnToggleDiff.style.borderColor = '';
    }
  });

  // Fullscreen
  el.btnFullscreen.addEventListener('click', () => {
    const stage = document.getElementById('image-stage');
    if (!document.fullscreenElement) {
      stage.requestFullscreen().catch(() => {});
    } else {
      document.exitFullscreen().catch(() => {});
    }
  });
}

function updateFrameOptions() {
  el.frameSelect.innerHTML = '';
  let frames = [];
  if (state.dataset === 'kitti') {
    frames = [
      { id: '000011', name: '000011 (Đa vật thể gần & xa)' },
      { id: '000021', name: '000021 (Đường phố)' },
      { id: '000049', name: '000049 (Nhiều ô tô)' },
      { id: '000001', name: '000001 (Baseline)' },
    ];
  } else if (state.dataset === 'nusc') {
    frames = [
      { id: 'scene-0103_010', name: 'scene-0103_010 (Góc nhìn rộng)' },
      { id: 'scene-0103_000', name: 'scene-0103_000' },
      { id: 'scene-1094_000', name: 'scene-1094_000' },
    ];
  } else {
    frames = [
      { id: '000000', name: '000000 (Synthetic test)' },
      { id: '000001', name: '000001' },
    ];
  }

  frames.forEach(f => {
    const opt = document.createElement('option');
    opt.value = f.id;
    opt.textContent = f.name;
    el.frameSelect.appendChild(opt);
  });
  state.frame_id = frames[0].id;
}

function setSliderValues(yaw, pitch, roll, tyCm, tzCm) {
  state.yaw = yaw;
  state.pitch = pitch;
  state.roll = roll;
  state.ty = tyCm / 100.0;
  state.tz = tzCm / 100.0;

  el.sliderYaw.value = yaw;
  el.sliderPitch.value = pitch;
  el.sliderRoll.value = roll;
  el.sliderTy.value = tyCm;
  el.sliderTz.value = tzCm;

  el.valYaw.textContent = `${yaw > 0 ? '+' : ''}${yaw.toFixed(1)}°`;
  el.valPitch.textContent = `${pitch > 0 ? '+' : ''}${pitch.toFixed(1)}°`;
  el.valRoll.textContent = `${roll > 0 ? '+' : ''}${roll.toFixed(1)}°`;
  el.valTy.textContent = `${tyCm > 0 ? '+' : ''}${tyCm} cm`;
  el.valTz.textContent = `${tzCm > 0 ? '+' : ''}${tzCm} cm`;
}

function resetSliders() {
  setSliderValues(0, 0, 0, 0, 0);
  clearActivePresets();
  el.presetBtns[0]?.classList.add('active');
}

function clearActivePresets() {
  el.presetBtns.forEach(b => b.classList.remove('active'));
}

function scheduleUpdate() {
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(() => {
    triggerUpdate(false);
  }, 40); // 40ms debounce (~25 FPS tương tác cực nhạy)
}

async function triggerUpdate(isHardChange = false) {
  if (isHardChange) {
    el.spinner.classList.add('visible');
  }

  try {
    const res = await fetch('/api/project', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(state),
    });

    if (!res.ok) throw new Error('API projection error');
    const data = await res.json();

    // Cache baseline image if at (0, 0, 0)
    if (state.yaw === 0 && state.pitch === 0 && state.roll === 0 && state.ty === 0 && state.tz === 0) {
      state.baselineImage = data.image;
    }
    state.currentImage = data.image;

    if (!state.isComparingBaseline) {
      el.liveImg.src = data.image;
    }

    updateTelemetry(data.metrics);
    updateObjectsList(data.objects);
  } catch (err) {
    console.error('Error fetching projection:', err);
  } finally {
    if (isHardChange) {
      el.spinner.classList.remove('visible');
    }
  }
}

function updateTelemetry(metrics) {
  // Numbers
  el.metricMeanShift.textContent = metrics.mean_pixel_shift.toFixed(2);
  el.metricP95Shift.textContent = metrics.p95_pixel_shift.toFixed(1);
  el.metricMaxShift.textContent = metrics.max_pixel_shift.toFixed(1);

  el.metricRetentionAll.textContent = metrics.retention_all_pct.toFixed(1);
  el.metricRetNear.textContent = `${metrics.retention_near_pct.toFixed(1)}%`;
  el.metricRetFar.textContent = `${metrics.retention_far_pct.toFixed(1)}%`;
  el.metricEdgeScore.textContent = metrics.edge_alignment_score.toFixed(4);

  // Status & Alerts
  el.statusText.textContent = metrics.status;
  el.statusPill.style.color = metrics.status_color;
  el.statusPill.style.borderColor = metrics.status_color;
  el.statusPill.style.background = `${metrics.status_color}22`;

  el.alertBanner.className = `alert-banner alert-${metrics.status.toLowerCase()}`;
  el.alertHeadline.textContent = `Trạng thái: ${metrics.status} — `;
  el.alertDetail.textContent = metrics.status_msg;
  el.alertIcon.innerHTML = metrics.status === 'OPTIMAL' ? '&#x2714;' : (metrics.status === 'WARNING' ? '&#x26A0;' : '&#x2622;');

  // Dynamic colors on retention
  if (metrics.retention_all_pct >= 85) {
    el.metricRetentionAll.className = 'metric-val text-green';
  } else if (metrics.retention_all_pct >= 60) {
    el.metricRetentionAll.className = 'metric-val text-amber';
  } else {
    el.metricRetentionAll.className = 'metric-val text-red';
  }
}

function updateObjectsList(objects) {
  el.objectsContainer.innerHTML = '';
  if (!objects || objects.length === 0) {
    el.objectsContainer.innerHTML = '<div style="color:var(--text-muted); padding:20px;">Không có nhãn vật thể 2D trên khung hình này.</div>';
    return;
  }

  objects.forEach(obj => {
    const card = document.createElement('div');
    card.className = 'object-card';

    let colorClass = 'var(--accent-green)';
    if (obj.retention_rate < 40) colorClass = 'var(--accent-red)';
    else if (obj.retention_rate < 80) colorClass = 'var(--accent-amber)';

    card.innerHTML = `
      <div class="obj-header">
        <span class="obj-name">${obj.type} #${obj.id}</span>
        <span class="obj-dist">${obj.distance} m</span>
      </div>
      <div class="obj-bar-wrapper">
        <div class="obj-bar-bg">
          <div class="obj-bar-fill" style="width: ${obj.retention_rate}%; background: ${colorClass};"></div>
        </div>
      </div>
      <div class="obj-stats-row">
        <span>Retention: <strong style="color:${colorClass}">${obj.retention_rate}%</strong></span>
        <span>${obj.retained_points} / ${obj.base_points} pts</span>
      </div>
    `;
    el.objectsContainer.appendChild(card);
  });
}

// Benchmark Charts Initialization
async function initBenchmarkCharts() {
  chartsInitialized = true;
  try {
    const res = await fetch('/api/benchmark-data');
    const data = await res.json();
    const sweep = data.sweep || [];
    const cross = data.cross_dataset || [];

    // Filter yaw sweep
    const yawData = sweep
      .filter(r => r.config.startsWith('yaw_') || r.config === 'baseline_0.0')
      .sort((a, b) => a.yaw_deg - b.yaw_deg);

    // 1. Chart Retention vs Yaw
    new Chart(document.getElementById('chart-retention'), {
      type: 'line',
      data: {
        labels: yawData.map(d => `${d.yaw_deg}°`),
        datasets: [
          {
            label: 'Vật ở gần (≤15m)',
            data: yawData.map(d => d.box_retention_near_pct),
            borderColor: '#10b981',
            backgroundColor: 'rgba(16, 185, 129, 0.1)',
            tension: 0.3,
            fill: false,
          },
          {
            label: 'Vật ở xa (>20m)',
            data: yawData.map(d => d.box_retention_far_pct),
            borderColor: '#f43f5e',
            backgroundColor: 'rgba(244, 63, 94, 0.1)',
            tension: 0.3,
            fill: false,
          },
          {
            label: 'Toàn bộ đối tượng',
            data: yawData.map(d => d.box_retention_all_pct),
            borderColor: '#38bdf8',
            borderDash: [5, 5],
            tension: 0.3,
            fill: false,
          }
        ]
      },
      options: chartDefaultOptions('Tỷ lệ lưu giữ điểm (%)', 0, 110)
    });

    // 2. Chart Shift across axes
    const angles = [0.0, 1.0, 2.0];
    const getShift = (cfgName) => sweep.find(r => r.config === cfgName)?.mean_pixel_shift || 0;
    new Chart(document.getElementById('chart-shift'), {
      type: 'line',
      data: {
        labels: ['0.0°', '1.0°', '2.0°'],
        datasets: [
          {
            label: 'Yaw (Độ trượt ngang)',
            data: [getShift('baseline_0.0'), getShift('yaw_+1.0deg'), getShift('yaw_+2.0deg')],
            borderColor: '#38bdf8',
            tension: 0.2
          },
          {
            label: 'Pitch (Độ trượt đứng)',
            data: [getShift('baseline_0.0'), getShift('pitch_+1.0deg'), getShift('pitch_+2.0deg')],
            borderColor: '#f59e0b',
            tension: 0.2
          },
          {
            label: 'Roll (Xoay quanh quang trục)',
            data: [getShift('baseline_0.0'), getShift('roll_+1.0deg'), getShift('roll_+2.0deg')],
            borderColor: '#a855f7',
            tension: 0.2
          }
        ]
      },
      options: chartDefaultOptions('Độ lệch pixel (px)', 0, 35)
    });

    // 3. Cross-Dataset Chart
    const kittiCross = cross.filter(r => r.dataset?.includes('KITTI') && (r.config.startsWith('yaw_') || r.config === 'baseline_0.0')).sort((a,b)=>a.yaw_deg-b.yaw_deg);
    const nuscCross = cross.filter(r => r.dataset?.includes('nuScenes') && (r.config.startsWith('yaw_') || r.config === 'baseline_0.0')).sort((a,b)=>a.yaw_deg-b.yaw_deg);

    new Chart(document.getElementById('chart-cross'), {
      type: 'line',
      data: {
        labels: kittiCross.map(d => `${d.yaw_deg}°`),
        datasets: [
          {
            label: 'KITTI (64 beam, góc hẹp)',
            data: kittiCross.map(d => d.box_retention_all_pct),
            borderColor: '#0284c7',
            backgroundColor: 'rgba(2, 132, 199, 0.2)',
            tension: 0.2
          },
          {
            label: 'nuScenes (32 beam, góc rộng)',
            data: nuscCross.map(d => d.box_retention_all_pct),
            borderColor: '#ea580c',
            backgroundColor: 'rgba(234, 88, 12, 0.2)',
            tension: 0.2
          }
        ]
      },
      options: chartDefaultOptions('Box Retention (%)', 20, 110)
    });

    // 4. Edge Alignment Chart
    new Chart(document.getElementById('chart-edge'), {
      type: 'line',
      data: {
        labels: yawData.map(d => `${d.yaw_deg}°`),
        datasets: [
          {
            label: 'Edge Alignment Score',
            data: yawData.map(d => d.edge_alignment_score),
            borderColor: '#10b981',
            backgroundColor: 'rgba(16, 185, 129, 0.1)',
            tension: 0.3
          }
        ]
      },
      options: chartDefaultOptions('Score [0 - 1]', 0.34, 0.37)
    });

  } catch (err) {
    console.error('Failed to load benchmark charts data:', err);
  }
}

function chartDefaultOptions(yTitle, yMin, yMax) {
  return {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        labels: { color: '#94a3b8', font: { family: 'Inter', size: 11 } }
      }
    },
    scales: {
      x: {
        grid: { color: 'rgba(255, 255, 255, 0.05)' },
        ticks: { color: '#64748b', font: { family: 'JetBrains Mono' } }
      },
      y: {
        min: yMin,
        max: yMax,
        grid: { color: 'rgba(255, 255, 255, 0.05)' },
        ticks: { color: '#64748b', font: { family: 'JetBrains Mono' } },
        title: {
          display: true,
          text: yTitle,
          color: '#94a3b8',
          font: { family: 'Inter', size: 10 }
        }
      }
    }
  };
}
