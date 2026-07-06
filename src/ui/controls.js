import { listDatasets } from '../data/index.js';
import { MODEL_PRESETS } from '../model/unet.js';

// Renders the control panel. Communicates outward only via callbacks so it stays
// ignorant of data/model/diffusion internals.
//
// createControls(containerEl, { onChange, onAction }) -> {
//   getValues(), setStatus(text), setButtonState({...}), setGenerateEnabled(bool)
// }
export function createControls(containerEl, { onChange, onAction }) {
  const state = {
    datasetId: 'mnist',
    T: 300,
    scheduleType: 'linear',
    betaStart: 1e-4,
    betaEnd: 0.02,
    learningRate: 2e-4,
    batchSize: 64,
    totalSteps: 1500,
    modelSize: 'medium',
    samplingMethod: 'ddim',
    ddimSteps: 50,
  };

  containerEl.classList.add('controls');
  containerEl.innerHTML = '';

  function tip(text) {
    return `<span class="info-icon" tabindex="0" data-tip="${text.replace(/"/g, '&quot;')}">?</span>`;
  }

  const datasets = listDatasets();
  const datasetOptions = datasets
    .map(
      (d) =>
        `<option value="${d.id}" ${d.disabled ? 'disabled' : ''} ${d.id === state.datasetId ? 'selected' : ''}>${d.label}</option>`
    )
    .join('');

  const presetOptions = Object.entries(MODEL_PRESETS)
    .map(([k, v]) => `<option value="${k}" ${k === state.modelSize ? 'selected' : ''}>${v.label} [${v.channels.join(', ')}]</option>`)
    .join('');

  containerEl.innerHTML = `
    <h2>Controls</h2>

    <div class="control-group">
      <h3>Data</h3>
      <label>Dataset${tip('Which image dataset to train on.')}
        <select id="c-dataset">${datasetOptions}</select>
      </label>
    </div>

    <div class="control-group">
      <h3>Diffusion Schedule</h3>
      <label>Timesteps T: <span id="c-T-val">${state.T}</span>${tip('Number of noise steps in the diffusion process. More steps gives smoother, more gradual noising and denoising, but slower sampling.')}
        <input id="c-T" type="range" min="50" max="1000" step="10" value="${state.T}" />
      </label>
      <label>Beta schedule${tip('Controls how much noise is added at each step. Linear adds noise at a steady rate. Cosine adds noise more gently at the start and end of the schedule.')}
        <select id="c-scheduleType">
          <option value="linear" selected>linear</option>
          <option value="cosine">cosine</option>
        </select>
      </label>
      <label>Beta start${tip('The smallest amount of noise added per step. Only used by the linear schedule.')}
        <input id="c-betaStart" type="number" step="0.0001" min="0" value="${state.betaStart}" />
      </label>
      <label>Beta end${tip('The largest amount of noise added per step. Only used by the linear schedule.')}
        <input id="c-betaEnd" type="number" step="0.001" min="0" value="${state.betaEnd}" />
      </label>
    </div>

    <div class="control-group">
      <h3>Model & Training</h3>
      <label>Model size${tip('How large the neural network is. Bigger models can learn more detail but train and generate more slowly.')}
        <select id="c-modelSize">${presetOptions}</select>
      </label>
      <label>Learning rate${tip('How big a step the optimizer takes when updating the model weights. Too high can make training unstable. Too low makes it slow to learn.')}
        <input id="c-learningRate" type="number" step="0.0001" min="0" value="${state.learningRate}" />
      </label>
      <label>Batch size${tip('How many images the model looks at per training step before updating its weights.')}
        <input id="c-batchSize" type="number" step="1" min="1" max="256" value="${state.batchSize}" />
      </label>
      <label>Training steps${tip('Total number of training steps to run.')}
        <input id="c-totalSteps" type="number" step="50" min="50" value="${state.totalSteps}" />
      </label>
    </div>

    <div class="control-group">
      <h3>Sampling</h3>
      <label>Method${tip('DDPM (full ancestral) walks through every reverse step. It is slow but matches the original algorithm exactly. DDIM (fast) skips steps for quicker, deterministic generation.')}
        <select id="c-samplingMethod">
          <option value="ddpm">DDPM (full ancestral)</option>
          <option value="ddim" selected>DDIM (fast)</option>
        </select>
      </label>
      <label>DDIM steps: <span id="c-ddimSteps-val">${state.ddimSteps}</span>${tip('How many reverse steps DDIM uses to generate an image. Fewer steps means faster generation, at some cost to quality.')}
        <input id="c-ddimSteps" type="range" min="10" max="200" step="5" value="${state.ddimSteps}" />
      </label>
    </div>

    <div class="control-group">
      <h3>Run</h3>
      <div class="button-row">
        <button id="c-start">Start</button>
        <button id="c-pause" disabled>Pause</button>
        <button id="c-resume" disabled>Resume</button>
        <button id="c-reset" disabled>Reset</button>
      </div>
      <div class="button-row">
        <button id="c-generate" disabled>Generate Samples</button>
      </div>
      <div id="c-status" class="status">Idle.</div>
    </div>
  `;

  const $ = (id) => containerEl.querySelector(id);

  function emitChange(key) {
    if (onChange) onChange(key, state[key], { ...state });
  }

  // Wire inputs.
  $('#c-dataset').addEventListener('change', (e) => { state.datasetId = e.target.value; emitChange('datasetId'); });

  $('#c-T').addEventListener('input', (e) => {
    state.T = parseInt(e.target.value, 10);
    $('#c-T-val').textContent = state.T;
    emitChange('T');
  });

  $('#c-scheduleType').addEventListener('change', (e) => {
    state.scheduleType = e.target.value;
    updateScheduleEnabled();
    emitChange('scheduleType');
  });

  $('#c-betaStart').addEventListener('change', (e) => { state.betaStart = parseFloat(e.target.value); emitChange('betaStart'); });
  $('#c-betaEnd').addEventListener('change', (e) => { state.betaEnd = parseFloat(e.target.value); emitChange('betaEnd'); });
  $('#c-modelSize').addEventListener('change', (e) => { state.modelSize = e.target.value; emitChange('modelSize'); });
  $('#c-learningRate').addEventListener('change', (e) => { state.learningRate = parseFloat(e.target.value); emitChange('learningRate'); });
  $('#c-batchSize').addEventListener('change', (e) => { state.batchSize = parseInt(e.target.value, 10); emitChange('batchSize'); });
  $('#c-totalSteps').addEventListener('change', (e) => { state.totalSteps = parseInt(e.target.value, 10); emitChange('totalSteps'); });

  $('#c-samplingMethod').addEventListener('change', (e) => {
    state.samplingMethod = e.target.value;
    updateSamplingEnabled();
    emitChange('samplingMethod');
  });
  $('#c-ddimSteps').addEventListener('input', (e) => {
    state.ddimSteps = parseInt(e.target.value, 10);
    $('#c-ddimSteps-val').textContent = state.ddimSteps;
    emitChange('ddimSteps');
  });

  // Buttons.
  $('#c-start').addEventListener('click', () => onAction && onAction('start'));
  $('#c-pause').addEventListener('click', () => onAction && onAction('pause'));
  $('#c-resume').addEventListener('click', () => onAction && onAction('resume'));
  $('#c-reset').addEventListener('click', () => onAction && onAction('reset'));
  $('#c-generate').addEventListener('click', () => onAction && onAction('generate'));

  function updateScheduleEnabled() {
    const linear = state.scheduleType === 'linear';
    $('#c-betaStart').disabled = !linear;
    $('#c-betaEnd').disabled = !linear;
  }
  function updateSamplingEnabled() {
    $('#c-ddimSteps').disabled = state.samplingMethod !== 'ddim';
  }
  updateScheduleEnabled();
  updateSamplingEnabled();

  return {
    getValues: () => ({ ...state }),
    setStatus(text) { $('#c-status').textContent = text; },
    setButtonState({ start, pause, resume, reset } = {}) {
      if (start !== undefined) $('#c-start').disabled = !start;
      if (pause !== undefined) $('#c-pause').disabled = !pause;
      if (resume !== undefined) $('#c-resume').disabled = !resume;
      if (reset !== undefined) $('#c-reset').disabled = !reset;
    },
    setGenerateEnabled(enabled) { $('#c-generate').disabled = !enabled; },
    setConfigEnabled(enabled) {
      const configIds = ['#c-dataset', '#c-T', '#c-scheduleType', '#c-modelSize',
        '#c-learningRate', '#c-batchSize', '#c-totalSteps', '#c-betaStart', '#c-betaEnd'];
      for (const id of configIds) $(id).disabled = !enabled;
      if (enabled) { updateScheduleEnabled(); }
    },
  };
}
