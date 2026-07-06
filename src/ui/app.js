import * as tf from '@tensorflow/tfjs';
import { loadDataset } from '../data/index.js';
import { makeSchedule } from '../diffusion/schedule.js';
import { buildUNet, MODEL_PRESETS } from '../model/unet.js';
import { createTrainer } from '../diffusion/train.js';
import { sample } from '../diffusion/sample.js';
import { createControls } from './controls.js';
import { renderForwardStrip } from './forwardStrip.js';
import { createLossChart } from './lossChart.js';
import { createSampleGrid } from './sampleGrid.js';
import { initTheme } from './theme.js';

const TIME_EMBED_DIM = 64;

// Reads the actual GPU renderer string from a throwaway WebGL context, so we
// can tell the user whether they're getting hardware acceleration or a
// software fallback (e.g. SwiftShader), which can be 10-100x slower.
function getGpuRendererInfo() {
  try {
    const canvas = document.createElement('canvas');
    const gl = canvas.getContext('webgl2') || canvas.getContext('webgl');
    if (!gl) return null;
    const dbg = gl.getExtension('WEBGL_debug_renderer_info');
    return dbg ? gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER);
  } catch {
    return null;
  }
}

export function mountApp(rootEl) {
  rootEl.innerHTML = `
    <header class="app-header">
      <div class="header-row">
        <div>
          <h1>DDPM Playground</h1>
          <p class="subtitle">Train a tiny diffusion model in your browser. Forward noising, live training, and reverse sampling.</p>
        </div>
        <label class="theme-picker">Theme
          <select id="theme-select">
            <option value="dark">Dark</option>
            <option value="light">Light</option>
            <option value="system">System</option>
          </select>
        </label>
      </div>
    </header>
    <div class="app-body">
      <aside id="controls-panel"></aside>
      <main class="viz-panels">
        <section class="panel">
          <h2>Forward Noising</h2>
          <p class="hint">This shows one example image from the dataset, always the same one, progressively corrupted with noise using the schedule above. It is pure math from the noise schedule: no model is involved yet. This example image is fixed and only illustrates how noise is added over time; it has no connection to what the model generates below.</p>
          <div id="forward-strip" class="forward-strip"></div>
        </section>
        <section class="panel">
          <h2>Training Loss</h2>
          <p class="hint">How well the model predicts the noise added to each training batch. Lower is better.</p>
          <canvas id="loss-chart" class="loss-chart"></canvas>
        </section>
        <section class="panel">
          <h2>Generated Samples</h2>
          <p class="hint">These images come from running the trained model backwards (inference): starting from random noise and removing a little noise at each step until a new image appears. A quick, low quality preview refreshes automatically during training. Click "Generate Samples" anytime for a fresh, full quality batch using the sampling method and step count selected in the panel on the left. This model is unconditional: it was never told to draw a particular digit or class, only trained to match the overall distribution of the dataset. Each of the 16 samples below is an independent random draw and can be any class the dataset contains, so do not expect them to match the example shown in Forward Noising above. The grid shows 16 samples at once so you can compare variety and quality together, not because there are 16 classes.</p>
          <div id="sample-grid"></div>
        </section>
      </main>
    </div>
    <div id="backend-note" class="backend-note"></div>
  `;

  initTheme(document.getElementById('theme-select'));

  const state = {
    datasetId: 'mnist',
    dataset: null,
    sampleImage: null,
    schedule: null,
    model: null,
    modelSize: 'medium',
    trainer: null,
    trainingStatus: 'idle', // idle | running | paused | done
    samplingMethod: 'ddim',
    ddimSteps: 50,
    hasCheckpoint: false,
    isSampling: false,
    modelDirty: true,
  };

  const controls = createControls(document.getElementById('controls-panel'), {
    onChange: handleChange,
    onAction: handleAction,
  });
  const lossChart = createLossChart(document.getElementById('loss-chart'));
  const sampleGrid = createSampleGrid(document.getElementById('sample-grid'), { rows: 4, cols: 4 });
  const forwardStripEl = document.getElementById('forward-strip');

  // ---- backend init ----
  (async () => {
    try {
      await tf.ready();
      try { await tf.setBackend('webgl'); } catch { /* fall back to whatever loaded */ }
      const backend = tf.getBackend();
      const gpu = backend === 'webgl' ? getGpuRendererInfo() : null;
      let note = `TensorFlow.js backend: ${backend}`;
      if (gpu) {
        note += `. GPU: ${gpu}`;
        if (/swiftshader|software|llvmpipe/i.test(gpu)) {
          note += '. This looks like software rendering, not a real GPU. Enable hardware acceleration in your browser settings for much faster training.';
        }
      }
      document.getElementById('backend-note').textContent = note;
    } catch (e) {
      document.getElementById('backend-note').textContent = `Backend error: ${e.message}`;
    }
    await initDatasetAndSchedule();
    sampleGrid.showNoise();
  })();

  // -------------------------------------------------------------------------
  function currentValues() { return controls.getValues(); }

  function disposeSampleImage() {
    if (state.sampleImage) { state.sampleImage.dispose(); state.sampleImage = null; }
  }

  async function initDatasetAndSchedule() {
    const v = currentValues();
    controls.setStatus('Loading dataset…');
    try {
      state.dataset = await loadDataset(v.datasetId);
      state.datasetId = v.datasetId;
    } catch (e) {
      controls.setStatus(`Dataset load failed: ${e.message}`);
      return;
    }
    disposeSampleImage();
    state.sampleImage = tf.tidy(() =>
      state.dataset.images.slice([0, 0, 0, 0], [1, -1, -1, -1]).reshape([28, 28, 1])
    );
    rebuildSchedule();
    controls.setStatus(`Loaded ${state.dataset.manifest.count} images from ${v.datasetId}.`);
  }

  function rebuildSchedule() {
    const v = currentValues();
    state.schedule = makeSchedule({
      T: v.T,
      scheduleType: v.scheduleType,
      betaStart: v.betaStart,
      betaEnd: v.betaEnd,
    });
    if (state.sampleImage) {
      renderForwardStrip(forwardStripEl, { image: state.sampleImage, schedule: state.schedule, numFrames: 10 });
    }
  }

  function rebuildModel() {
    if (state.model) { state.model.dispose(); state.model = null; }
    const preset = MODEL_PRESETS[currentValues().modelSize] || MODEL_PRESETS.medium;
    state.model = buildUNet({ imageSize: 28, channels: preset.channels, timeEmbedDim: TIME_EMBED_DIM });
    state.modelDirty = false;
  }

  function isBusy() {
    return state.trainingStatus === 'running' || state.trainingStatus === 'paused';
  }

  // -------------------------------------------------------------------------
  async function handleChange(key, value) {
    if (isBusy()) {
      // Config is disabled in the DOM while busy, but guard anyway.
      controls.setStatus('Pause and Reset before changing configuration.');
      return;
    }
    if (key === 'datasetId') {
      await initDatasetAndSchedule();
      state.modelDirty = true; // fresh run recommended
    } else if (['T', 'scheduleType', 'betaStart', 'betaEnd'].includes(key)) {
      rebuildSchedule();
    } else if (key === 'modelSize') {
      state.modelDirty = true;
      controls.setStatus('Model size changed. It will rebuild on Start.');
    } else if (key === 'samplingMethod') {
      state.samplingMethod = value;
    } else if (key === 'ddimSteps') {
      state.ddimSteps = value;
    }
  }

  async function handleAction(action) {
    if (action === 'start') return startTraining();
    if (action === 'pause') return pauseTraining();
    if (action === 'resume') return resumeTraining();
    if (action === 'reset') return resetTraining();
    if (action === 'generate') return generateSamples();
  }

  // -------------------------------------------------------------------------
  async function startTraining() {
    if (isBusy()) return;
    if (!state.dataset) { controls.setStatus('Dataset not loaded yet.'); return; }

    // Ensure schedule + model fresh.
    if (!state.schedule) rebuildSchedule();
    if (state.modelDirty || !state.model) rebuildModel();

    lossChart.reset();
    const v = currentValues();

    state.trainer = createTrainer({
      model: state.model,
      schedule: state.schedule,
      images: state.dataset.images,
      config: {
        learningRate: v.learningRate,
        batchSize: v.batchSize,
        totalSteps: v.totalSteps,
        sampleEveryNSteps: Math.max(50, Math.round(v.totalSteps / 8)),
        timeEmbedDim: TIME_EMBED_DIM,
      },
    });

    state.trainer.onStep((step, loss, total) => {
      lossChart.push(step, loss);
      if (step % 10 === 0) {
        controls.setStatus(`Training… step ${step + 1}/${total}  loss ${loss.toFixed(4)}`);
      }
      if (step + 1 >= total) {
        state.trainingStatus = 'done';
        controls.setStatus(`Training complete (${total} steps).`);
        syncButtons();
      }
    });

    state.trainer.onSampleCheckpoint((step) => {
      state.hasCheckpoint = true;
      controls.setGenerateEnabled(true);
      runPreview();
    });

    state.trainingStatus = 'running';
    controls.setConfigEnabled(false);
    syncButtons();
    controls.setStatus('Compiling model on GPU (first step can take a while on some devices)…');
    state.trainer.start();
  }

  function pauseTraining() {
    if (state.trainingStatus !== 'running') return;
    state.trainer.pause();
    state.trainingStatus = 'paused';
    controls.setStatus('Paused.');
    syncButtons();
  }

  function resumeTraining() {
    if (state.trainingStatus !== 'paused') return;
    state.trainer.resume();
    state.trainingStatus = 'running';
    controls.setStatus('Resumed.');
    syncButtons();
  }

  function resetTraining() {
    if (state.trainer) { state.trainer.dispose(); state.trainer = null; }
    // Dispose the model so the next run starts from fresh weights and no WebGL
    // memory piles up across repeated resets.
    if (state.model) { state.model.dispose(); state.model = null; }
    state.modelDirty = true;
    state.trainingStatus = 'idle';
    state.hasCheckpoint = false;
    lossChart.reset();
    sampleGrid.showNoise();
    controls.setGenerateEnabled(false);
    controls.setConfigEnabled(true);
    syncButtons();
    controls.setStatus('Reset. Ready for a new run.');
  }

  function syncButtons() {
    const s = state.trainingStatus;
    controls.setButtonState({
      start: s === 'idle' || s === 'done',
      pause: s === 'running',
      resume: s === 'paused',
      reset: s !== 'idle',
    });
  }

  // -------------------------------------------------------------------------
  async function runPreview() {
    if (state.isSampling || !state.model) return;
    state.isSampling = true;
    try {
      const out = await sample({
        method: 'ddim',
        model: state.model,
        schedule: state.schedule,
        numSamples: 16,
        imageSize: 28,
        ddimSteps: 20,
        eta: 0,
        timeEmbedDim: TIME_EMBED_DIM,
      });
      await sampleGrid.updateFromTensor(out);
      out.dispose();
    } catch (e) {
      console.error('preview failed', e);
    } finally {
      state.isSampling = false;
    }
  }

  async function generateSamples() {
    if (!state.hasCheckpoint || !state.model) {
      controls.setStatus('Train at least one checkpoint before generating.');
      return;
    }
    if (state.isSampling) return;
    state.isSampling = true;
    const v = currentValues();
    controls.setGenerateEnabled(false);
    sampleGrid.showNoise();
    controls.setStatus(`Running reverse diffusion (inference) with ${v.samplingMethod.toUpperCase()}…`);
    try {
      const out = await sample({
        method: v.samplingMethod,
        model: state.model,
        schedule: state.schedule,
        numSamples: 16,
        imageSize: 28,
        ddimSteps: v.ddimSteps,
        eta: 0,
        timeEmbedDim: TIME_EMBED_DIM,
        onStep: async (t, xt) => {
          await sampleGrid.updateFromTensor(xt);
        },
      });
      await sampleGrid.updateFromTensor(out);
      out.dispose();
      controls.setStatus(`Done generating (${v.samplingMethod.toUpperCase()}).`);
    } catch (e) {
      console.error(e);
      controls.setStatus(`Sampling failed: ${e.message}`);
    } finally {
      state.isSampling = false;
      controls.setGenerateEnabled(true);
    }
  }
}
